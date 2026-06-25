# PPO on LunarLanderContinuous-v3: Design and Sweep Results

> Created on: 12 June 2026
>
> Updated on: 24 June 2026

This note documents a from-scratch PyTorch implementation of Proximal Policy Optimisation (PPO) that I built for the `LunarLanderContinuous-v3` environment from [Gymnasium](https://gymnasium.farama.org/), together with an empirical analysis of a 12-configuration hyperparameter sweep I ran on it.

The note is in two parts. **Part I** (Sections [1](#1-why-two-scripts-for-one-algorithm)–[5](#5-environment-and-reward-decisions)) covers the implementation and the design decisions behind it, including the trade-offs that were considered and the reasons for the chosen path. **Part II** (Sections [6](#6-experimental-setup)–[9](#9-empirical-summary)) reports what the sweep actually produced: which configurations solved the task, what drove the difference, and what the training curves reveal about PPO's failure modes on this environment. [Section 10](#10-what-i-would-extend-next) closes with planned extensions.

The implementation is split across two scripts, `ppo_lunarlander_tensorboard.py` and `ppo_lunarlander_wandb.py`, which share their algorithmic core verbatim and differ only in the observability layer. The full source code can be found on [GitHub](https://github.com/nhan-dam/rl-foundations/tree/main/src).

---

**Part I: Implementation and Design**

---

## 1. Why Two Scripts for One Algorithm

The two scripts implement the same PPO algorithm, with the same `TrainingConfig` dataclass, the same `ActorCritic` network, the same `RolloutBuffer`, the same loss function, the same training loop, and the same parallel sweep harness. The only point of divergence is the logger. `ppo_lunarlander_tensorboard.py` uses `torch.utils.tensorboard.SummaryWriter`. `ppo_lunarlander_wandb.py` uses `wandb.init` and `wandb.log`.

The motivation for splitting along this axis rather than collapsing both loggers behind a common abstraction is pedagogical. A side-by-side diff of the two files isolates exactly the surface area that an observability library touches: writer construction, scalar emission per update, metric-axis declaration, sweep grouping, and shutdown. Hiding this behind an interface would obscure the very contrast the note is meant to demonstrate.

A combined design note, rather than two separate ones, is the right choice for the same reason. The algorithmic and engineering content does not vary across the two files. Presenting it once and then framing the observability layer as the controlled variable produces a clearer artefact than two near-duplicate write-ups.

---

## 2. PPO Algorithmic Decisions

### 2.1. Shared-Trunk Actor-Critic With Diagonal Gaussian Head

The `ActorCritic` network is a two-hidden-layer multilayer perceptron (MLP) with Tanh activations. A shared trunk feeds two heads. The actor head outputs the per-dimension mean $\mu$ of a diagonal Gaussian policy. The critic head outputs a scalar state value $V(s)$. The log standard deviation $\log\sigma$ is a state-independent learnable parameter, registered as `nn.Parameter` rather than produced by the network.

Three design decisions are worth noting.

First, the trunk is shared. The alternative is two independent networks. A shared trunk halves the parameter count and forces the actor and critic to develop a common feature representation, which is the convention in continuous-control PPO benchmarks. The trade-off is gradient interference, i.e. the policy and value losses can pull the trunk in different directions. The standard remedy is the value-loss coefficient $c_1 = 0.5$, which keeps the value gradient from dominating, and is what this implementation uses.

Second, the Gaussian is diagonal. A full covariance matrix introduces $O(d^2)$ parameters and offers no measurable benefit on benchmark continuous-control tasks because the action dimensions are weakly coupled at the policy level. A diagonal Gaussian also makes the action log-probability separable across dimensions, computed as $\sum_i \log p(a_i \mid s)$, which is what `Normal(mean, std).log_prob(action).sum(dim=-1)` evaluates.

Third, $\log\sigma$ is state-independent. Making it a function of the state (i.e. another network head) is supported in principle but empirically tends to destabilise training: the policy can collapse to a near-deterministic distribution in some states and a high-variance one in others, and the resulting advantage estimator becomes very high-variance. A state-independent learnable scalar per action dimension is the standard PPO choice and works well here.

Activations are Tanh rather than ReLU. PPO reference implementations almost universally use Tanh because it bounds the pre-softmax (or pre-Gaussian-mean) logits, which is empirically more stable for policy gradient methods that are sensitive to scale.

### 2.2. On-Policy Rollout Buffer

The `RolloutBuffer` is fixed-length (2048 steps), filled once per outer iteration, drained by `compute_gae` and `update`, and reset. The contrast with the off-policy replay buffer used in the deep Q-network (DQN) implementation is the central reason the buffer is structured this way.

In DQN, transitions are reusable across many gradient updates because Q-learning is off-policy: the Bellman target $r + \gamma \max_{a'} Q(s', a')$ does not depend on the policy that generated $(s, a, r, s')$. In PPO, the surrogate loss depends on the importance-sampling ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)},$$

which is only meaningful when $\pi_{\theta_\text{old}}$ is the policy that actually produced the action. After K epochs of updates, the ratio between the current and old policy is still bounded by the clip, but the samples themselves are no longer i.i.d. draws under a coherent old policy. They are discarded.

The rollout length of 2048 is chosen because it divides cleanly into mini-batches of 64 (giving 32 mini-batches per epoch and 320 gradient steps per rollout at K=10) and matches the default in reference implementations such as Stable Baselines3. The mini-batch split is delegated to a PyTorch `DataLoader` (`shuffle=True`, `drop_last=False`), so an indivisible rollout is handled gracefully: the final batch is simply smaller rather than dropped. With the default 2048/64 the division is exact and every batch is full, but keeping the remainder is the better default for on-policy data, since it avoids discarding freshly collected samples.

### 2.3. Generalised Advantage Estimation

Advantages are computed via Generalised Advantage Estimation (GAE) (Schulman et al., 2015), defined as the exponentially weighted sum of $n$-step temporal-difference residuals,

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

For implementation, GAE is unrolled backwards from the rollout's final step using the recurrence

$$\hat{A}_t = \delta_t + \gamma\lambda (1 - \text{done}_{t+1}) \hat{A}_{t+1},$$

which avoids the explicit infinite sum. The bootstrap value $V(s_T)$ for the state after the final stored transition is computed via `get_value`, and is used only if the rollout did not end on a terminal. The returns used as the value-function regression target are $\hat{R}_t = \hat{A}_t + V(s_t)$, which is equivalent to the $\lambda$-return and is the standard PPO choice.

The $\lambda = 0.95$ default interpolates between high-bias low-variance ($\lambda = 0$, pure one-step temporal difference) and low-bias high-variance ($\lambda = 1$, pure Monte Carlo). It is one of the few PPO hyperparameters that essentially never needs tuning, so it is not exposed as a sweep dimension.

### 2.4. Terminated vs Truncated

The Gymnasium step function returns five values, two of which signal episode end: `terminated` and `truncated`. `terminated` is True only when the environment has reached a genuine terminal state, e.g. the lander has crashed or landed. `truncated` is True when the episode ended for an external reason, typically a step-count time limit.

The two flags require different treatment in GAE. A terminal state has no successor, so the bootstrap $V(s_{t+1})$ should be zeroed. A time-limit truncation leaves the underlying state valid, and the value of that state is still a meaningful estimate of expected future return. Zeroing the bootstrap on a truncation biases the return estimate downward at every time-limit boundary.

The implementation folds the truncation bootstrap into the stored reward: when a step ends with `truncated` and not `terminated`, the critic's value of the post-truncation state is added to that step's reward as $r_t \leftarrow r_t + \gamma V(s_{t+1})$, and the step is then stored as a hard episode boundary (`done = True`). GAE thereafter treats every stored boundary uniformly, i.e. the bootstrap term is zeroed at all of them, yet the truncated step still carries its continuation value through the reward. This is the mechanism Stable Baselines3 uses (`handle_timeout_termination`), and it also handles the corner case where a truncation lands exactly on the rollout's final step: the bootstrap is already in the reward, so the end-of-rollout bootstrap can simply be zeroed whenever the last transition closed an episode.

A simpler approach, storing `terminated` directly in the done flag, is subtly wrong: it leaves a mid-rollout truncation unmarked, so GAE bootstraps the truncated step from the next stored value, which belongs to the freshly reset episode rather than the post-truncation state. Folding the continuation value into the reward avoids this.

### 2.5. Clipped Surrogate Loss

The PPO objective combines three terms,

$$\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 H[\pi_\theta],$$

where

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat{A}_t\right)\right],$$

and $\mathcal{L}^{\text{VF}} = \mathbb{E}_t[(V_\theta(s_t) - \hat{R}_t)^2]$ is the value mean squared error. The implementation flips the sign on $\mathcal{L}^{\text{CLIP}}$ and $H$ for gradient descent, since PyTorch minimises.

The clip range $\varepsilon = 0.2$ is the standard default. Two further design points are worth flagging.

First, advantages are standardised to zero mean and unit variance (commonly called advantage normalisation in PPO) at the rollout level, once, before any of the K epochs begin, not per mini-batch. Per-mini-batch normalisation is sometimes seen in the wild and works empirically, but it changes the optimisation target each mini-batch (the normalisation statistics shift between mini-batches), which is theoretically less clean. Normalising once over the full rollout keeps the target stationary across the K epochs.

Second, the value function is not clipped. Some PPO implementations clip the value update analogously to the policy update,

$$\mathcal{L}^{\text{VF},\text{clip}} = \max\left((V_\theta(s_t) - \hat{R}_t)^2,\ (\text{clip}(V_\theta(s_t), V_{\theta_\text{old}}(s_t) \pm \varepsilon) - \hat{R}_t)^2\right),$$

to prevent large value updates from destabilising the policy via shared-trunk gradients. On LunarLander this is unnecessary because the reward magnitudes are moderate and the value loss is already coefficient-weighted. Including it without need adds complexity and a hyperparameter without measurable benefit.

### 2.6. K-Epoch Mini-Batch Update

Each outer iteration runs K=10 PPO epochs over the rollout. Within each epoch, a PyTorch `DataLoader` (`shuffle=True`) reshuffles the rollout and yields non-overlapping mini-batches of size 64. This guarantees that every sample is visited exactly K times across the K epochs, rather than K times in expectation under with-replacement sampling. The deterministic visitation matters because PPO's clip can lock out updates entirely on samples that have already drifted far in earlier epochs of the same rollout. A noisy visitation pattern compounds that effect.

### 2.7. Gradient Norm Clipping

The combined gradient is clipped to L2 norm 0.5 via `nn.utils.clip_grad_norm_`. This is the standard PPO safety net. It complements the surrogate clip, which bounds the per-sample objective contribution, by additionally bounding the aggregate parameter update. In practice it rarely fires when the rest of the hyperparameters are sensibly chosen, but the cost of including it is negligible and it cleanly catches outlier mini-batches.

---

## 3. Engineering Scaffolding

### 3.1. TrainingConfig Dataclass

All hyperparameters live on a single typed dataclass with field-level defaults and a hashed `label` property derived from the swept fields. The label gives every config a stable identifier that is used as the matplotlib output filename and the TensorBoard subdirectory; the wandb variant additionally derives a human-readable `run_name` from the same swept fields. This makes sweep artefacts trivially traceable back to the originating config, even after several launches.

### 3.2. Agent Decoupled From Config

`PPOAgent.__init__` takes individual parameters rather than a `TrainingConfig` instance. Each dependency is therefore explicit in the signature, the agent is reusable in contexts that do not construct a `TrainingConfig` (e.g. inference, ad hoc experiments), and the agent does not pin to a single config schema. The cost is a slightly verbose constructor call inside `train`, where the dataclass is unpacked field by field. This is a deliberate trade.

The same principle is applied to the environment. Rather than receiving the `gym.Env` object, the agent takes the four primitives it actually needs, i.e. `obs_dim`, `action_dim`, `action_low`, and `action_high`, extracted by the caller. The agent therefore has no dependency on the `gym.Env` interface at all, which makes it trivially constructible in tests and inference paths without standing up an environment. The wrapped env is still produced and returned by `train` (its `RecordEpisodeStatistics` queues drive the plots, and the caller closes it), but it is no longer threaded through the agent.

### 3.3. Best-Checkpoint Selection With Mid-Training Evaluation

Mid-training evaluation probes fire every `eval_interval_steps` environment steps. Each probe runs 20 deterministic evaluation episodes and reports mean, max, and standard deviation of return. The agent's `actor_critic.state_dict()` is deep-copied whenever the probe's mean exceeds the running best, with the standard deviation acting as a tiebreaker on near-equal means. After the training loop completes, the best weights are restored.

Means are compared with a small tolerance, $\frac{0.5}{n_{\text{eval}}}$: two probes whose means fall within it are treated as tied and the lower standard deviation wins. With only 20 episodes per probe, the sampling error on the mean far exceeds a gap this small, so preferring the more consistent policy at effectively equal means is the more robust selection rule than letting noise pick the winner. [Section 8.4](#84-late-training-degradation-and-the-value-of-best-checkpoint-selection) quantifies how much return this mechanism recovers in practice.

### 3.4. Deterministic Evaluation

`test_agent` uses `dist.mean` rather than `dist.sample`. This removes sampling variance from the score and reports the quality of the learnt mean policy. The justification is analogous to the use of $\varepsilon = 0.05$ rather than $\varepsilon = 0$ at evaluation time in DQN (Mnih et al., 2015): a deterministic readout of a stochastic-policy network is a more honest measure of what the agent has learnt than a single noisy sample.

### 3.5. Seeding and Reproducibility

`train` seeds PyTorch, NumPy's global generator, and the environment's initial reset from `config.seed` before the agent is constructed, so weight initialisation, mini-batch shuffling, action sampling, and the initial state sequence are all reproducible for a given config. Evaluation episodes are deliberately left unseeded: the mid-training probes estimate performance over the environment's initial-condition distribution, and pinning them to a fixed sequence would understate exactly the variance the probe exists to measure.

### 3.6. Parallel Sweep With Multiprocessing

The sweep is a Cartesian product over learning rate, clip $\varepsilon$, and entropy coefficient, evaluated in a `multiprocessing.Pool` with the `spawn` start method. Three points are worth flagging.

First, `spawn` is mandatory on macOS. Fork is unsafe with PyTorch and with the Objective-C runtime that underlies macOS GUI processes. The `spawn` start method is set explicitly before any pool is created.

Second, the worker function `run_one_config` is at module scope. The `spawn` start method launches a fresh interpreter that inherits none of the parent's memory, so the worker is shipped to it via `pickle`. Pickling a function serialises only a reference to it (its module and qualified name), which the child re-imports by name, so the function must be importable as a top-level module attribute. Nested functions and lambdas have no such importable name and therefore cannot be pickled, which is why module-level functions are the standard way to satisfy this constraint.

Third, workers are pinned to CPU. PPO on a small MLP is CPU-bound: the rollout collection loop is a Python-level for loop that the GPU cannot accelerate, and the per-mini-batch forward pass on a 64-wide hidden layer is negligible. Pinning to CPU avoids the GPU contention that would otherwise occur on a multi-config sweep. The same reasoning extends to automatic device selection outside the sweep: `_select_device` prefers CUDA when present but never auto-selects MPS, because on Apple silicon the per-op launch overhead and the per-step CPU-to-GPU transfer in the rollout loop make MPS slower than CPU at this model scale. MPS remains available by passing it explicitly to `train`.

---

## 4. The Observability Axis: TensorBoard vs Weights and Biases

### 4.1. What Is Shared

Both scripts expose the same metric set. Per-update scalars (logged inside `PPOAgent.update`) are `train/policy_loss`, `train/value_loss`, `train/entropy`, and `train/clip_fraction`. Per-iteration scalars (logged in `train`) are `train/episode_return` (one entry per completed episode) and the eval probes' `eval/mean_return`, `eval/max_return`, `eval/std_return`. Two further scalars, `eval/train_clip_fraction` and `eval/train_entropy`, are logged in the same block but are not products of the evaluation episodes: they are the recent (last-100) average of the *training* clip fraction and entropy from `update`, snapshotted at the eval cadence so they share the env-step axis with the eval returns for side-by-side comparison. The `train` prefix in their leaf names signals this origin, while the `eval/` namespace keeps them grouped with the eval curves. Both scripts open the logger inside `train`, hand the run handle to the agent via a typed attribute (`PPOAgent.writer` or `PPOAgent.wandb_run`), and close the handle in a `finally` block so the logs flush even if training raises.

The matplotlib plots and the sweep results CSV are also unchanged across the two scripts, providing a local-only audit trail independent of either live logger.

One sizing detail underpins the episode-return stream in both scripts. `RecordEpisodeStatistics` keeps completed-episode returns in a bounded deque, and the logging drain reads only the unseen tail of that deque each iteration. If the deque ever evicts an entry before it has been drained, the tail indexing breaks and logging silently stops. The buffer is therefore sized to `total_steps`: an episode consumes at least one environment step, so the episode count can never exceed it and eviction can never occur.

### 4.2. TensorBoard Variant

Each `train` call constructs a `SummaryWriter` at `./results/lunarlander_ppo_tensorboard/tb/<config_label>/`. The label-based subdirectory means that running `tensorboard --logdir ./results/lunarlander_ppo_tensorboard/tb` aggregates every config in the sweep into a single comparison view, with no further configuration. There is no account, no network dependency, and no per-launch coordination required.

### 4.3. Weights and Biases Variant

Each `train` call opens a fresh `wandb.run`, passes the hyperparameter set via `config=`, and is always closed via `run.finish()` in a `finally` block, so a pool worker that trains several configs starts each from a clean run. `define_metric` declares two x-axes: `train/update_step` for the per-update scalars, and `env/step` for `train/episode_return` and all `eval/*` scalars (the standard x-axis for reinforcement learning (RL) learning curves). Both counters are logged as ordinary values, and no `wandb.log` call passes an explicit `step=`. This last point is load-bearing: wandb's global step must increase monotonically, and the per-update, per-episode, and eval streams interleave. Passing an explicit `step=` on one stream while others advance the global counter implicitly makes wandb silently drop every subsequent log whose explicit step falls behind, which hollows out the training curves without raising any error.

Three further pieces of glue are worth noting.

First, `wandb.Settings(start_method="thread")` is set explicitly. Wandb runs an internal IO thread, and the default `fork` start method for that thread clashes with the outer `multiprocessing` workers running under `spawn`. The documented fix is to force the wandb-internal thread to use `thread` rather than `fork`. This is one of the few wandb-specific gotchas that is not discoverable from the API alone.

Second, every sweep config is assigned a common `wandb_group = "sweep-<timestamp>"`, set in `__main__` and threaded through the `TrainingConfig`. This is what makes the wandb UI's Group filter pull a single launch's runs together, which is the primary value proposition of wandb over TensorBoard for this kind of work.

Third, `run.summary` is populated with `best/mean_return`, `best/max_return`, `best/std_return`, `best/step`, and `first_threshold_step` before `run.finish`. Summary scalars appear in the sortable run table at the project level, where they double as the natural columns for cross-config comparison. Without this, the final sweep table would only show the wandb-default columns (run name, runtime, last-logged value of each metric), which are not what one wants to sort by.

### 4.4. Side-by-Side Comparison

| Aspect | TensorBoard | Weights and Biases |
|---|---|---|
| Setup cost | None beyond the dependency. | `wandb login` once, account required. |
| Per-run output | Local TFEvent file. | Cloud-hosted run, plus local cache. |
| Cross-run comparison | Aggregated by `--logdir parent/`, view-level config. | First-class UI primitive, persistent. |
| Hyperparameter dashboard | HParams plugin, clunky. | Native, via `wandb.config`. |
| System metric autocapture | None. | CPU, GPU, MPS, RAM. |
| Sweep grouping | Subdirectory convention. | `wandb_group` field, queryable. |
| Sharing | Local folder, requires the recipient to run TensorBoard. | URL. |
| Air-gapped operation | Native. | `mode="offline"`, `wandb sync` later. |
| Vendor risk | None. | External dependency, account, quota. |

The choice between the two is not a quality judgement. It is a deployment decision. TensorBoard fits when local artefacts are preferred, when the work is not shareable, or when the run environment lacks network. Wandb fits when sweep comparison and cross-machine persistence are the primary concerns. The split into two files allows both deployment modes to coexist in the same repository without an abstraction layer obscuring either. The results in Part II were collected with the TensorBoard variant.

---

## 5. Environment and Reward Decisions

### 5.1. No Entropy Bonus by Default

`entropy_coef = 0.0` is the default. The reasoning is that the diagonal Gaussian policy with learnable $\log\sigma$ already has a natural exploration mechanism, so LunarLander was expected to solve without an explicit entropy bonus. The bonus is included as a sweep dimension (0.0 vs 0.01) to test this empirically rather than carry it as an article of faith. Section [7.2](#72-the-entropy-bonus-is-the-dominant-reliability-driver) reports the outcome, which refines the expectation: the bonus is not strictly necessary for a peak score, but it materially de-risks the search.

### 5.2. A Densely Shaped Reward

LunarLander's reward is densely *shaped*: every step returns a sum of intermediate terms rather than a single end-of-episode payoff. The agent is rewarded for moving towards the landing pad and slowing down, for staying upright, and for each leg that makes ground contact; it is penalised a small amount for every main- or side-engine firing (a fuel cost); and the episode closes with a large terminal bonus for a safe landing or penalty for a crash (of order $\pm 100$). The exact coefficients are in the Gymnasium documentation. This shaping is why the task is learnable in roughly a million steps and why no entropy bonus is strictly needed: the policy receives a corrective gradient at every timestep about whether its last action helped.

This is worth contrasting with the other two environments used elsewhere in this repository, because the reward structure largely dictates which algorithm is appropriate. CartPole's reward is dense but *unshaped*: a constant $+1$ for every timestep the pole stays balanced, with no term for how centred or vertical the system is. The signal arrives every step but only distinguishes 'alive' from 'done', so the return is simply the number of steps survived. Blackjack sits at the opposite extreme with a *sparse* reward: zero on every intermediate step and a single terminal payout of $+1$ for a win, $-1$ for a loss, $0$ for a draw (optionally $+1.5$ for a natural). This is the hardest credit-assignment case of the three, since a whole sequence of hit/stick decisions is judged only by its final outcome. The progression from Blackjack's sparse terminal reward through CartPole's flat per-step signal to LunarLander's rich shaping mirrors the progression from tabular methods, through simple value-based or policy-gradient methods, to the full actor-critic PPO implemented here.

---

**Part II: Empirical Results**

---

## 6. Experimental Setup

The sweep evaluates the PPO agent on `LunarLanderContinuous-v3` across a $3 \times 2 \times 2 = 12$ config grid: three learning rates ($10^{-4}$, $3 \times 10^{-4}$, $10^{-3}$), two clip ranges ($\varepsilon = 0.1, 0.2$), and two entropy coefficients ($c_2 = 0.0, 0.01$). Each run trains for 1,000,000 environment steps. The solve criterion is the canonical LunarLander bar: the best checkpoint's mean return over a 100-episode deterministic evaluation is at least 200. The 100-episode evaluation is the reliable signal here, because the mid-training probes average only 20 episodes each and are too noisy to decide solve status on their own.

The mid-training probes still serve two purposes. They drive best-checkpoint selection ([Section 3.3](#33-best-checkpoint-selection-with-mid-training-evaluation)), and they feed a coarse time-to-threshold diagnostic, `first_threshold_step`, i.e. the first probe step whose 20-episode mean exceeds 200. This diagnostic is reported alongside the results but is not the solve criterion, and a config can solve on the 100-episode evaluation without any single probe having crossed 200.

All other hyperparameters are held fixed at the defaults in [Section 2](#2-ppo-algorithmic-decisions): discount $\gamma = 0.99$, GAE $\lambda = 0.95$, rollout length 2048, K=10 epochs, mini-batch size 64, hidden width 64, value coefficient $c_1 = 0.5$, and gradient-norm clip 0.5. Probes fire every 50,000 environment steps (20 probes per run). After training, the best checkpoint is restored and scored over 100 deterministic episodes, and it is this 100-episode mean and standard deviation that [Table 1](#tab-sweep) reports. The 12 configs run concurrently in a CPU-pinned `multiprocessing.Pool`.

---

## 7. Sweep Results

### 7.1. Overall Outcome

Nine of the twelve configs satisfy the solve criterion. [Table 1](#tab-sweep) lists all twelve, sorted by the best checkpoint's 100-episode mean return. The three failures are not catastrophic: each lands the craft on a good fraction of episodes, reaching a best single-episode return near 290–300 (the `100-ep max` column), barely below the solved configs. What sinks them is consistency, not capability: their best checkpoint averages below 200 over the 100-episode evaluation (174.2, 164.8, and 154.9) and carries a standard deviation above 110, more than four times that of the strongest solved configs. The headline question is therefore not whether PPO can land the craft, but what separates a reliable config from a volatile one.

<a id="tab-sweep"></a>

| Config | Learning rate | $\varepsilon_{\text{clip}}$ | $c_2$ | Solved | First-threshold step | 100-ep mean | 100-ep max | 100-ep std |
|---|---|---|---|---|---|---|---|---|
| `cbeefb0b` | $3 \times 10^{-4}$ | 0.2 | 0.0 | <span style="color: green;">✓</span> | 204,800 | <span style="color: green;">280.2</span> | 312.6 | <span style="color: green;">18.7</span> |
| `b97050c1` | $3 \times 10^{-4}$ | 0.1 | 0.01 | <span style="color: green;">✓</span> | 409,600 | 278.3 | 319.3 | 22.5 |
| `cccccebc` | $1 \times 10^{-4}$ | 0.2 | 0.01 | <span style="color: green;">✓</span> | 512,000 | 268.5 | 316.1 | 43.9 |
| `56f68680` | $1 \times 10^{-4}$ | 0.2 | 0.0 | <span style="color: green;">✓</span> | 563,200 | 259.3 | 311.9 | 55.5 |
| `fb20fcb3` | $3 \times 10^{-4}$ | 0.2 | 0.01 | <span style="color: green;">✓</span> | 153,600 | 257.7 | 313.0 | 64.3 |
| `6e1b8a66` | $1 \times 10^{-3}$ | 0.2 | 0.01 | <span style="color: green;">✓</span> | <span style="color: green;">102,400</span> | 253.3 | 310.5 | 63.1 |
| `c509e60a` | $1 \times 10^{-3}$ | 0.1 | 0.01 | <span style="color: green;">✓</span> | 358,400 | 252.6 | 307.8 | 29.9 |
| `f263520d` | $1 \times 10^{-4}$ | 0.1 | 0.01 | <span style="color: green;">✓</span> | 512,000 | 244.0 | 321.2 | 77.1 |
| `7049df9e` | $3 \times 10^{-4}$ | 0.1 | 0.0 | <span style="color: green;">✓</span> | 256,000 | 202.7 | 304.4 | 89.6 |
| `fe42cbac` | $1 \times 10^{-3}$ | 0.1 | 0.0 | <span style="color: red;">✗</span> | — | <span style="color: red;">174.2</span> | 300.5 | <span style="color: red;">121.4</span> |
| `9771fd6a` | $1 \times 10^{-3}$ | 0.2 | 0.0 | <span style="color: red;">✗</span> | — | <span style="color: red;">164.8</span> | 296.6 | <span style="color: red;">133.7</span> |
| `19d04c7a` | $1 \times 10^{-4}$ | 0.1 | 0.0 | <span style="color: red;">✗</span> | — | <span style="color: red;">154.9</span> | 291.9 | <span style="color: red;">112.7</span> |

*Table 1: All 12 sweep configs, sorted by the best checkpoint's mean return over the 100-episode evaluation. A config is solved when this mean is at least 200. The 100-ep max is the best single-episode return over the same evaluation; even the three failures reach a max near 290–300, close to the solved configs, which is what makes them volatile rather than incapable. The first-threshold step is the first mid-training (20-episode) probe step to exceed 200, a noisy time-to-threshold diagnostic rather than the solve criterion. Green marks the best config and the fastest to threshold; red marks the three failures.*

### 7.2. The Entropy Bonus Is the Dominant Reliability Driver

The cleanest split in the data is along the entropy coefficient. All six configs with $c_2 = 0.01$ solve the task; only three of the six with $c_2 = 0.0$ do, and every one of the three failures has $c_2 = 0.0$. The bonus also compresses variance: the worst standard deviation among the $c_2 = 0.01$ configs is 77.1, whereas all three failures (all $c_2 = 0.0$) exceed 110.

The mechanism is visible in the entropy curves ([Section 8.2](#82-entropy-schedule)). With $c_2 = 0.01$, the bonus pushes the learnable $\log\sigma$ upward, so policy entropy is held high or even rises over training (e.g. `6e1b8a66` ends at 5.06 against a 2.84 start). The Gaussian stays wide, the agent keeps exploring, and it avoids the premature collapse to a narrow, locally optimal descent profile that strands the $c_2 = 0.0$ failures.

This refines the design-note expectation in [Section 5.1](#51-no-entropy-bonus-by-default). The learnable $\log\sigma$ alone is *sufficient* in the best case, since the single highest-scoring config, `cbeefb0b` (280.2), uses $c_2 = 0.0$. It is not *robust*: at the same coefficient, three of the remaining five configs fail outright and a fourth (`7049df9e`) only scrapes across at 202.7, leaving just one other clean solve (`56f68680`). A small entropy bonus is the cheaper path to a config that solves regardless of the other two knobs. The pragmatic recommendation is therefore the opposite of the implemented default: prefer $c_2 = 0.01$ for the search, and drop to $c_2 = 0.0$ only when squeezing out the last few points of return on an already-stable config.

### 7.3. Learning Rate: Speed Against Stability

Learning rate sets the trade-off between how fast a config reaches a solving policy and how likely it is to get there at all. The quickest to threshold is `6e1b8a66` at $\text{lr} = 10^{-3}$, whose mid-training probe first clears 200 at just 102,400 steps, half the figure for the best-scoring config (`cbeefb0b`, 204,800) and a fifth of the slowest $\text{lr} = 10^{-4}$ configs (512,000–563,200 steps). However, $\text{lr} = 10^{-3}$ is also the most fragile setting: two of its four configs fail (`fe42cbac`, `9771fd6a`), both with $c_2 = 0.0$. The high learning rate that accelerates a well-regularised run also amplifies the policy oscillations that sink an under-regularised one.

The middle rate $3 \times 10^{-4}$ is the safest all-rounder: all four of its configs solve, and it produces the best score overall. The lowest rate $10^{-4}$ also solves whenever it is paired with the entropy bonus, but pays for stability with the slowest convergence in the grid. The practical reading is that $3 \times 10^{-4}$ is the right default, $10^{-3}$ is worth the risk only with entropy regularisation in place, and $10^{-4}$ is a conservative fallback that wastes budget.

### 7.4. Clip Epsilon: A Secondary Effect

Clip range is the weakest of the three swept factors. The tighter $\varepsilon = 0.1$ solves four of six configs and the looser $\varepsilon = 0.2$ solves five of six, with the lone $\varepsilon = 0.2$ failure (`9771fd6a`) attributable to its $\text{lr} = 10^{-3}$, $c_2 = 0.0$ pairing rather than to the clip itself. The best config uses $\varepsilon = 0.2$, but matched pairs do not show a consistent advantage in either direction once learning rate and entropy are controlled for. Clip range is best left at the $0.2$ default and not spent on sweep budget until the other two factors are bracketed.

---

## 8. Training Curves and Diagnostics

Each config's matplotlib output is a $2 \times 3$ panel grid: episode returns, episode lengths, and clip fraction on the top row; policy loss, value loss, and policy entropy on the bottom row. The two diagnostics that most cleanly separate success from failure are the clip fraction and the entropy schedule.

### 8.1. Clip Fraction as a Health Signal

The clip fraction, i.e. the proportion of mini-batch samples whose importance ratio is clamped by $\varepsilon$, behaves very differently in stable and unstable runs. Solved configs show a moderate early clip fraction (roughly 0.05–0.18 averaged over the first tenth of updates), a brief transient peak as the policy first moves (up to $\approx 0.6$), and then a decay to a stable low value (0.03–0.22 for eight of the nine, with the high-learning-rate solve `c509e60a` the outlier at 0.33). The two high-learning-rate failures instead let the clip fraction stay elevated: `fe42cbac` averages 0.316 early, peaks at 0.922, and is still at 0.750 at the end of training, meaning three of every four samples are pinned to the clip boundary on the final updates, and `9771fd6a` ends near 0.47. The policy is thrashing against the trust region rather than settling inside it.

The curriculum offers a heuristic that a clip fraction spiking above 0.4 early in training indicates the learning rate is too high. The sweep refines this in two ways. First, the early value alone is not decisive: `fe42cbac` averages 0.316 early yet fails, while its matched twin `c509e60a` averages an almost identical 0.314 and solves, the difference being the entropy bonus rather than anything visible in the early clip fraction. The more reliable reading is the *sustained* clip fraction late in training, but it flags only the overshoot failure mode: the two high-learning-rate failures stay above 0.45, far above the solved band, whereas the low-learning-rate failure `19d04c7a` ends at just 0.23, indistinguishable from a healthy run. `19d04c7a` fails for the opposite reason, i.e. too-slow learning compounded by an early entropy collapse, so its small updates never push many samples past the clip boundary. Second, the highest learning rate ($10^{-3}$) does produce the highest early clip fractions in the grid (roughly 0.18–0.32), consistent with the heuristic's direction, but whether that early pressure resolves or compounds is decided by the entropy bonus, not the learning rate alone.

Plotting all twelve per-update clip-fraction traces on one axis is unreadable, so [Figure 1](#fig-clip-bar) summarises each run by its sustained clip fraction instead. The two high-learning-rate failures clear the $\approx 0.45$ flag, while `19d04c7a` sits down among the solved configs, the split just described.

<figure id="fig-clip-bar" style="text-align: center;">
  <img src="/assets/images/lunarlander_clip_fraction_sustained_bar.png" alt="Sustained clip fraction by config." style="width: 90%;">
  <figcaption>Figure 1: Sustained clip fraction (final logged value) for each config, coloured by outcome and sorted. The two high-learning-rate failures (<code>fe42cbac</code>, <code>9771fd6a</code>) sit above the ~0.45 overshoot flag, whereas the low-learning-rate failure <code>19d04c7a</code> (0.23) sits among the solved configs. This is the same point as the raw curves but legible: a high sustained clip fraction flags the overshoot failure mode, not slow-learning failures.</figcaption>
</figure>

### 8.2. Entropy Schedule

Policy entropy starts at 2.84 for every config (the entropy of the freshly initialised Gaussian). What happens next depends on whether the entropy coefficient puts a floor under it, and the contrast between $c_2 = 0.0$ and $c_2 = 0.01$ is the clearest curve-level explanation of the reliability gap in [Section 7.2](#72-the-entropy-bonus-is-the-dominant-reliability-driver).

At $c_2 = 0.0$, the entropy schedule is governed by the policy gradient alone, and it can go two ways. In a clean run the policy sharpens around a good landing trajectory and entropy decays smoothly to a low value (`cbeefb0b` settles near 0.64), which produces the sharpest mean policy and the best score in the sweep. In a failing run the policy instead overcommits early and entropy collapses (`9771fd6a` dives to 0.45), leaving a narrow policy that performs poorly. Because $\log\sigma$ is a learnable parameter driven by the surrogate-loss gradient and not only by the entropy term, the gradient then reinflates $\sigma$ once the collapsed mean underperforms and better-than-expected actions land off that mean (the gradient of the Gaussian log-probability with respect to $\log\sigma$ is proportional to $(a - \mu)^2/\sigma^2 - 1$, which is positive when actions fall more than one standard deviation from the mean). `9771fd6a` accordingly climbs back to 1.44, but this rebound is an oscillating, unconverged policy rather than recovered exploration. The salient point is that at $c_2 = 0.0$ nothing prevents the early collapse, so whether a config lands in the clean-decay regime or the collapse-and-thrash regime is left to chance.

At $c_2 = 0.01$, the bonus adds a constant upward push on $\log\sigma$, holding entropy high throughout training (`b97050c1` ends at 3.07, `c509e60a` at 4.41, `6e1b8a66` at 5.06). This is precisely the advantage: the floor removes the premature-collapse failure mode by construction, so the policy keeps exploring until its value estimates are reliable, and all six $c_2 = 0.01$ configs solve. The wider policy is almost free at evaluation, because the deterministic eval scores the mean action ([Section 3.4](#34-deterministic-evaluation)) and ignores $\sigma$, so a high training-time variance barely affects the reported return. The one thing the bonus gives up is the very top of the range, since the single sharpest policy comes from a clean $c_2 = 0.0$ decay, which is why $c_2 = 0.01$ is the robust default rather than the score-maximising one.

This entropy effect leaves a second fingerprint on the clip fraction. Because the bonus keeps the policy from collapsing and lurching between narrow distributions, it also relieves pressure on the surrogate clip, so the importance ratio is clamped less often. [Figure 2](#fig-clip-pairs) makes this visible across the three matched failure-success pairs, each fixing the learning rate and clip range and varying only $c_2$. Read pairwise, the $c_2 = 0.01$ twin sustains a lower clip fraction than its $c_2 = 0.0$ partner in every pair, and the gap is widest at the high learning rate ($10^{-3}$), where overshoot is the dominant risk. The plot must be read pairwise rather than as absolute bands: the solved `c509e60a` sits above the failed `19d04c7a`, so a low clip fraction does not by itself imply success, consistent with [Section 8.1](#81-clip-fraction-as-a-health-signal).

<figure id="fig-clip-pairs" style="text-align: center;">
  <img src="/assets/images/lunarlander_train_clip_fraction_tb.png" alt="Clip fraction for three matched failure-success pairs." style="width: 100%;">
  <figcaption class="arithmatex">Figure 2: TensorBoard <code>train/clip_fraction</code> (smoothing 0.95) for the three matched failure-success pairs, each fixing learning rate and clip range and varying only \(c_2\): <code>9771fd6a</code>/<code>6e1b8a66</code> (lr \(10^{-3}\), \(\varepsilon\) 0.2), <code>fe42cbac</code>/<code>c509e60a</code> (lr \(10^{-3}\), \(\varepsilon\) 0.1), and <code>19d04c7a</code>/<code>f263520d</code> (lr \(10^{-4}\), \(\varepsilon\) 0.1). Within each pair the \(c_2\) = 0.01 twin sustains a lower clip fraction, most clearly for the high-learning-rate pair. The bands track learning rate, not outcome, so the figure must be read pairwise.</figcaption>
</figure>

### 8.3. A Clean Solve and a Representative Failure

[Figure 3](#fig-best) shows the training curves for the best config, `cbeefb0b`. Returns rise steadily and the clip fraction and entropy both relax to stable values, the signature of a run that has settled inside its trust region.

<figure id="fig-best" style="text-align: center;">
  <img src="/assets/images/lunarlander_ppo_training_cbeefb0b.png" alt="Training curves for config cbeefb0b." style="width: 100%;">
  <figcaption class="arithmatex">Figure 3: Training curves for the best config <code>cbeefb0b</code> (lr \(= 3 \times 10^{-4}\), \(\varepsilon\) = 0.2, \(c_2\) = 0.0). Top row: episode returns, episode lengths, clip fraction. Bottom row: policy loss, value loss, policy entropy.</figcaption>
</figure>

[Figure 4](#fig-fail) shows a representative failure, `9771fd6a`. All three failures are the $c_2 = 0.0$ members of matched pairs whose $c_2 = 0.01$ twins solved (`9771fd6a`/`6e1b8a66`, `fe42cbac`/`c509e60a`, `19d04c7a`/`f263520d`, each pair sharing its learning rate and clip range), which is the controlled evidence behind [Section 7.2](#72-the-entropy-bonus-is-the-dominant-reliability-driver). Strikingly, `9771fd6a`'s twin `6e1b8a66` is the fastest solve in the sweep, so a single entropy bonus flips this pair from the most unstable failure to the quickest success. `9771fd6a` is the most representative of the three: it carries the highest 100-episode standard deviation in the entire sweep (133.7), its clip fraction stays elevated (ending near 0.47) rather than relaxing to the solved band, and its entropy follows the early-collapse-then-rebound trajectory dissected in [Section 8.2](#82-entropy-schedule), diving to 0.45 before climbing back to 1.44.

<figure id="fig-fail" style="text-align: center;">
  <img src="/assets/images/lunarlander_ppo_training_9771fd6a.png" alt="Training curves for config 9771fd6a." style="width: 100%;">
  <figcaption class="arithmatex">Figure 4: Training curves for a representative failure <code>9771fd6a</code> (lr \(= 10^{-3}\), \(\varepsilon\) = 0.2, \(c_2\) = 0.0), the most unstable run in the sweep (100-episode std 133.7); its matched \(c_2\) = 0.01 twin <code>6e1b8a66</code> is the fastest solve. The elevated clip fraction (ending near 0.47) and the entropy collapse-and-rebound (0.45 → 1.44) mark a policy that oscillates rather than converging.</figcaption>
</figure>

The TensorBoard aggregate `eval/mean_return` view ([Figure 5](#fig-tb-returns)) makes the entropy-bonus split legible at a glance: grouping all twelve runs separates the solved band from the three failures.

<figure id="fig-tb-returns" style="text-align: center;">
  <img src="/assets/images/lunarlander_eval_mean_return_tb.png" alt="TensorBoard eval mean return across all 12 configs." style="width: 100%;">
  <figcaption>Figure 5: TensorBoard <code>eval/mean_return</code> (mid-training 20-episode probes) across all 12 configs. The three failing runs separate as a low band that stays under the 200 mark.</figcaption>
</figure>

### 8.4. Late-Training Degradation and the Value of Best-Checkpoint Selection

PPO's mid-training evaluation curve on this task is markedly non-monotonic: a run frequently peaks partway through training and then drifts down. For `cbeefb0b`, the best checkpoint is taken at step 665,600 and scores 280.2 over the 100-episode evaluation, whereas the final mid-training probe at one million steps falls back to roughly 197. Several solved configs show the same pattern (`b97050c1`'s best checkpoint scores 278.3, but its final probe is near 146), and the failures swing even harder.

This validates the best-checkpoint mechanism described in [Section 3.3](#33-best-checkpoint-selection-with-mid-training-evaluation) as a measurable contributor, not mere hygiene. A last-iterate policy, i.e. one reported from whatever weights happen to exist at step one million, would forfeit on the order of 80–130 return on the stronger configs and would misrank the sweep. Restoring the best checkpoint recovers that margin and is what makes the [Table 1](#tab-sweep) ordering reflect each config's true capability rather than the noise of its final probe.

---

## 9. Empirical Summary

The sweep shows that PPO solves `LunarLanderContinuous-v3` comfortably, with nine of twelve configs clearing the 200-return bar on the 100-episode evaluation of their best checkpoint, but that *reliability* is governed primarily by the entropy bonus and secondarily by the learning rate. A small entropy coefficient ($c_2 = 0.01$) solved every config it touched and accounted for the difference between the solved and failed groups, by holding the policy's exploration up and preventing the premature distribution collapse that strands the un-regularised runs. Learning rate trades convergence speed against stability: $3 \times 10^{-4}$ is the safe default, $10^{-3}$ is the fastest but only safe with the entropy bonus, and $10^{-4}$ is a slow but dependable fallback. Clip range is a third-order effect and is best left at 0.2.

The most useful training-curve diagnostic is the *sustained* clip fraction, with one caveat. A value that stays above roughly 0.45 into late training reliably flags the overshoot failure mode, where too-high a learning rate leaves the policy thrashing against the trust region, and it does so more cleanly than the early-spike heuristic alone. The converse does not hold: a low sustained clip fraction does not guarantee success, since the low-learning-rate failure decays into the solved band yet still fails, for the unrelated reason of learning too slowly with no entropy floor to keep it exploring. Finally, because PPO's evaluation curve peaks mid-training and drifts, best-checkpoint selection is not optional polish but a mechanism that recovers 80–130 return on the stronger configs and is necessary for the sweep ranking to be meaningful.

---

## 10. What I Would Extend Next

The implementation is intentionally scoped to the single-environment, single-seed regime that the project's curriculum prescribes. Several extensions would be natural next steps and are worth flagging.

First, vectorised environments via `gym.vector.AsyncVectorEnv` would amortise the rollout collection cost across multiple parallel environment instances per worker, decoupling sample throughput from the single-environment step rate. This is the standard scaling lever for PPO on continuous control and would compress the wall-clock time of the sweep substantially.

Second, multi-seed runs per config would give variance estimates on the reported mean return. The current single-seed sweep cannot fully distinguish a genuinely-better config from a luckier seed, which is the main caveat on the factor analysis in [Section 7](#7-sweep-results). A small bump to three or five seeds per config, adding the existing `seed` field to the sweep grid, would close that gap and put error bars on the entropy-bonus effect.

Third, the parallel sweep harness is a fixed Cartesian product. Replacing it with a wandb-managed sweep (Bayesian or Hyperband) would let the search adapt to early signals from completed runs, which is a more sample-efficient way to explore high-dimensional hyperparameter spaces.

Fourth, a recurrent policy (gated recurrent unit (GRU) or long short-term memory (LSTM)) would generalise the implementation to partially observable environments. The rollout buffer would then need to carry the recurrent hidden state alongside each transition, and the mini-batch sampling would have to respect episode boundaries rather than shuffling across them.

Each of these is a controlled extension of the existing scaffolding. The current file structure (typed config, decoupled agent, factored rollout buffer, observability split along its own axis) was chosen with such extensions in mind.
