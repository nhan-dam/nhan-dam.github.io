# Deep Q-Network on Gymnasium CartPole-v1

> Created on: 9 May 2026
>
> Updated on: 14 May 2026

---

This note documents an implementation of the hands-on project described in [Section 1.4](phase01.md#project-req-dqn-cartpole): deep Q-network on `CartPole-v1` environment from [Gymnasium](https://gymnasium.farama.org/).

The full source code can be found on [GitHub](https://github.com/nhan-dam/rl-foundations/blob/main/src/dqn_cartpole.py).

---

## 1. Implementation and Design Choices

### 1.1. Core DQN Mechanisms

Vanilla Q-learning diverges when the Q-function is approximated by a neural network, because the training signal is both non-stationary (targets move as the network updates) and temporally correlated (consecutive transitions share state). The Deep Q-Network (DQN) algorithm (Mnih et al., 2015) stabilises training through two mechanisms.

**Experience replay.** Transitions $(s, a, r, s')$ are stored in a circular replay buffer of capacity 50,000 and sampled in random mini-batches during training. This breaks temporal correlation, producing approximately independent and identically distributed (i.i.d.) training samples, and allows each transition to contribute to multiple gradient updates, improving sample efficiency.

**Target network.** A separate copy of the Q-network, whose weights are frozen for $N$ steps, computes the Bellman targets $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$. Without this, the network chases a target that shifts every gradient step, a feedback loop that causes divergence. This implementation uses a hard update every 100 steps, copying the online network's weights wholesale. The alternative, a soft update $\theta_{\text{target}} \leftarrow \tau \theta_{\text{online}} + (1 - \tau)\theta_{\text{target}}$, smooths the target trajectory continuously but means the online network always has some influence over the target. The hard update provides a cleaner separation between prediction and target, and is sufficient for a low-dimensional task such as CartPole.

### 1.2. $\varepsilon$ Decay: Per-Step Rather Than Per-Episode

In tabular Q-learning, $\varepsilon$ is decayed per episode. This reflects the structure of discretised state spaces: each bin in the Q-table is an independent estimate that only becomes reliable after repeated visits to that bin specifically. An episode boundary is a natural checkpoint by which the agent has traversed enough of the discretised space to justify reducing exploration pressure.

DQN operates on the raw, continuous state representation and approximates Q-values with a neural network that generalises across nearby states via gradient descent. A single transition in one region of the state space propagates useful updates to the Q-values of neighbouring states, hence no state needs to be revisited repeatedly for its estimate to improve. The individual transition is therefore the natural unit at which to reduce $\varepsilon$, making per-step decay the appropriate choice for DQN.

### 1.3. Evaluation Policy: Retaining $\varepsilon = 0.05$

In tabular Q-learning, setting $\varepsilon = 0$ during evaluation is appropriate because each $Q(s, a)$ entry is updated independently with no generalisation across states. Estimation errors remain local and a fully greedy policy gives an unbiased readout of the learned values.

DQN's neural network generalises across states, which is precisely what makes it scalable but also introduces an overfitting risk: the network is exposed to some states far more than others during training, and its Q-value estimates for frequently visited states can become confident but brittle. Mnih et al. (2015) retain $\varepsilon = 0.05$ during evaluation to guard against this, producing a more conservative and honest estimate of the policy's generalisation ability. This implementation follows the same practice.

---

## 2. Experimental Setup

The sweep evaluates a DQN agent on `CartPole-v1` across a $5 \times 3 \times 2 = 30$ config grid: five learning rates ($10^{-4}$, $5 \times 10^{-4}$, $10^{-3}$, $3 \times 10^{-3}$, $5 \times 10^{-3}$), three batch sizes (32, 64, 128), and two $\varepsilon$ decay strategies (linear and exponential). The solve criterion is mean return > 475 evaluated over 100 episodes within 50,000 environment steps, i.e. the agent must balance the pole near-perfectly before it has seen 50,000 transitions.

All other hyperparameters are held fixed: discount factor $\gamma = 0.99$, replay buffer capacity 50,000, minimum buffer fill 1,000 steps before training begins, target network hard-update every 100 steps, hidden dimension 128, $\varepsilon$ decay completing at 10,000 steps, and 1,000 training episodes per config. The final reported mean return is computed over 1,000 evaluation episodes using the best checkpoint from mid-training probes.

---

## 3. Sweep Results

### 3.1. Overall Outcome

There are 14 of 30 configs that satisfy the solve criterion. The 16 failures are not total failures; most eventually achieve high mean return but take longer than 50,000 steps to do so. The core question is therefore what drives early-phase learning speed and stability.

### 3.2. The Dominant Factor: Learning Rate

Learning rate is the single most influential hyperparameter. Every config with $\text{lr} = 10^{-4}$ fails the 50,000-step criterion, and most take 100,000–225,000 steps to first cross the threshold, around two to four times the budget. By contrast, configs with $\text{lr} \geq 5 \times 10^{-4}$ overwhelmingly pass, with the sole clear exception being `690440aa` ($\text{lr} = 5 \times 10^{-3}$, batch 32, exponential), which never achieves reliable performance at all.

The effective learning rate range for this task and architecture is approximately $[5 \times 10^{-4},\, 5 \times 10^{-3}]$: fast enough to update Q-values meaningfully from each mini-batch but not so large that gradient steps destabilise the function approximator against the non-stationary temporal-difference (TD) targets.

### 3.3. Batch Size: Secondary and Inconclusive

Batch size has no monotonic relationship with performance. All three sizes (32, 64, 128) appear in the top-performing tier. Batch 64 is slightly over-represented among the perfect-score configs (mean = 500, std = 0), but there is no clean mechanistic explanation for this: both gradient variance and sample diversity from the replay draw improve with larger batch, so neither factor predicts a middle-batch advantage. The most likely explanation is sampling noise from a single-seed sweep. The practical conclusion is that batch size does not drive success or failure here; the strongest predictor of failure is always learning rate.

### 3.4. Decay Strategy: Minor Advantage for Exponential at Higher LRs

Across matched (LR, batch) pairs, exponential decay tends to reach `solved_step` slightly earlier than linear. The most likely reason is that exponential decay is front-loaded: the agent's exploration rate falls faster in the early steps, accelerating the transition to exploitation when the Q-network starts producing useful estimates. Linear decay is uniform, keeping $\varepsilon$ higher for longer. The difference rarely changes whether a config passes the criterion, but it noticeably affects *when* it first passes.

### 3.5. Best and Worst Configurations

The table below summarises the six perfect-score configs and the single clear failure.

| Config | LR | Batch | Decay | Solved step | Mean | Std |
|---|---|---|---|---|---|---|
| `66e09612` | $5 \times 10^{-4}$ | 64 | exp | 50,000 | <span style="color: green;">500.0</span> | <span style="color: green;">0.0</span> |
| `ab7edcd5` | $3 \times 10^{-3}$ | 32 | exp | 25,000 | <span style="color: green;">500.0</span> | <span style="color: green;">0.0</span> |
| `9aae1900` | $5 \times 10^{-4}$ | 128 | linear | 50,000 | <span style="color: green;">500.0</span> | <span style="color: green;">0.0</span> |
| `d7733cbc` | $10^{-3}$ | 64 | exp | 25,000 | <span style="color: green;">500.0</span> | <span style="color: green;">0.0</span> |
| `9aaebea6` | $3 \times 10^{-3}$ | 64 | linear | 25,000 | <span style="color: green;">500.0</span> | <span style="color: green;">0.0</span> |
| `88694cf2` | $5 \times 10^{-3}$ | 32 | linear | 50,000 | <span style="color: green;">500.0</span> | <span style="color: green;">0.0</span> |
| `690440aa` | $5 \times 10^{-3}$ | 32 | **exp** | N/A | <span style="color: red;">437.1</span> | <span style="color: red;">160.1</span> |

The juxtaposition of `88694cf2` and `690440aa` is instructive: the same learning rate and batch size, with only the decay strategy differing. Linear decay succeeds; exponential fails. At $\text{lr} = 5 \times 10^{-3}$, the gradient updates are already large. Exponential decay removes exploration more aggressively in the early steps, committing the agent to a noisy, high-LR policy before the Q-network has stabilised. The result is self-reinforcing instability.

### 3.6. Transferable Lessons

- **Start with $\text{lr} \in [5 \times 10^{-4}, 2 \times 10^{-3}]$ for MLP-based DQN.** Both extremes of the swept range fail or are fragile. The failure mode of too-low LR is slow convergence; the failure mode of too-high LR is catastrophic forgetting and Q-value divergence.
- **$\varepsilon$ decay strategy matters more at extreme learning rates.** In the moderate-LR regime ($10^{-3}$ to $3 \times 10^{-3}$), both strategies succeed. Exponential is a safer default when LR is moderate because it accelerates early exploitation. At high LR, prefer linear decay to avoid overcommitting before the Q-network is reliable.
- **Batch size is a secondary concern at this scale.** Do not spend sweep budget on fine-grained batch search until learning rate is bracketed.
- **Std of zero is a meaningful signal.** Configs achieving mean = 500, std = 0 over 1,000 evaluation episodes have converged to a deterministic near-optimal policy; high std (> 50) signals persistent instability even if the mean looks adequate.

---

## 4. Training Curves

Training plots show three panels: a 50-episode rolling mean of episode return, the same for episode length (which is identical to return in CartPole since reward is +1 per step), and a 2,000-step rolling mean of the TD loss.

### 4.1. Success: Config `66e09612` (lr $= 5 \times 10^{-4}$, batch 64, exponential)

[Figure 1](#fig-66e09612) shows the training curves for this config, the cleanest successful run in the sweep.

<figure id="fig-66e09612" style="text-align: center;">
  <img src="/assets/images/cartpole_dqn_training_66e09612.png" alt="Training curves for config 66e09612." style="width: 90%;">
  <figcaption>Figure 1: Training curves for config <code>66e09612</code> (lr = 5 × 10⁻⁴, batch 64, exponential decay). Episode returns (left), episode lengths (centre), and TD loss (right).</figcaption>
</figure>

**Episodes 1–100 (buffer fill).** Return sits near the random baseline ($\approx 20–30$ steps). The replay buffer requires 1,000 steps before training begins; given that early episodes are short, roughly 25–50 episodes pass in pure data collection. TD loss is near zero because no gradient updates have occurred.

**Episodes 100–300 (rapid learning).** Return climbs steeply from $\approx 50$ to $\approx 350$ steps. The Q-network begins to correctly rank actions, distinguishing 'push toward the pole lean' from 'push away'. Simultaneously, $\varepsilon$ is decaying and the buffer accumulates diverse transitions across a range of pole angles. TD loss rises to $\approx 10–20$, reflecting growing Q-value magnitudes rather than deteriorating fit.

**Episodes 300–900 (plateau with bounded variance).** Return oscillates between $\approx 350$ and $\approx 500$ around a stable high mean. TD loss continues rising to $\approx 30–40$ as the discounted returns the network must represent grow, but the rate of increase is gradual and the loss does not diverge. The training procedure saves the best checkpoint, so the terminal drop at episodes $\approx 900–950$ (after the best point has passed) does not affect the evaluated policy.

Most other successful configs follow the same three phases, differing only in the speed of the rise: configs with $\text{lr} \in \{10^{-3},\, 3 \times 10^{-3}\}$ reach the plateau roughly 100 episodes earlier. Configs with $\text{lr} = 10^{-4}$ exhibit a slow ramp, taking 300–400 episodes to arrive where successful configs arrive by episode 200, because the per-update gradient step is too small to overcome noise in the single-sample Bellman targets quickly enough.

### 4.2. Failure: Config `690440aa` (lr $= 5 \times 10^{-3}$, batch 32, exponential)

[Figure 2](#fig-690440aa) shows the training curves for this config, the only one that never achieves reliable performance.

<figure id="fig-690440aa" style="text-align: center;">
  <img src="/assets/images/cartpole_dqn_training_690440aa.png" alt="Training curves for config 690440aa." style="width: 90%;">
  <figcaption>Figure 2: Training curves for config <code>690440aa</code> (lr = 5 × 10⁻³, batch 32, exponential decay). Episode returns (left), episode lengths (centre), and TD loss (right).</figcaption>
</figure>

**Episodes 1–100 (fast initial rise).** Return climbs to $\approx 200$ steps faster than most configs, as the high learning rate moves Q-values substantially per update and exponential decay removes exploration quickly.

**Episodes 100 onward (repeated collapse).** In contrast to successful configs, where TD loss rises gradually and return stabilises, here TD loss diverges to $> 400$ and return cycles repeatedly between $\approx 300–350$ and near-random levels. The underlying mechanism is compounding Q-value overestimation. High LR causes the online network to develop inflated Q-values through large, noisy gradient updates: each mini-batch update overshoots the true Bellman target, and with a constantly shifting data distribution the corrections never converge. When the target network syncs, it copies these already-inflated values and holds them fixed as regression targets for the next 100 steps; the online network is then trained to match inflated targets, pushing its Q-values higher still, ready to seed the next sync. The early exhaustion of exploration from exponential decay exacerbates this, as the near-deterministic trajectories that result reduce buffer diversity and deprive the network of the varied Bellman targets that would otherwise dampen overestimation. The contrast with `88694cf2` (same LR, same batch, linear decay; [Figure 3](#fig-88694cf2)) confirms the diagnosis: linear decay keeps $\varepsilon$ higher for longer, breaking the feedback loop between premature exploitation and diverging Q-values.

<figure id="fig-88694cf2" style="text-align: center;">
  <img src="/assets/images/cartpole_dqn_training_88694cf2.png" alt="Training curves for config 88694cf2." style="width: 90%;">
  <figcaption>Figure 3: Training curves for config <code>88694cf2</code> (lr = 5 × 10⁻³, batch 32, linear decay). Episode returns (left), episode lengths (centre), and TD loss (right). Identical to <code>690440aa</code> in every hyperparameter except decay strategy; the stable plateau and bounded TD loss confirm that keeping exploration higher for longer prevents the overestimation cycle.</figcaption>
</figure>

The key distinction between this failure mode and the bounded oscillation in successful runs is therefore not the presence of performance dips, since all DQN runs exhibit some variance, but whether the TD loss diverges. A TD loss that plateaus signals healthy learning, whereas one that compounds without bound signals overestimation taking hold.

---

## 5. Summary

The sweep demonstrates that DQN on CartPole is more sensitive to learning rate than to any other hyperparameter in the swept grid. The sweet spot is $5 \times 10^{-4}$ to $3 \times 10^{-3}$. Below this range the algorithm is too slow to meet a 50,000-step criterion. Above it, specifically $5 \times 10^{-3}$ with exponential decay, the high learning rate drives Q-value overestimation into a divergent cycle. Batch size and decay strategy are secondary, since their effects are most visible at the edges of the learning rate range, where small changes in gradient noise or exploration schedule tip the balance between stability and divergence.

The most important training-curve diagnostic is the TD loss scale. Values in the range 10–80 indicate healthy learning, whilst values above 100 and still rising at training end are a reliable predictor of the overestimation cycle and should prompt reducing the learning rate or switching to Double DQN.

---

## Appendix A. Double DQN

### A.1. Motivation

Standard DQN computes the Bellman target as $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$. The $\max$ operator performs two operations with the same network: it selects the highest-valued next action and evaluates it. Because the same weights drive both steps, any action whose Q-value is overestimated is both preferentially selected and assigned that inflated value as the regression target. This feeds back into subsequent target computations, causing overestimation to compound across training. The effect is mild on short, simple tasks but becomes a meaningful source of policy degradation in environments with large action spaces, long horizons, or sparse rewards.

### A.2. Key Implementation Difference

Double DQN (van Hasselt et al., 2016) decouples the two operations across the two networks that already exist in standard DQN. The online network selects the greedy next action, then the target network evaluates it:

$$
y = r + \gamma \, Q_{\text{target}}\!\left(s',\, \arg\max_{a'} Q_{\text{online}}(s', a')\right)
$$

Because the two networks carry different weights, an action that the online network overestimates tends to receive a more conservative value from the target network, breaking the self-reinforcing bias. The change touches only the TD target computation. Everything else, including environment-interaction action selection via the online network, the replay buffer, and the target network sync schedule, is identical to standard DQN. No additional parameters or computational cost are introduced.

---

## Appendix B. Parallelising the Hyperparameter Sweep

### B.1. Motivation

A grid search over learning rate, batch size, and $\varepsilon$ decay strategy produces 30 configurations. Running these sequentially is wasteful: each configuration is independent of the others, and the M4 Pro workstation provides 12 central processing unit (CPU) cores that would otherwise sit idle. Distributing the sweep across worker processes allows multiple configurations to train simultaneously, reducing wall-clock time by roughly an order of magnitude.

### B.2. Why Processes, Not Threads

Python's global interpreter lock (GIL) prevents multiple threads from executing Python bytecode in parallel. For CPU-bound workloads such as neural network training on small models, multithreading provides no throughput gain. The correct primitive is `multiprocessing`, which spawns independent OS-level processes, each with its own Python interpreter and memory space, and therefore unaffected by the GIL.

### B.3. Process Start Method: `spawn`

Python's `multiprocessing` module supports two process start methods: `fork` and `spawn`.

`fork` creates a child process by duplicating the parent's entire memory space using the UNIX `fork()` system call. The child inherits all open file descriptors, locks, and library state. Because the copy is performed at the OS level via copy-on-write, `fork` is fast. However, it is inherently unsafe when the parent holds internal state that is not designed to be duplicated, such as OS-level locks, GPU contexts, or Objective-C runtime objects; duplicating such state produces two processes that each believe they hold exclusive ownership, leading to undefined behaviour.

`spawn` creates a child process by launching a fresh Python interpreter from scratch. The child starts with a clean slate, re-imports only the modules it needs, and receives its task via inter-process communication (IPC). This is slower to start than `fork` but safe in all contexts.

macOS prohibits the `fork` start method when PyTorch is in use: the Objective-C runtime and Metal Performance Shaders (MPS) contexts are not fork-safe, and forking a process that has already initialised these subsystems produces undefined behaviour. The `spawn` start method must therefore be set explicitly before any pool is created:

```python
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

Under `spawn`, each subprocess is a fresh Python interpreter that re-imports all modules from scratch. This has two consequences. First, the worker function must be defined at module level rather than as a local function or lambda, because `spawn` pickles the function by name and unpickling requires the function to be importable. Second, all code in `__main__` must be guarded by `if __name__ == "__main__":` to prevent recursive subprocess spawning.

### B.4. Device Assignment: CPU per Worker

A natural question is whether worker processes can exploit the M4 Pro's GPU via MPS to accelerate training. They cannot, for two reasons.

First, GPU shader cores are single instruction, multiple data (SIMD) lanes that execute the same instruction across many data elements in lockstep. They have no independent program counters and no OS-level process isolation. Multiple processes submitting MPS commands therefore serialise on the hardware, providing no parallelism over sequential execution.

Second, even in a sequential setting, MPS is slower than CPU for this workload. The Q-network is a two-hidden-layer multilayer perceptron (MLP) of width 128 with approximately 17,000 parameters. At this scale, Metal shader compilation and dispatch overhead exceeds the compute time saved by the GPU. The crossover point at which MPS outperforms CPU lies in the millions-of-parameters regime for the batch sizes used here.

Each worker process is therefore pinned explicitly to CPU:

```python
worker_device = torch.device("cpu")
```

This ensures that all 12 CPU cores are utilised for compute, and the GPU is left idle rather than becoming a bottleneck.

### B.5. Suppressing Per-Worker Output

When multiple worker processes write to `stdout` simultaneously, `tqdm` progress bars and console output from different workers interleave and become unreadable. The `train` function accepts a `silent` flag that suppresses its inner `tqdm` bar and mid-training evaluation prints when called from a worker process. Each worker instead emits a single one-line summary upon completion, which is short enough to remain legible even if lines from different workers interleave.

### B.6. Worker Count

With 12 CPU cores and 30 configurations, the number of workers is set to:

```python
n_workers = min(len(configs), max(1, (os.cpu_count() or 4) - 2))
```

Subtracting 2 from the core count reserves headroom for the OS scheduler and the main process, which handles progress tracking and result collection. With 12 cores this gives 10 workers, completing the 30-configuration sweep in three rounds.

### B.7. Result Collection

The worker function `run_one_config` is mapped over the configuration list using `pool.imap_unordered`, which yields results as each worker finishes rather than waiting for the entire pool. A `tqdm` bar wrapping the iterator tracks how many of the 30 configurations have completed. Results are collected into a list and written to a CSV file for inspection after the sweep finishes.

```python
with mp.Pool(processes=n_workers) as pool:
    results = list(
        tqdm(
            pool.imap_unordered(run_one_config, configs),
            total=len(configs),
            desc="Sweep",
        )
    )
```

### B.8. Memory Estimate

#### B.8.1. Model Weights

The Q-network is a three-layer MLP with input dimension 4 (the CartPole observation), two hidden layers of width 128, and output dimension 2 (one Q-value per action). The parameter count per layer is the weight matrix plus the bias vector.

- Layer 1 (input → hidden): $(4 \times 128) + 128 = 640$ parameters.
- Layer 2 (hidden → hidden): $(128 \times 128) + 128 = 16{,}512$ parameters.
- Layer 3 (hidden → output): $(128 \times 2) + 2 = 258$ parameters.
- Total per network: $640 + 16{,}512 + 258 = 17{,}410$ parameters.

Each parameter is stored as a 32-bit float (float32), occupying 4 bytes. One network therefore requires $17{,}410 \times 4 = 69{,}640$ bytes, approximately 68 KB. Each worker holds two networks (online and target), giving $2 \times 68 \approx 136$ KB for weights.

The Adam optimiser maintains two additional float32 tensors per parameter (the first moment estimate $m$ and the second moment estimate $v$), applied only to the online network. This adds a further $2 \times 17{,}410 \times 4 = 139{,}280$ bytes, approximately 136 KB.

Total for weights and optimiser state: $136 + 136 \approx 272$ KB, i.e. well under 1 MB.

#### B.8.2. Replay Buffer

Each transition is a 5-tuple `(obs, action, reward, next_obs, done)` stored as Python objects in a `collections.deque`. The two observation arrays each have shape `(4,)` and dtype float32. A NumPy array of this size carries a fixed object header of approximately 96 bytes plus $4 \times 4 = 16$ bytes of data, totalling 112 bytes per array.

The remaining three fields are CPython heap-allocated objects. In CPython, scalar types store their value inline within the object structure. The total object size (header plus inline value) for each type is:

- `action` as `int`: 16-byte PyObject header + 4-byte digit field + 8-byte alignment padding = **28 bytes**.
- `reward` as `float`: 16-byte PyObject header + 8-byte IEEE 754 `double` = **24 bytes**.
- `done` as `bool`: `bool` is a subclass of `int` in CPython and shares its layout = **28 bytes**.

The enclosing 5-tuple container holds five 8-byte pointers to the above objects, plus a 40-byte base tuple header, totalling $40 + 5 \times 8 = 80$ bytes.

Per-transition cost:

$$\underbrace{2 \times 112}_{\text{obs, next_obs}} + \underbrace{28}_{\text{action}} + \underbrace{24}_{\text{reward}} + \underbrace{28}_{\text{done}} + \underbrace{80}_{\text{tuple}} = 384 \text{ bytes.}$$

At capacity, the buffer holds 50,000 transitions:

$$50{,}000 \times 384 = 19{,}200{,}000 \text{ bytes} \approx 19 \text{ MB.}$$

#### B.8.3. Python Heap and Imported Libraries

Under `spawn`, each subprocess re-imports PyTorch, NumPy, Gymnasium, and the other dependencies into its own heap. The private, writable portion of this (interpreter state, module objects, Python-level caches) amounts to approximately 60 MB per process. This figure excludes the read-only code pages of shared libraries, which are accounted for in [Section B.8.5](#b85-shared-library-pages).

#### B.8.4. Per-Worker Total (Private Memory)

Summing the components above:

| Component | Size |
|---|---|
| Model weights (online + target) | 136 KB |
| Adam optimiser state | 136 KB |
| Replay buffer (50k transitions) | ~19 MB |
| Python heap and per-process state | ~60 MB |
| Environment wrappers, loss history, misc | ~5 MB |
| **Total private per worker** | **~84 MB** |

#### B.8.5. Shared Library Pages

The PyTorch and NumPy dynamic libraries are memory-mapped files. The macOS kernel loads their read-only code pages once into physical memory and maps them into each process's virtual address space via copy-on-write. The physical cost of these pages is therefore paid once regardless of worker count, at approximately 350 MB.

#### B.8.6. Total for 10 Workers

$$350 \text{ MB (shared)} + 10 \times 84 \text{ MB (private)} = 1{,}190 \text{ MB} \approx 1.2 \text{ GB.}$$

Adding a 20% headroom for allocation peaks during mini-batch sampling and Matplotlib plot generation gives a comfortable ceiling of approximately 1.5 GB.
