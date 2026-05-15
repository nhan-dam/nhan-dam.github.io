# Tabular Q-Learning on Gymnasium CartPole-v1

> Created on: 30 April 2026
>
> Updated on: 7 May 2026

## 1. Overview

This note documents an implementation of tabular Q-learning on `CartPole-v1` environment from [Gymnasium](https://gymnasium.farama.org/). The primary challenge relative to a discrete environment such as Blackjack is that CartPole exposes a **continuous, four-dimensional observation space**, which must be discretised before a Q-table can be indexed. The experiment includes a hyperparameter sweep over learning rate and $\varepsilon$-decay strategy, and analyses the learned policy via 2-D slices of the Q-table.

The full source code can be found on [GitHub](https://github.com/nhan-dam/rl-foundations/blob/main/src/qlearning_cartpole.py).

---

## 2. Implementation

### 2.1. The Discretisation Challenge

CartPole observations are continuous four-vectors:

$$\mathbf{s} = [\underbrace{x}_{\text{cart pos}},\ \underbrace{\dot{x}}_{\text{cart vel}},\ \underbrace{\theta}_{\text{pole angle}},\ \underbrace{\dot{\theta}}_{\text{pole ang. vel}}] \in \mathbb{R}^4.$$

A tabular Q-table requires a finite, discrete key for each state. The solution is to partition each dimension into $B$ equal-width bins and map each raw observation to a bin-index tuple via `np.digitize`. With $B = 8$ bins per dimension this yields $8^4 = 4{,}096$ discrete states, a manageable table size.

Bin ranges are set to cover the region the agent actually visits rather than the environment's theoretical extremes, and values outside the range are clipped to the nearest bin:

| Dimension | Range | Bins |
|---|---|---|
| Cart position | $[-2.4,\ 2.4]$ m | 8 |
| Cart velocity | $[-3.0,\ 3.0]$ m/s | 8 |
| Pole angle | $[-0.25,\ 0.25]$ rad | 8 |
| Pole angular velocity | $[-3.5,\ 3.5]$ rad/s | 8 |

### 2.2. Key Design Choices

- **Q-table representation.** A `defaultdict` keyed by the discretised state tuple, initialised to zero vectors. This is memory-efficient as only visited states are allocated.
- **Learning rule.** Standard Q-learning (Bellman optimality backup) with a scalar learning rate $\alpha$ and discount factor $\gamma = 0.99$.
- **$\varepsilon$-greedy exploration.** Epsilon decays from $1.0$ to $0.01$ over the first half of training. Both linear and exponential schedules are compared.
- **Best-checkpoint tracking.** The agent is evaluated every 500 episodes from the halfway point of training; the Q-values achieving the highest mean return over 200 evaluation episodes are deep-copied and restored before returning, so the sweep records peak rather than end-of-training performance.
- **Final evaluation.** After training, the restored best-checkpoint agent is evaluated over 1,000 greedy episodes to obtain statistically reliable mean, max, and standard deviation of return.

The core training loop is standard:

<figure id="alg-qlearning-cartpole" style="text-align:center;">
<div style="text-align:left;">

```
Algorithm: Tabular Q-Learning (CartPole)

Input:
  α         learning rate
  γ         discount factor ∈ (0, 1]
  ε_0, ε_f  initial and final exploration rates
  T         number of training episodes
  B         bin edges per observation dimension

Output:
  Q         learned action-value table

Initialise Q(s, a) = 0 for all s, a (via defaultdict)
For episode e = 1, …, T:
    obs ← env.reset()
    While not done:
        s ← discretise(obs, B)
        a ← ε-greedy action from Q(s, ·)
        obs′, r, done ← env.step(a)
        s′ ← discretise(obs′, B)
        Q(s, a) ← Q(s, a) + α [r + γ max_a′ Q(s′, a′) − Q(s, a)]
        obs ← obs′
    ε ← decay(ε)
    If checkpoint interval reached (second half only):
        Probe mean return; update best Q if improved
Restore Q from best checkpoint
Return Q
```

</div>
<figcaption>Algorithm 1: Tabular Q-learning with best-checkpoint tracking for CartPole-v1.</figcaption>
</figure>

### 2.3. Hyperparameter Sweep

A grid search was run over:

- **Learning rates:** $\alpha \in \{0.05, 0.1, 0.3, 0.5\}$.
- **Decay strategies:** linear and exponential, both reaching $\varepsilon_f = 0.01$ by episode 25,000 (half of the 50,000-episode budget).

All other hyperparameters were fixed: $\gamma = 0.99$, $\varepsilon_0 = 1.0$, and 50,000 training episodes.

---

## 3. Results

### 3.1. Sweep Summary

The table below reports final evaluation performance (1,000 greedy episodes) for all eight configurations, sorted by mean return:

| Learning rate | Decay strategy | Best episode | Mean return | Max return | Std dev return |
|---|---|---|---|---|---|
| **0.1** | **linear** | **25,000** | **500.0** | **500** | **0.0** |
| 0.5 | linear | 31,000 | 500.0 | 500 | 0.0 |
| 0.5 | exponential | 44,500 | 500.0 | 500 | 0.0 |
| 0.1 | exponential | 33,500 | 499.985 | 500 | 0.47 |
| 0.3 | linear | 28,000 | 499.768 | 500 | 7.33 |
| 0.05 | exponential | 30,500 | 499.222 | 500 | 13.35 |
| 0.3 | exponential | 31,000 | 499.06 | 500 | 21.00 |
| 0.05 | linear | 27,000 | 498.979 | 500 | 10.26 |

Observations:

- **Linear decay dominates.** All four linear-decay configurations reach or nearly reach a mean return of 500, whereas their exponential counterparts show higher variance in three of four cases.
- **The best configuration is $\alpha = 0.1$, linear decay**, achieving a perfect mean return of 500.0 with zero standard deviation across 1,000 evaluation episodes.
- **High learning rates with exponential decay are riskier.** The $\alpha = 0.3$, exponential configuration achieves a reasonable mean (499.06) but the highest variance of all (std = 21.0), suggesting the policy is less stable.
- **Low learning rates ($\alpha = 0.05$) converge reliably but noisily** under both schedules, indicating insufficient step size leads to incomplete value propagation within the episode budget.
- **Best checkpoints for exponential decay occur later** (episodes 30,500–44,500) compared to linear decay (25,000–31,000), despite both schedules reaching $\varepsilon_f$ at episode 25,000. Exponential decay drives $\varepsilon$ down more aggressively early on, reducing exploration before the Q-table is well-estimated. Meanwhile, linear decay maintains higher exploration through the early-to-mid training phase, providing more uniform state-space coverage and likely more stable convergence.

### 3.2. Training Curves (Best Configuration)

[Figure 1](#fig-training) shows the training dynamics of the best configuration (learning rate 0.1, linear decay).

<figure id="fig-training" style="text-align: center;">
  <img src="/assets/images/cartpole_training_20c69788.png" alt="Training curves for config 20c69788." style="width: 100%;">
  <figcaption>Figure 1: Training curves for the best configuration (learning rate 0.1, linear decay). Left: episode returns (500-episode rolling mean). Centre: episode lengths. Right: TD error per step.</figcaption>
</figure>

Observations:

- **Rapid performance improvement around episode 20,000.** Returns climb sharply from ~100 to the 500-step ceiling within a few thousand episodes, coinciding with the final phase of $\varepsilon$-decay.
- **Episode lengths mirror returns exactly.** In CartPole the reward is +1 per step, so episode length and return are identical; the two left panels are therefore redundant, but confirm internal consistency.
- **TD error shrinks and stabilises.** The temporal-difference (TD) error begins with large magnitude during early exploration and contracts to a low-amplitude oscillation around zero by roughly $10^7$ steps, indicating the Q-values have largely converged.
- **Persistent variance in returns after convergence.** Even after episode 25,000 the smoothed return fluctuates slightly below 500; the std = 0 in the *evaluation* phase ($\varepsilon = 0$) confirms these dips are caused by residual exploration noise during training rather than policy instability.

### 3.3. Learned Policy (Best Configuration)

[Figure 2](#fig-policy) shows a 2-D slice of the learned policy projected onto the two dimensions most relevant to the balancing task: pole angle and pole angular velocity. Cart position and cart velocity are fixed at their centre bins.

<figure id="fig-policy" style="text-align: center;">
  <img src="/assets/images/cartpole_policy_20c69788.png" alt="Policy slice for config 20c69788." style="width: 90%;">
  <figcaption>Figure 2: Learned policy slice (0 = push left, 1 = push right) for the best configuration, projected onto pole angle × pole angular velocity with cart dimensions fixed at centre bins.</figcaption>
</figure>

Observations:

- **The policy is physically interpretable in the central angular-velocity band.** For moderate $|\omega|$, the agent pushes right (green) when the pole angle is positive (leaning right) and left (red) when negative — consistent with the correct corrective strategy of pushing in the direction of the lean.
- **Extreme angular velocities map uniformly to 'push left'.** Both the top ($\omega \approx −3.1 rad/s$) and bottom ($\omega \approx +3.1 rad/s$) bands are uniformly red. The top band is physically sensible: a pole spinning fast leftward is unrecoverable. The bottom band is less interpretable and likely reflects a discretisation artefact, i.e. these near-terminal states are rarely visited during successful episodes, so Q-values there are poorly estimated.
- **The staircase boundary** between red and green regions is a direct consequence of the 8-bin discretisation: smooth optimal decision boundaries can only be approximated as piecewise-constant step functions over the discrete state space.

---

## 4. Summary

Tabular Q-learning solves CartPole-v1 despite the continuous state space, provided the observation space is discretised appropriately. The key implementation challenge is choosing bin granularity and ranges that balance coverage against table size; 8 bins per dimension ($8^4 = 4{,}096$ states) proved sufficient. The sweep identifies $\alpha = 0.1$ with linear $\varepsilon$-decay as the most reliable configuration, converging to a perfect mean return of 500 by episode 25,000 with zero evaluation variance. Linear decay consistently outperforms exponential decay in this setting, and the learned policy exhibits physically interpretable structure in the angle–angular velocity plane despite the constraints of discrete state representation.
