# Tabular Q-Learning on Gymnasium Blackjack-v1

> Created on: 29 April 2025
>
> Updated on: 7 May 2025

## 1. Overview

This article documents the implementation and evaluation of a tabular Q-learning agent on the `Blackjack-v1` environment from [Gymnasium](https://gymnasium.farama.org/). The work covers a clean implementation of the Q-learning update rule, an epsilon-decay sweep across learning rates and decay strategies, and an analysis of the resulting policy.

---

## 2. Environment

The `Blackjack-v1` environment models a simplified, single-player game of Blackjack against a fixed dealer. Key properties:

- **State space:** a 3-tuple `(player_sum, dealer_showing, usable_ace)`, where `player_sum ∈ [4, 21]`, `dealer_showing ∈ [1, 10]`, and `usable_ace ∈ {True, False}`.
- **Action space:** binary — `0` (stick) or `1` (hit).
- **Rewards:** `+1` for a win, `-1` for a bust or loss, `0` for a draw. Blackjack pays the same as a regular win (i.e. no 1.5× bonus) by default.
- **`sab=False`:** the natural blackjack flag is disabled, so all wins yield `+1`.

The state space is small enough for exact tabular representation, making this an ideal testbed for Q-learning before scaling to function approximation.

---

## 3. Implementation

### 3.1. Q-Table and Update Rule

The Q-table is implemented as a `defaultdict` mapping each `(state, action)` pair to a scalar value initialised at zero. On each time step, the agent applies the standard temporal-difference (TD) update,

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right],$$

where $\alpha$ is the learning rate, $\gamma$ is the discount factor, and $r + \gamma \max_{a'} Q(s', a')$ is the TD target. The term in brackets is the temporal-difference (TD) error, which measures the inconsistency between the current estimate and the one-step bootstrapped target.

### 3.2. Action Selection

The agent uses an $\varepsilon$-greedy policy: with probability $\varepsilon$ it samples a uniformly random action (exploration), and with probability $1 - \varepsilon$ it selects the action with the highest Q-value for the current state (exploitation).

### 3.3. Epsilon Decay

Two decay strategies were compared:

- **Linear decay:** $\varepsilon_{t+1} = \max(\varepsilon_\text{min},\; \varepsilon_t - \delta)$, where $\delta = (\varepsilon_0 - \varepsilon_\text{min}) / (N/2)$.
- **Exponential decay:** $\varepsilon_{t+1} = \max(\varepsilon_\text{min},\; \varepsilon_t \cdot \rho)$, where $\rho = (\varepsilon_\text{min} / \varepsilon_0)^{2/N}$.

Both strategies are calibrated so that $\varepsilon$ reaches `final_epsilon` at the halfway point of training ($N/2$ episodes), leaving the second half for near-greedy exploitation and fine-tuning.

Key parameter values:

| Parameter | Value |
|---|---|
| `initial_epsilon` | 1.0 |
| `final_epsilon` | 0.1 |
| `discount_factor` ($\gamma$) | 0.95 |
| `n_episodes` | 100,000 |

### 3.4. Best-Checkpoint Selection

Rather than returning the final agent, training periodically evaluates the greedy policy (with $\varepsilon = 0$) every 500 episodes during the second half of training, using 5,000 evaluation episodes per checkpoint. The checkpoint with the highest win rate is restored at the end of training. This guards against late-training regression caused by residual noise in the Q-table.

### 3.5. Hyperparameter Sweep

A grid sweep was run over:

- **Learning rates:** `{0.001, 0.01, 0.05, 0.1}`.
- **Decay strategies:** `{linear, exponential}`.

Each configuration was identified by an 8-character MD5 hash of its hyperparameters to allow unambiguous file naming across runs.

---

## 4. Results

### 4.1. Sweep Summary

The table below shows the final evaluation results (over 10,000 test episodes) for all eight configurations, sorted by win rate.

| Learning rate | Decay strategy | Best episode | Win rate | Average reward |
|---|---|---|---|---|
| **0.05** | **linear** | **63,500** | **0.4353** | **−0.0401** |
| 0.01 | exponential | 51,000 | 0.4303 | −0.0468 |
| 0.01 | linear | 54,000 | 0.4278 | −0.0524 |
| 0.001 | exponential | 98,500 | 0.4280 | −0.0591 |
| 0.1 | exponential | 88,500 | 0.4280 | −0.0518 |
| 0.1 | linear | 94,500 | 0.4237 | −0.0584 |
| 0.05 | exponential | 83,000 | 0.4190 | −0.0660 |
| 0.001 | linear | 94,000 | 0.4173 | −0.0761 |

Key observations:

- The best configuration is `lr=0.05` with linear decay, achieving a **win rate of 43.5%** and average reward of **−0.040**, which compares favourably with the theoretical optimal of approximately −0.029 for this variant of the game.
- Win rates across configurations are tightly clustered in the range `[41.7%, 43.5%]`, suggesting the task is not highly sensitive to these hyperparameters once the learning rate is in a reasonable range.
- `lr=0.001` consistently underperforms, likely because updates are too small to propagate reward signals efficiently within 100,000 episodes.
- `lr=0.05` with exponential decay is a notable outlier: it performs well with linear decay but drops to 41.9% with exponential. The faster early decay under exponential may cause the agent to commit to a suboptimal policy before the Q-table is sufficiently populated.

### 4.2. Training Curves (Best Configuration)

[Figure 1](#fig-training) shows the smoothed training statistics for the best configuration (`lr=0.05`, linear decay), averaged over a 500-episode rolling window.

<figure id="fig-training" style="text-align: center;">
  <img src="/assets/images/blackjack_training_stats_6b1a67a6.png" alt="Training statistics for the best configuration." style="width: 100%;">
  <figcaption>Figure 1: Training statistics for the best configuration (lr=0.05, linear decay). Left: episode reward. Centre: episode length. Right: TD error.</figcaption>
</figure>

Observations:

- **Episode reward** rises steadily from approximately −0.40 in early training to around −0.08 by episode 100,000, indicating consistent policy improvement.
- **Episode length** grows from ~1.33 to ~1.55 actions per hand. This reflects the agent learning to hit more aggressively on weak hands early in training, then gradually moderating as the policy converges.
- **TD error** exhibits a large transient negative spike in the first ~10,000 steps as the Q-table is populated from zero, then converges towards zero with diminishing variance. Near-zero TD error in the second half of training confirms that the value estimates have largely stabilised.

### 4.3. Learnt Policy (Best Configuration)

[Figure 2](#fig-policy-ace) and [Figure 3](#fig-policy-noace) show the learnt state values and greedy policy for the best configuration, split by whether the player holds a usable ace.

<figure id="fig-policy-ace" style="text-align: center;">
  <img src="/assets/images/blackjack_policy_ace_6b1a67a6.png" alt="Learnt policy with a usable ace." style="width: 100%;">
  <figcaption>Figure 2: Learnt state values (left) and greedy policy (right) when the player holds a usable ace.</figcaption>
</figure>

<figure id="fig-policy-noace" style="text-align: center;">
  <img src="/assets/images/blackjack_policy_noace_6b1a67a6.png" alt="Learnt policy without a usable ace." style="width: 100%;">
  <figcaption>Figure 3: Learnt state values (left) and greedy policy (right) when the player does not hold a usable ace.</figcaption>
</figure>

Observations:

- **With a usable ace ([Figure 2](#fig-policy-ace)):** the agent consistently hits on all hands up to player sum 17, regardless of the dealer's card, and sticks on 18–21. This is broadly consistent with basic strategy: a usable ace makes hitting less risky because the ace can revert from 11 to 1 to avoid a bust.
- **Without a usable ace ([Figure 3](#fig-policy-noace)):** the policy is more conservative and noisy. The agent sticks more aggressively, particularly at sums of 13–16 against a weak dealer (2–6), and hits more against a strong dealer (7–10). The policy partially recovers the classical basic-strategy boundary but retains some irregular cells, characteristic of residual Q-table noise at states visited infrequently.
- **State values:** the value surfaces are qualitatively correct — values rise with player sum and are higher when holding a usable ace, reflecting reduced bust risk. The no-ace surface shows a steeper gradient, consistent with the harder position of hard totals near the bust threshold.

---

## 5. Summary

Tabular Q-learning converges to a reasonable Blackjack policy within 100,000 episodes. The best configuration (`lr=0.05`, linear decay) achieves a win rate of 43.5%, close to the theoretical ceiling for this environment variant. The policy plots confirm that the agent has learnt the key structural features of basic strategy, with the greatest fidelity in the usable-ace regime where visit counts are higher and the Q-table is better populated.
