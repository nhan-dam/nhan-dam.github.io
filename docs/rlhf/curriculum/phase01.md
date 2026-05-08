# Phase 1 — Reinforcement Learning Foundations

> Created on: 22 April 2026
>
> Updated on: 28 April 2026

## 1. Module 1: The RL Problem

### 1.1. Theory

Reinforcement learning studies agents that learn by interacting with an environment.

<figure markdown="span" id="fig-ae-loop" style="text-align:center;">
  <img src="/assets/images/AE_loop.png" alt="The agent-environment interaction loop in reinforcement learning." style="width: 40%;">
  <figcaption>Figure 1: Agent-environment interaction in RL. Source: https://gymnasium.farama.org/_images/AE_loop.png.</figcaption>
</figure>

As shown in [Figure 1](#fig-ae-loop), at each time step the agent observes the current state of the environment and receives a scalar reward; it then selects an action, which causes the environment to transition to a new state. This cycle of observation, action, and reward is the fundamental unit of interaction in every RL algorithm covered in this course, from Q-learning in [Section 2.1](#21-module-1-the-rl-problem) to PPO-based RLHF in [Section 3.3](#33-module-5-the-full-ppo-rlhf-loop). In the language model setting, the 'environment' is the human (or reward model) that evaluates the generated response, and the 'action' at each step is the next token emitted by the policy.

The standard formalism of RL is the **Markov Decision Process (MDP)**, a tuple $(S, A, P, R, \gamma)$.

- $S$ is the state space, the complete description of the world at a given time step.
- $A$ is the action space, the set of choices available to the agent.
- $P(s' \mid s, a)$ is the transition dynamics, the probability of landing in state $s'$ after taking action $a$ from state $s$.
- $R(s, a, s')$ is the reward function, a scalar signal that encodes what the agent should maximise.
- $\gamma \in [0, 1)$ is the discount factor, which down-weights future rewards relative to immediate ones.

The most general definition of the reward function is $R(s, a, s')$: the reward depends on the state the agent was in, the action it took, and the state it landed in. In many formulations this is simplified to $R(s, a)$ by marginalising out $s'$ under the transition dynamics, $R(s, a) = \mathbb{E}_{s' \sim P(\cdot \mid s, a)}[R(s, a, s')]$, and occasionally further to $R(s)$ when the reward depends only on the destination state. All three conventions are equivalent in expressiveness; the choice is a matter of notational convenience. In the RLHF setting, the reward collapses to $r_\phi(x, y)$, a function of the full (prompt, response) pair, which maps cleanly onto the $R(s, a)$ convention.

The discount factor $\gamma$ serves three related purposes. First, it ensures the infinite-horizon return $\sum_{t=0}^\infty \gamma^t R_t$ is bounded by $R_{\max}/(1-\gamma)$, making the optimisation problem well-posed; without discounting, the sum may diverge. Second, it encodes a preference for earlier rewards over later ones: a reward received $k$ steps in the future is worth only $\gamma^k$ of its face value today, reflecting the intuition that future outcomes are less certain and that earlier rewards reduce the variance of the return estimator. Third, $\gamma$ implicitly defines an effective planning horizon of approximately $1/(1-\gamma)$ steps, beyond which future rewards contribute negligibly. This is a useful design lever: tasks requiring long-horizon credit assignment need $\gamma$ close to 1, but higher $\gamma$ also increases the variance of return estimates. In the short-episode setting of RLHF, where a single (prompt, response) pair constitutes an entire episode, $\gamma$ is typically set to 1.0 or 0.99.

The core structural assumption of an MDP is the **Markov property**: the future depends only on the present, not on the history of how the present was reached. Formally:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t).
$$

The current state $s_t$ is a sufficient statistic for all future states, i.e. knowing the full trajectory history gives no additional predictive power beyond knowing $s_t$ alone. Everything else in the MDP framework, including the Bellman equations, Q-learning convergence, and policy optimisation, is a consequence of this property holding. In practice the Markov property is often violated (large language model (LLM) generation is non-Markovian in the strict sense, since the full token history constitutes the state), but the approximation is usually workable.

The agent's behaviour is described by a **policy** $\pi(a \mid s)$, a distribution over actions conditioned on the current state. The objective is to find a policy that maximises the expected discounted return:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right],
$$

where $\tau = (s_0, a_0, s_1, a_1, \ldots)$ is a trajectory sampled by running the policy in the environment.

Two quantities are central to almost every RL algorithm.

The **state-value function** $V^\pi(s)$ estimates the expected return starting from state $s$ and following policy $\pi$ thereafter:

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \mid s_0 = s \right].
$$

The **action-value function** $Q^\pi(s, a)$ estimates the expected return starting from state $s$, taking action $a$, and then following $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \mid s_0 = s, a_0 = a \right].
$$

The relationship $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$ connects the two. The **advantage function**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

is particularly important. It measures how much better (or worse) action $a$ is relative to the average action under the current policy. A positive advantage means the action was better than expected; a negative advantage means it was worse. The advantage is the right signal to use when updating the policy.

**Connection to deep learning.** In modern RL, the policy $\pi_\theta(a \mid s)$ is a neural network with parameters $\theta$. For a discrete action space, the output is a softmax over actions. For a continuous action space, the output parameterises a distribution (e.g. a Gaussian). In the LLM setting, the 'state' is the token sequence so far, the 'action' is the next token, and the policy is precisely the language model's conditional distribution over the vocabulary.

#### 1.1.1. Value-Based Methods: Q-Learning and Deep Q-Network

Q-learning is a **model-free**, **off-policy** algorithm that learns $Q^*(s, a)$, the optimal action-value function, by iteratively applying the **Bellman optimality equation**:

$$
Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right].
$$

This equation expresses a self-consistency condition: the value of taking action $a$ in state $s$ must equal the immediate reward plus the discounted value of acting optimally from every possible next state $s'$, weighted by the transition probability. The expectation over $s'$ is what makes the equation exact; computing it analytically requires a known model $P$, which is the defining property of model-based, dynamic programming (DP) methods such as value iteration. Q-learning dispenses with the model by replacing the expectation with a single sampled transition: rather than summing over all possible $s'$, the agent simply observes the $s'$ that actually occurs and uses it as an unbiased estimate of the target. The Q-learning update rule is therefore:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_t \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right],
$$

where the bracketed term is the **temporal-difference (TD) error**, the difference between the current estimate and the single-sample Bellman target, and $\alpha_t$ is a step-dependent learning rate.

This single-sample approximation is valid because the observed $s_{t+1}$ is drawn from $P(\cdot \mid s_t, a_t)$, so the target $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ is a noisy but unbiased estimate of the true Bellman target. The $\alpha_t$-weighted update nudges $Q(s_t, a_t)$ a small step towards this noisy target. Every subsequent visit to the same $(s, a)$ pair yields a fresh independent sample of the target, and the sequence of updates averages out the noise over time, by the same mechanism that makes stochastic gradient descent converge to the same minimum as full-batch gradient descent.

Formally, convergence to $Q^*$ is guaranteed provided every $(s, a)$ pair is visited infinitely often and $\alpha_t$ satisfies the **Robbins-Monro conditions**:

$$
\sum_t \alpha_t = \infty \qquad \text{and} \qquad \sum_t \alpha_t^2 < \infty.
$$

The two conditions answer complementary questions about how $\alpha_t$ should decay. The first condition requires that the steps never shrink so aggressively that the algorithm effectively stops learning before it has corrected its early estimation errors; if the total step size were finite, updates would freeze prematurely. The second condition requires that steps eventually become small enough for the noise in each single-sample update to average out rather than accumulate; if steps stayed large forever, the estimate would keep bouncing and never settle. A concrete schedule satisfying both is $\alpha_t = 1/t$: the harmonic series $\sum 1/t$ diverges (first condition met), while $\sum 1/t^2 = \pi^2/6$ is finite (second condition met).

In practice, Deep Q-Network (DQN) uses a constant learning rate $\alpha_t = \alpha$, which violates the second condition and therefore lacks the tabular convergence guarantee. This is acceptable because the target network and replay buffer provide stability through other means, and because neural network function approximation introduces approximation error that a decaying learning rate cannot eliminate anyway.

Q-learning is therefore best understood as **stochastic dynamic programming**: it applies the Bellman operator sample-by-sample rather than in full sweeps, inheriting the fixed-point guarantee of DP while requiring no model of the environment.

Once $Q^*$ has been learned, the connection to the original RL objective $J(\pi)$ is immediate. The optimal policy is recovered greedily:

$$
\pi^*(s) = \arg\max_a Q^*(s, a),
$$

and this greedy policy provably maximises $J(\pi)$. Q-learning thus solves the RL objective indirectly: rather than differentiating through $J(\pi)$ as policy gradient methods do, it learns $Q^*$ as an intermediate object from which the optimal policy is read off for free.

In the tabular setting, $Q(s, a)$ is a lookup table with one entry per $(s, a)$ pair, and the update touches exactly one cell per transition. Deep Q-Networks (DQN) replace the table with a neural network and add two stabilising tricks that are worth understanding because they recur in later algorithms: *experience replay* and *target network*.

**Experience replay.** Without intervention, the transitions $(s_t, a_t, r_t, s_{t+1})$ fed to the network arrive in temporal order, meaning consecutive samples are highly correlated, as the same region of state space is visited for many steps in a row. Training a neural network on correlated samples violates the i.i.d. assumption that gradient descent relies on, producing biased gradient estimates and causing the network to overfit to the current region of the environment while catastrophically forgetting others. Experience replay breaks this correlation by storing every observed transition in a fixed-size circular buffer $\mathcal{D}$ and sampling uniformly at random from it at each update step. [Algorithm 1](#alg-exp-rep) illustrates this training technique.

Action selection at each step follows an **$\varepsilon$-greedy policy**, a simple strategy for balancing exploration (trying actions whose Q-values are uncertain) and exploitation (taking the action the current network considers best). At each step the agent draws $u \sim \text{Uniform}(0, 1)$: if $u < \varepsilon$ it selects a random action; otherwise it selects $\arg\max_a Q(s_t, a; \theta)$. $\varepsilon$ is annealed from 1.0 at the start of training, when the network knows nothing and pure exploration is warranted, down to a small value such as 0.05 once the network has learned a reasonable policy. Without this exploration mechanism, the agent may never visit large parts of the state space, violating the condition that every $(s, a)$ pair must be visited sufficiently often for Q-learning to converge.

<figure id="alg-exp-rep" style="text-align:center;">
<div style="text-align:left;">

```
Input:  Q(·; θ)  — online Q-network with parameters θ
        N        — replay buffer capacity
        k        — mini-batch size
        ε        — exploration rate (annealed over training)

Output: θ        — updated Q-network parameters

Initialise replay buffer D with capacity N
For each time step t:
    Draw u ~ Uniform(0, 1)
    If u < ε:
        a_t = random action from A           # explore
    Else:
        a_t = argmax_a Q(s_t, a; θ)         # exploit
    Execute a_t, observe r_t and s_{t+1}
    Store (s_t, a_t, r_t, s_{t+1}) in D     # overwrite oldest entry if full
    Sample a random mini-batch of k transitions from D
    Compute Bellman targets and update Q-network
Return θ
```

</div>
<figcaption>Algorithm 1: DQN training loop with experience replay.</figcaption>
</figure>

The buffer decouples data collection from data consumption. A transition collected early in training may be replayed many times later, improving sample efficiency. The random sampling ensures each mini-batch approximates an i.i.d. draw from the agent's historical experience, recovering the regime in which supervised learning is well-behaved.

**Target network.** The Bellman target for a given transition is $r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$, which depends on the same parameters $\theta$ being updated. This creates a moving-target problem: every gradient step changes not only the Q-value being corrected but also the target it is being corrected towards, producing oscillations or divergence. The target network resolves this by maintaining a separate copy of the Q-network with parameters $\theta^-$, held frozen for $C$ update steps at a time. The Bellman target is computed using $\theta^-$, not $\theta$. This training technique is shown in [Algorithm 2](#alg-target-net).

<figure id="alg-target-net" style="text-align:center;">
<div style="text-align:left;">

```
Input:  Q(·; θ)  — online Q-network with parameters θ
        D        — replay buffer populated by experience replay
        k        — mini-batch size
        C        — target network update frequency (steps)
        γ        — discount factor

Output: θ        — updated Q-network parameters

Initialise online network Q(·; θ) and target network Q(·; θ⁻) with θ⁻ ← θ
For each update step:
    Sample mini-batch of k transitions from D
    For each transition (s, a, r, s'):
        y = r + γ · max_a' Q(s', a'; θ⁻)     # frozen target network
    Loss = MSE(Q(s, a; θ), y)
    Update θ via gradient descent on Loss
    Every C steps: θ⁻ ← θ                    # periodic hard copy
Return θ
```

</div>
<figcaption>Algorithm 2: DQN training loop with target network.</figcaption>
</figure>

During the $C$ steps between copies, the target is stationary, giving the online network a stable regression objective. The periodic hard copy then refreshes the target to track the latest policy. A soft update variant, $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ with $\tau \ll 1$ (e.g. $\tau = 0.005$), is common in continuous-control algorithms such as DDPG and SAC. Unlike the hard copy, the soft update is applied at every step rather than every $C$ steps, so the target is never fully stationary; instead it drifts imperceptibly slowly, trading the clean frozen window of the hard copy for smoother, continuous target evolution.

Both tricks are instances of a general principle: RL training is destabilised by correlation and non-stationarity in the targets. Recognising this is the single most useful debugging intuition for all of Phase 1.

#### 1.1.2. Best Practices for ε-Decay in Q-Learning
 
**Overview**

In Q-learning, ε (epsilon) controls the balance between exploration, taking random actions to discover new information, and exploitation, selecting the action with the highest known Q-value. As training progresses, the agent should gradually shift from exploring to exploiting. The rate at which ε is reduced, i.e. the decay schedule, meaningfully affects both the speed and stability of learning.

**Decay Schedule**

Exponential decay is generally preferred over linear decay. Because exponential decay reduces ε multiplicatively each episode, 

$$\varepsilon \leftarrow \max(\varepsilon_{\min},\ \varepsilon \cdot \lambda),$$

where $\lambda \in (0, 1)$ is the decay rate (e.g. $\lambda = 0.9995$), it frontloads exploration into the early phase of training, where the Q-table is largely uninformed and random actions are most valuable. Linear decay, by contrast, reduces exploration at a constant rate regardless of how much the agent has already learnt, which can leave the agent over-exploring in later episodes when exploitation would be more productive.
 
**Minimum Epsilon**
 
ε should never be decayed to zero. Retaining a small minimum value, typically $\varepsilon_{\min} \in [0.05, 0.1]$, ensures the agent continues to explore occasionally throughout training. A fully greedy policy is brittle: any errors remaining in the Q-table will never be corrected if the agent stops exploring entirely.
 
**Timing of Decay**

A widely used heuristic is to finish the decay schedule approximately halfway through training, leaving the second half predominantly for exploitation-based refinement of the Q-values already acquired. This can be configured by choosing $\lambda$ such that $\varepsilon$ reaches $\varepsilon_{\min}$ at episode $N/2$, where $N$ is the total number of training episodes.
 
**Granularity of Decay**

Decay should be applied per episode rather than per step, unless episodes vary substantially in length. In environments where episode lengths are highly inconsistent, per-step decay prevents the agent from losing exploration capacity too rapidly during long episodes.

### 1.2. Intuitions for When RL Works

RL tends to be reliable when the reward signal is dense (frequent feedback per step), unambiguous, and cheap to compute. It tends to fail when rewards are sparse (the agent must stumble upon the correct behaviour before it can learn from it), when the reward function is misspecified (the agent learns to optimise a proxy that diverges from true intent), or when the environment dynamics are highly stochastic. These conditions map directly onto failure modes in RLHF.

### 1.3. Suggested Reading

Mnih et al. (2015), 'Human-level control through deep reinforcement learning.' Read before the project; the paper introduces the exact architecture and training procedure you are about to implement.

#### 1.3.1. Overview

Mnih et al. (2015) introduced the Deep Q-Network (DQN), a reinforcement learning (RL) agent capable of learning to play 49 Atari 2600 games at human-level performance from raw pixel input and game scores alone, with no hand-crafted features or game-specific knowledge. The work represented a significant milestone in demonstrating that deep neural networks could serve as stable, general-purpose function approximators within an RL framework.

#### 1.3.2. Problem

Prior RL methods were ill-suited to high-dimensional sensory inputs such as raw video frames. Naïvely combining deep neural networks with RL leads to training instability, due to two structural problems: strong temporal correlations between consecutive training samples, and non-stationary learning targets, i.e. the Q-value targets shift as the network parameters are updated.

#### 1.3.3. Key Innovations

DQN addressed both instabilities with two complementary techniques.

- **Experience replay.** Agent transitions (state, action, reward, next state) are stored in a replay buffer and sampled uniformly at random during training. This breaks temporal correlations between samples and enables data reuse.
- **Target network.** A separate copy of the Q-network, updated only periodically, is used to compute temporal-difference (TD) targets. This stabilises training by holding the targets fixed across multiple gradient updates.

#### 1.3.4. Architecture

The network receives the four most recent game frames as input, preprocessed to 84×84 grayscale. These pass through convolutional layers that extract spatial features, followed by fully connected layers that output a Q-value for each possible action. Crucially, the same architecture and hyperparameters were applied across all 49 games without modification. [Algorithm 3](#alg-dqn) shows the training loop of DQN.

<figure id="alg-dqn" style="text-align:center;">
<div style="text-align:left;">

```
Algorithm: DQN Training

Input:
  env        Atari game environment
  N          replay buffer capacity
  C          target network update frequency
  γ ∈ (0,1]  discount factor

Output:
  θ          trained Q-network parameters

Initialise replay buffer D with capacity N
Initialise Q-network with parameters θ
Initialise target network with parameters θ⁻ ← θ

For each episode:
  Observe initial state s
  While s is not terminal:
    Select action a via ε-greedy policy under Q(s, ·; θ)
    Execute a; observe reward r and next state s'
    Store (s, a, r, s') in D
    Sample random minibatch from D
    Compute TD target: y = r + γ max_{a'} Q(s', a'; θ⁻)
    Update θ by minimising (y − Q(s, a; θ))²
    Every C steps: θ⁻ ← θ
    s ← s'
Return θ
```

</div>
<figcaption>Algorithm 3: DQN training loop with experience replay and target network.</figcaption>
</figure>

#### 1.3.5. Results

DQN outperformed all prior RL methods and achieved human-level or superhuman performance on the majority of the 49 games tested. Performance was strongest on games requiring visual pattern recognition (e.g. Breakout, Pong) and weakest on games demanding long-horizon planning (e.g. Montezuma's Revenge), where sparse rewards make credit assignment difficult.

#### 1.3.6. Significance

The paper established that end-to-end learning from raw sensory data is feasible at scale using a single, fixed architecture. It laid the foundation for much of modern deep RL research, directly motivating subsequent advances including Double DQN, Duelling DQN, and Prioritised Experience Replay.

### 1.4. Hands-on Project: DQN on CartPole

**Objective.** Implement DQN from scratch using PyTorch and solve the `CartPole-v1` environment from Gymnasium.

**Setup.**

```bash
cd /Volumes/ML_Workspace/projects/
mkdir rl-foundations && cd rl-foundations
uv init --python 3.12
uv add torch torchvision gymnasium numpy matplotlib rich
```

**Implementation outline.**

1. Define a two-hidden-layer MLP `QNetwork(obs_dim, n_actions)` that outputs per-action Q-values.
2. Implement a `ReplayBuffer` storing $(s, a, r, s', \text{done})$ tuples, supporting random mini-batch sampling.
3. Implement the DQN training loop:
   - Collect transitions by running the $\varepsilon$-greedy policy (anneal $\varepsilon$ from 1.0 to 0.05 over 10 000 steps).
   - Sample a mini-batch and compute the Bellman target using the target network.
   - Compute the MSE loss against the online network's Q-value for the taken action.
   - Update the online network with Adam; copy weights to the target network every 100 steps.
4. Log episodic return with `rich` and plot the learning curve.

**What to observe.** CartPole should solve (mean return > 475 over 100 episodes) within approximately 50 000 environment steps. If training is unstable, the first thing to check is whether the replay buffer is large enough (at least 10 000 transitions) and whether $\varepsilon$ is decaying too quickly.