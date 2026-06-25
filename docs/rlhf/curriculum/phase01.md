# Phase 1 — Reinforcement Learning Foundations

> Created on: 22 April 2026
>
> Updated on: 12 June 2026

## 1. Module 1: The Reinforcement Learning Problem {: #rl-problem}

### 1.1. Theory

Reinforcement learning (RL) studies agents that learn by interacting with an environment.

<figure markdown="span" id="fig-ae-loop" style="text-align:center;">
  <img src="/assets/images/AE_loop.png" alt="The agent-environment interaction loop in reinforcement learning." style="width: 40%;">
  <figcaption>Figure 1: Agent-environment interaction in RL. Source: https://gymnasium.farama.org/_images/AE_loop.png.</figcaption>
</figure>

As shown in [Figure 1](#fig-ae-loop), at each time step the agent observes the current state of the environment and receives a scalar reward. It then selects an action, which causes the environment to transition to a new state. This cycle of observation, action, and reward is the fundamental unit of interaction in every RL algorithm covered in this course, from Q-learning in [Section 1](#rl-problem) to proximal policy optimisation (PPO)-based reinforcement learning from human feedback (RLHF) in [Section 5](phase02.md#ppo-rlhf-loop). In the language model setting, the 'environment' is the human (or reward model) that evaluates the generated response, and the 'action' at each step is the next token emitted by the policy.

The standard formalism of RL is the **Markov Decision Process (MDP)**, a tuple $(S, A, P, R, \gamma)$.

- $S$ is the state space, the complete description of the world at a given time step.
- $A$ is the action space, the set of choices available to the agent.
- $P(s' \mid s, a)$ is the transition dynamics, the probability of landing in state $s'$ after taking action $a$ from state $s$.
- $R(s, a, s')$ is the reward function, a scalar signal that encodes what the agent should maximise.
- $\gamma \in [0, 1)$ is the discount factor, which down-weights future rewards relative to immediate ones.

The most general definition of the reward function is $R(s, a, s')$: the reward depends on the state the agent was in, the action it took, and the state it landed in. In many formulations this is simplified to $R(s, a)$ by marginalising out $s'$ under the transition dynamics, $R(s, a) = \mathbb{E}_{s' \sim P(\cdot \mid s, a)}[R(s, a, s')]$, and occasionally further to $R(s')$ when the reward depends only on the destination state. All three conventions are equivalent in expressiveness. The choice is a matter of notational convenience. In the RLHF setting, the reward collapses to $r_\phi(x, y)$, a function of the full (prompt, response) pair, which maps cleanly onto the $R(s, a)$ convention.

The discount factor $\gamma$ serves three related purposes. First, it ensures the infinite-horizon return $\sum_{t=0}^\infty \gamma^t R_t$ is bounded by $R_{\max}/(1-\gamma)$, making the optimisation problem well-posed. Without discounting, the sum may diverge. Second, it encodes a preference for earlier rewards over later ones: a reward received $k$ steps in the future is worth only $\gamma^k$ of its face value today, reflecting the intuition that future outcomes are less certain and that earlier rewards reduce the variance of the return estimator. Third, $\gamma$ implicitly defines an effective planning horizon of approximately $1/(1-\gamma)$ steps, beyond which future rewards contribute negligibly. This is a useful design lever: tasks requiring long-horizon credit assignment need $\gamma$ close to 1, but higher $\gamma$ also increases the variance of return estimates. In the short-episode setting of RLHF, where a single (prompt, response) pair constitutes an entire episode, $\gamma$ is typically set to 1.0 or 0.99.

The core structural assumption of an MDP is the **Markov property**: the future depends only on the present, not on the history of how the present was reached. Formally:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t).
$$

The current state $s_t$ is a sufficient statistic for all future states, i.e. knowing the full trajectory history gives no additional predictive power beyond knowing $s_t$ alone. Everything else in the MDP framework, including the Bellman equations, Q-learning convergence, and policy optimisation, is a consequence of this property holding. Whether the Markov property holds depends on how the state is defined. In large language model (LLM) generation, the state is taken to be the prompt plus the entire token history so far, which makes the process Markovian by construction: each new token is appended to the state, so the state always carries everything the future depends on. The property genuinely fails when the observation is poorer than the underlying state (e.g. a single Atari frame, which shows object positions but not velocities), and the standard remedy is to enrich the state, such as stacking recent frames ([Section 1.3.6](#136-architecture)), until the Markov approximation becomes workable.

The agent's behaviour is described by a **policy** $\pi(a \mid s)$, a distribution over actions conditioned on the current state. The objective is to find a policy that maximises the expected discounted return:

$$
\boxed{J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right],}
$$

where $\tau = (s_0, a_0, s_1, a_1, \ldots)$ is a trajectory sampled by running the policy in the environment.

Two quantities are central to almost every RL algorithm.

The **state-value function** $V^\pi(s)$ estimates the expected return starting from state $s$ and following policy $\pi$ thereafter:

$$
\boxed{V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \mid s_0 = s \right].}
$$

The **action-value function** $Q^\pi(s, a)$ estimates the expected return starting from state $s$, taking action $a$, and then following $\pi$:

$$
\boxed{Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t R(s_t, a_t) \mid s_0 = s, a_0 = a \right].}
$$

The relationship $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$ connects the two. The **advantage function**,

$$
\boxed{A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)},
$$

is particularly important. It measures how much better (or worse) action $a$ is relative to the average action under the current policy. A positive advantage means the action was better than expected. A negative advantage means it was worse. The advantage is the right signal to use when updating the policy.

**Connection to deep learning.** In modern RL, the policy $\pi_\theta(a \mid s)$ is a neural network with parameters $\theta$. For a discrete action space, the output is a softmax over actions. For a continuous action space, the output parameterises a distribution (e.g. a Gaussian). In the LLM setting, the 'state' is the token sequence so far, the 'action' is the next token, and the policy is precisely the language model's conditional distribution over the vocabulary.

#### 1.1.1. Value-Based Methods: Q-Learning and Deep Q-Network {: #q-learning-dqn}

Q-learning is a **model-free**, **off-policy** algorithm that learns $Q^*(s, a)$, the optimal action-value function, by iteratively applying the **Bellman optimality equation**:

$$
\boxed{Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right].}
$$

This equation expresses a self-consistency condition: the value of taking action $a$ in state $s$ must equal the immediate reward plus the discounted value of acting optimally from every possible next state $s'$, weighted by the transition probability. The expectation over $s'$ is what makes the equation exact. Computing it analytically requires a known model $P$, which is the defining property of model-based, dynamic programming (DP) methods such as value iteration. Q-learning dispenses with the model by replacing the expectation with a single sampled transition: rather than summing over all possible $s'$, the agent simply observes the $s'$ that actually occurs and uses it as an unbiased estimate of the target. The Q-learning update rule is therefore:

$$
\boxed{Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_t \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right],}
$$

where the bracketed term is the **temporal-difference (TD) error**, the difference between the current estimate and the single-sample Bellman target, and $\alpha_t$ is a step-dependent learning rate.

This single-sample approximation is valid because the observed $s_{t+1}$ is drawn from $P(\cdot \mid s_t, a_t)$, so the target $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ is a noisy but unbiased estimate of the true Bellman target. The $\alpha_t$-weighted update nudges $Q(s_t, a_t)$ a small step towards this noisy target. Every subsequent visit to the same $(s, a)$ pair yields a fresh independent sample of the target, and the sequence of updates averages out the noise over time, by the same stochastic-approximation mechanism that lets stochastic gradient descent make progress even though each mini-batch gradient is only a noisy estimate of the full-batch gradient.

Formally, convergence to $Q^*$ is guaranteed provided every $(s, a)$ pair is visited infinitely often and $\alpha_t$ satisfies the **Robbins-Monro conditions**:

$$
\sum_t \alpha_t = \infty \qquad \text{and} \qquad \sum_t \alpha_t^2 < \infty.
$$

The two conditions answer complementary questions about how $\alpha_t$ should decay. The first condition requires that the steps never shrink so aggressively that the algorithm effectively stops learning before it has corrected its early estimation errors. If the total step size were finite, updates would freeze prematurely. The second condition requires that steps eventually become small enough for the noise in each single-sample update to average out rather than accumulate. If steps stayed large forever, the estimate would keep bouncing and never settle. A concrete schedule satisfying both is $\alpha_t = 1/t$: the harmonic series $\sum 1/t$ diverges (first condition met), while $\sum 1/t^2 = \pi^2/6$ is finite (second condition met).

In practice, Deep Q-Network (DQN) uses a constant learning rate $\alpha_t = \alpha$, which violates the second condition and therefore lacks the tabular convergence guarantee. This is acceptable because the target network and replay buffer provide stability through other means, and because neural network function approximation introduces approximation error that a decaying learning rate cannot eliminate anyway.

Q-learning is therefore best understood as **stochastic dynamic programming**: it applies the Bellman operator sample-by-sample rather than in full sweeps, inheriting the fixed-point guarantee of DP while requiring no model of the environment.

Once $Q^*$ has been learned, the connection to the original RL objective $J(\pi)$ is immediate. The optimal policy is recovered greedily:

$$
\pi^*(s) = \arg\max_a Q^*(s, a),
$$

and this greedy policy provably maximises $J(\pi)$. Q-learning thus solves the RL objective indirectly: rather than differentiating through $J(\pi)$ as policy gradient methods do, it learns $Q^*$ as an intermediate object from which the optimal policy is read off for free.

In the tabular setting, $Q(s, a)$ is a lookup table with one entry per $(s, a)$ pair, and the update touches exactly one cell per transition. DQN replaces the table with a neural network and adds two stabilising tricks that are worth understanding because they recur in later algorithms: *experience replay* and *target network*.

**Experience replay.** Without intervention, the transitions $(s_t, a_t, r_t, s_{t+1})$ fed to the network arrive in temporal order, meaning consecutive samples are highly correlated, as the same region of state space is visited for many steps in a row. Training a neural network on correlated samples violates the i.i.d. assumption that gradient descent relies on, producing biased gradient estimates and causing the network to overfit to the current region of the environment while catastrophically forgetting others. Experience replay breaks this correlation by storing every observed transition in a fixed-size circular buffer $\mathcal{D}$ and sampling uniformly at random from it at each update step. [Algorithm 1](#alg-exp-rep) illustrates this training technique.

Action selection at each step follows an **$\varepsilon$-greedy policy**, a simple strategy for balancing exploration (trying actions whose Q-values are uncertain) and exploitation (taking the action the current network considers best). At each step the agent draws $u \sim \text{Uniform}(0, 1)$: if $u < \varepsilon$ it selects a random action, otherwise it selects $\arg\max_a Q(s_t, a; \theta)$. $\varepsilon$ is annealed from 1.0 at the start of training, when the network knows nothing and pure exploration is warranted, down to a small value such as 0.05 once the network has learned a reasonable policy. Without this exploration mechanism, the agent may never visit large parts of the state space, violating the condition that every $(s, a)$ pair must be visited sufficiently often for Q-learning to converge.

<figure id="alg-exp-rep" style="text-align: center;" markdown="1">
<div style="border: 1px solid #ccc; display: inline-block; text-align: left; padding: 1em; font-family: monospace;" markdown="1">

**Input:**<br>
$Q(\cdot;\theta)$: online Q-network with parameters $\theta$<br>
$N$: replay buffer capacity<br>
$k$: mini-batch size<br>
$\varepsilon$: exploration rate (annealed over training)

**Output:**<br>
$\theta$: updated Q-network parameters

Initialise replay buffer $D$ with capacity $N$<br>
**for each** time step $t$:<br>
$\quad$ Draw $u \sim \mathrm{Uniform}(0,1)$<br>
$\quad$ **if** $u < \varepsilon$:<br>
$\quad\quad$ $a_t \leftarrow$ random action from $A$ *// explore*<br>
$\quad$ **else**:<br>
$\quad\quad$ $a_t \leftarrow \arg\max_a Q(s_t, a; \theta)$ *// exploit*<br>
$\quad$ Execute $a_t$, observe $r_t$ and $s_{t+1}$<br>
$\quad$ Store $(s_t, a_t, r_t, s_{t+1})$ in $D$ *// overwrite oldest entry if full*<br>
$\quad$ Sample a random mini-batch of $k$ transitions from $D$<br>
$\quad$ Compute Bellman targets and update Q-network

**return** $\theta$

</div>
<figcaption>Algorithm 1: DQN training loop with experience replay.</figcaption>
</figure>

The buffer decouples data collection from data consumption. A transition collected early in training may be replayed many times later, improving sample efficiency. The random sampling ensures each mini-batch approximates an i.i.d. draw from the agent's historical experience, recovering the regime in which supervised learning is well-behaved.

**Target network.** The Bellman target for a given transition is $r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$, which depends on the same parameters $\theta$ being updated. This creates a moving-target problem: every gradient step changes not only the Q-value being corrected but also the target it is being corrected towards, producing oscillations or divergence. The target network resolves this by maintaining a separate copy of the Q-network with parameters $\theta^-$, held frozen for $C$ update steps at a time. The Bellman target is computed using $\theta^-$, not $\theta$. This training technique is shown in [Algorithm 2](#alg-target-net).

<figure id="alg-target-net" style="text-align: center;" markdown="1">
<div style="border: 1px solid #ccc; display: inline-block; text-align: left; padding: 1em; font-family: monospace;" markdown="1">

**Input:**<br>
$Q(\cdot;\theta)$: online Q-network with parameters $\theta$<br>
$D$: replay buffer populated by experience replay<br>
$k$: mini-batch size<br>
$C$: target network update frequency (steps)<br>
$\gamma \in (0,1]$: discount factor

**Output:**<br>
$\theta$: updated Q-network parameters

Initialise online network $Q(\cdot;\theta)$ and target network $Q(\cdot;\theta^-)$ with $\theta^- \leftarrow \theta$<br>
**for each** update step:<br>
$\quad$ Sample mini-batch of $k$ transitions from $D$<br>
$\quad$ **for each** transition $(s, a, r, s')$:<br>
$\quad\quad$ $y \leftarrow r + \gamma \max_{a'} Q(s', a'; \theta^-)$ *// frozen target network*<br>
$\quad$ $\mathcal{L} \leftarrow \mathrm{MSE}(Q(s, a; \theta),\, y)$<br>
$\quad$ Update $\theta$ via gradient descent on $\mathcal{L}$<br>
$\quad$ Every $C$ steps: $\theta^- \leftarrow \theta$ *// periodic hard copy*

**return** $\theta$

</div>
<figcaption>Algorithm 2: DQN training loop with target network.</figcaption>
</figure>

During the $C$ steps between copies, the target is stationary, giving the online network a stable regression objective. The periodic hard copy then refreshes the target to track the latest policy. A soft update variant, $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ with $\tau \ll 1$ (e.g. $\tau = 0.005$), is common in continuous-control algorithms such as Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC). Unlike the hard copy, the soft update is applied at every step rather than every $C$ steps, so the target is never fully stationary. Instead it drifts imperceptibly slowly, trading the clean frozen window of the hard copy for smoother, continuous target evolution.

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

**Tabular Q-Learning Versus DQN**

The recommendations above are calibrated for the tabular setting, where each Q-table entry improves only through repeated visits to that specific state, making the episode a natural unit of learning progress. DQN inverts two of them: the standard choice (used by Mnih et al. (2015) and in the [Section 1.4](#project-req-dqn-cartpole) project) is *linear* annealing applied *per step*. The difference is principled rather than conventional. A neural network generalises across nearby states, so a single transition improves the value estimates of states never visited, making the individual step, not the episode, the unit at which exploration pressure should be reduced.

### 1.2. Intuitions for When Reinforcement Learning Works

RL tends to be reliable when the reward signal is dense (frequent feedback per step), unambiguous, and cheap to compute. CartPole ([Section 1.4](#project-req-dqn-cartpole)) is the archetype: a reward arrives at every step, it admits no conflicting interpretation, and the simulator computes it for free, which is why the environment yields to a modest DQN within minutes.

RL tends to fail under three recognisable conditions, each of which maps onto a failure mode in RLHF.

- **Sparse rewards.** The agent must stumble upon the correct behaviour by chance before any learning signal exists at all. Montezuma's Revenge ([Section 1.3.7](#137-results)) is the canonical example. The RLHF counterpart is structural rather than incidental: the reward model can only score a complete response, so a single terminal scalar must be apportioned across every token of the generation, which is exactly the credit-assignment burden that the critic and GAE ([Section 2.1.3](#gae)) are introduced to carry.
- **Misspecified rewards.** The agent optimises the proxy that was written down, not the intent behind it, and any gap between the two is eventually found and exploited. In RLHF the reward function is a learned proxy by construction, so the gap is guaranteed to exist. Exploiting it is called reward hacking, and containing it is the purpose of the KL penalty in Phase 2.
- **Highly stochastic environments.** When the same action from the same state yields wildly different returns, the learning signal is buried in return variance and sample requirements grow accordingly. In RLHF the stochasticity lives in the sampled generations and in noisy, sometimes contradictory human preference labels, one reason preference-data quality matters more than quantity ([Section 5.3](phase02.md#53-suggested-reading) in Phase 2).

### 1.3. Suggested Reading

Mnih et al. (2015), *Human-Level Control through Deep Reinforcement Learning.* Read before the project. The paper introduces the exact architecture and training procedure you are about to implement.

#### 1.3.1. Overview

Mnih et al. (2015) introduced DQN, an RL agent that learns to play Atari 2600 games from raw pixel input and game scores alone, with no hand-crafted features or game-specific knowledge. Evaluated on 49 games with a single fixed architecture and hyperparameter set, it surpassed the best existing RL methods on 43 games and reached a level comparable to a professional human games tester. The work represented a significant milestone in demonstrating that deep neural networks could serve as stable, general-purpose function approximators within an RL framework.

#### 1.3.2. Problem

Prior RL methods were ill-suited to high-dimensional sensory inputs such as raw video frames. Naïvely combining deep neural networks with RL leads to training instability, due to three structural problems identified in the paper: strong temporal correlations between consecutive training samples, a feedback loop in which small updates to $Q$ change the policy and therefore shift the distribution of the data the network trains on, and non-stationary learning targets, i.e. the Q-value targets shift as the network parameters are updated.

#### 1.3.3. Key Innovations

DQN addressed these instabilities with two complementary techniques.

- **Experience replay.** Agent transitions (state, action, reward, next state) are stored in a replay buffer and sampled uniformly at random during training. This breaks temporal correlations between samples, enables data reuse, and addresses the feedback-loop problem: because each mini-batch mixes transitions generated by many past policies, the training distribution is smoothed over time instead of being dictated by the current policy. A subtle consequence noted in the paper is that replay forces learning to be off-policy (the sampled transitions were generated by older parameters), which is what motivates building on Q-learning rather than an on-policy method.
- **Target network.** A separate copy of the Q-network, updated only every $C$ steps, is used to compute TD targets. This stabilises training by holding the targets fixed across multiple gradient updates: without it, an update that increases $Q(s_t, a_t)$ typically also increases the targets built from the very next states, creating the oscillation the delay is designed to break.

#### 1.3.4. Errata: Loss Function and Gradient

This section identifies and corrects two errors in the paper's mathematical presentation: an incorrect sign in the variance term of the loss function, and an unexplained omission in the gradient derivation.

The paper formulates the loss using the expected TD target, integrating out the stochastic next state $s'$,

$$L_i(\theta_i) = \mathbb{E}_{s,a,r}\left[\Bigl(\mathbb{E}_{s'}[y_i \mid s,a,r] - Q(s,a;\theta_i)\Bigr)^2\right],$$

where $y_i = r + \gamma \max_{a'} Q(s', a'; \theta_i^-)$ is the TD target computed with the frozen target network. Since $\mathbb{E}_{s'}[y_i]$ is intractable in practice, training minimises the surrogate loss,

$$\tilde{L}_i(\theta_i) = \mathbb{E}_{s,a,r,s'}\left[(y_i - Q(s,a;\theta_i))^2\right],$$

obtained by sampling $s'$ rather than integrating over it.

**Relationship between $L_i$ and $\tilde{L}_i$.** Let $\bar{y}_i = \mathbb{E}_{s'}[y_i \mid s,a,r]$. Expanding the inner expectation of $\tilde{L}_i$ by adding and subtracting $\bar{y}_i$ yields

$$\mathbb{E}_{s'}\left[(y_i - Q)^2\right] = \mathbb{E}_{s'}\left[(y_i - \bar{y}_i)^2\right] + 2(\bar{y}_i - Q)\underbrace{\mathbb{E}_{s'}[y_i - \bar{y}_i]}_{=\,0} + (\bar{y}_i - Q)^2.$$

The cross-term vanishes since $\mathbb{E}_{s'}[y_i] = \bar{y}_i$ by definition. Taking the outer expectation over $(s,a,r)$,

$$\tilde{L}_i(\theta_i) = \mathbb{E}_{s,a,r}\left[\operatorname{Var}_{s'}(y_i)\right] + L_i(\theta_i).$$

Rearranging gives the correct relationship,

$$L_i(\theta_i) = \tilde{L}_i(\theta_i) - \mathbb{E}_{s,a,r}\left[\operatorname{Var}_{s'}(y_i)\right].$$

The paper's Methods section instead writes $L_i = \tilde{L}_i + \mathbb{E}_{s,a,r}\left[\operatorname{Var}_{s'}(y_i)\right]$, stating the variance term with the opposite sign, which is incorrect. The negative sign is necessary: the surrogate $\tilde{L}_i$ includes extra noise from sampling $s'$, so the true loss must be strictly smaller than $\tilde{L}_i$ by the expected variance of the target. Since $\operatorname{Var}_{s'}(y_i)$ depends only on $\theta_i^-$, which is frozen, it does not vary with $\theta_i$. Both losses therefore share the same gradient,

$$\nabla_{\theta_i} L_i(\theta_i) = \nabla_{\theta_i} \tilde{L}_i(\theta_i),$$

which justifies optimising the tractable $\tilde{L}_i$ in practice.

**Gradient derivation.** Applying the chain rule to a single sample with $\delta_i = y_i - Q(s,a;\theta_i)$,

$$\nabla_{\theta_i} \delta_i^2 = 2\delta_i \cdot \nabla_{\theta_i} \delta_i = 2\delta_i \cdot \Bigl(\underbrace{\nabla_{\theta_i} y_i}_{=\,0} - \nabla_{\theta_i} Q(s,a;\theta_i)\Bigr).$$

Since $y_i$ depends on $\theta_i^-$ rather than $\theta_i$, the first term vanishes. Taking the expectation over $(s,a,r,s')$,

$$\nabla_{\theta_i} L_i(\theta_i) = -2 \mathbb{E}\left[\delta_i \cdot \nabla_{\theta_i} Q(s,a;\theta_i)\right].$$

The paper reports this result without the factor of $-2$. The scalar $2$ is absorbed into the learning rate $\alpha$, and the negative sign is absorbed into the gradient descent update $\theta \leftarrow \theta - \alpha\nabla_\theta L$, which already descends in the negative gradient direction. Neither omission changes the algorithm, but both are made without remark, making the paper's gradient appear inconsistent with a direct application of the chain rule.

#### 1.3.5. Error Clipping

The TD error $\delta_i = y_i - Q(s,a;\theta_i)$ can be very large early in training when Q-values are poorly initialised, producing large gradients that destabilise parameter updates. The paper addresses this by clipping $\delta_i$ to $[-1, 1]$, which is equivalent to using the Huber loss in place of mean squared error (MSE).

The mechanism is grounded in the derivative of the absolute value loss $|\delta_i|$, which equals $\pm 1$ everywhere except at zero. This constant derivative is precisely what bounds the gradient: for any $|\delta_i| > 1$, the gradient contribution is fixed at unit magnitude regardless of how large the error is. The Huber loss exploits this by combining both regimes: it behaves as MSE for $|\delta_i| \leq 1$, where the gradient is proportional to the error and allows precise convergence near the optimum, and as the absolute value loss for $|\delta_i| > 1$, where the constant-magnitude gradient prevents any single outlier transition from dominating the update.

Error clipping should not be confused with **reward clipping**, a separate mechanism the paper applies during training: all positive rewards are clipped to $+1$ and all negative rewards to $-1$, with zero rewards unchanged. Because game scores vary enormously in scale across the 49 games, it is reward clipping that bounds the scale of the error derivatives and allows the same learning rate to be used on every game without per-game tuning. The paper notes its cost explicitly: the agent cannot distinguish rewards of different magnitudes. Error clipping is then an additional stabilisation on top, protecting the update from outlier TD errors that survive even with clipped rewards.

#### 1.3.6. Architecture

The network receives the four most recent game frames as input, each preprocessed to an 84×84 luminance (grayscale) image and stacked into an 84×84×4 tensor. Stacking frames matters because a single screen is partially observed: it shows positions but not velocities, and the four-frame stack restores motion information. These pass through three convolutional layers that extract spatial features, followed by a fully connected layer of 512 rectifier units and a linear output layer with one Q-value per action.

The output design is itself a deliberate choice. Earlier work fed the (state, action) pair into the network and produced a single Q-value, requiring one forward pass per action. DQN instead takes only the state as input and emits all action values at once, so the $\max_{a'} Q(s', a')$ in every TD target costs a single forward pass. Crucially, the same architecture and hyperparameters were applied across all 49 games without modification, with the hyperparameters tuned informally on just five games and then frozen. [Algorithm 3](#alg-dqn) shows the training loop of DQN.

<figure id="alg-dqn" style="text-align: center;" markdown="1">
<div style="border: 1px solid #ccc; display: inline-block; text-align: left; padding: 1em; font-family: monospace;" markdown="1">

**Input:**<br>
$\mathrm{env}$: Atari game environment<br>
$N$: replay buffer capacity<br>
$k$: mini-batch size<br>
$C$: target network update frequency (steps)<br>
$\gamma \in (0,1]$: discount factor

**Output:**<br>
$\theta$: trained Q-network parameters

Initialise replay buffer $D$ with capacity $N$<br>
Initialise Q-network with parameters $\theta$<br>
Initialise target network with parameters $\theta^- \leftarrow \theta$<br>
**for each** episode:<br>
$\quad$ Observe initial state $s$<br>
$\quad$ **while** $s$ is not terminal:<br>
$\quad\quad$ Select action $a$ via $\varepsilon$-greedy policy under $Q(s, \cdot; \theta)$<br>
$\quad\quad$ Execute $a$, observe reward $r$ and next state $s'$<br>
$\quad\quad$ Store $(s, a, r, s')$ in $D$<br>
$\quad\quad$ Sample random mini-batch of $k$ transitions from $D$<br>
$\quad\quad$ $y \leftarrow r + \gamma \max_{a'} Q(s', a'; \theta^-)$<br>
$\quad\quad$ Update $\theta$ by minimising $(y - Q(s, a; \theta))^2$<br>
$\quad\quad$ Every $C$ steps: $\theta^- \leftarrow \theta$<br>
$\quad\quad$ $s \leftarrow s'$

**return** $\theta$

</div>
<figcaption>Algorithm 3: DQN training loop with experience replay and target network.</figcaption>
</figure>

#### 1.3.7. Results

DQN outperformed the best existing RL methods on 43 of the 49 games tested and reached at least 75% of the professional human tester's score (the paper's operationalisation of human-level performance) on 29 of them. Performance was strongest on games requiring visual pattern recognition (e.g. Breakout, Pong) and weakest on games demanding long-horizon planning (e.g. Montezuma's Revenge), where sparse rewards make credit assignment difficult. Ablation experiments confirmed that both innovations carry weight: disabling either experience replay or the target network substantially degraded scores, with replay removal being the more damaging of the two.

#### 1.3.8. Significance

The paper established that end-to-end learning from raw sensory data is feasible at scale using a single, fixed architecture. It laid the foundation for much of modern deep RL research, directly motivating subsequent advances including Double DQN, Duelling DQN, and Prioritised Experience Replay.

### 1.4. Hands-on Project: Deep Q-Network on CartPole {: #project-req-dqn-cartpole}

**Objective.** Implement DQN from scratch using PyTorch and solve the `CartPole-v1` environment from Gymnasium.

**Setup.**

```bash
cd /Volumes/ML_Workspace/projects/
mkdir rl-foundations && cd rl-foundations
uv init --python 3.12
uv add torch torchvision gymnasium numpy matplotlib rich
```

**Implementation outline.**

1. Define a two-hidden-layer multilayer perceptron (MLP) `QNetwork(obs_dim, n_actions)` that outputs per-action Q-values.
2. Implement a `ReplayBuffer` storing $(s, a, r, s', \text{done})$ tuples, supporting random mini-batch sampling.
3. Implement the DQN training loop:
   - Collect transitions by running the $\varepsilon$-greedy policy (anneal $\varepsilon$ from 1.0 to 0.05 over 10 000 steps).
   - Sample a mini-batch and compute the Bellman target using the target network.
   - Compute the MSE loss against the online network's Q-value for the taken action.
   - Update the online network with Adam and copy weights to the target network every 100 steps.
4. Log episodic return with `rich` and plot the learning curve.

**What to observe.** CartPole should solve (mean return > 475 over 100 episodes) within approximately 50,000 environment steps. If training is unstable, the first thing to check is whether the replay buffer is large enough (at least 10,000 transitions) and whether $\varepsilon$ is decaying too quickly.

---

## 2. Module 2: Policy Gradients and Proximal Policy Optimisation

### 2.1. Theory

Value-based methods learn Q-values and derive a policy implicitly. Policy gradient methods optimise the policy directly by computing the gradient of $J(\pi_\theta)$ with respect to $\theta$.

Three situations make the direct route necessary. First, extracting a policy from a Q-function requires $\arg\max_a Q(s, a)$, which is trivial over a handful of discrete actions but cannot be evaluated exactly over a continuous action space and is expensive over a very large discrete one. A policy network sidesteps the maximisation entirely by outputting a distribution over actions. Second, the greedy policy read off a Q-function is deterministic, whereas some problems require genuinely stochastic behaviour, and a policy network represents stochastic policies natively. Third, and most relevant to this course, a language model already *is* a parameterised policy: a distribution over a vocabulary of tens of thousands of tokens (the actions) conditioned on the sequence so far (the state). Optimising it directly through its log-probabilities is the natural fit, which is why the RLHF pipeline of Phase 2 is built on a policy gradient method.

#### 2.1.1. The Policy Gradient Theorem

The derivation starts from the definition of $J(\pi_\theta)$ given in [Section 1.1](#rl-problem) and proceeds in four steps: writing the objective as an integral over trajectories, applying the log-derivative trick, exploiting causality to replace the full return with the Q-function, and subtracting a state-dependent baseline to arrive at the advantage.

**Step 1: Trajectory integral.** Recall from [Section 1.1](#rl-problem) that

<span id="eq-J-integral"></span>

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t R(s_t, a_t)\right] = \int p(\tau;\theta) G(\tau) \mathrm{d}\tau, \tag{1}$$

where $G(\tau) = \sum_{t=0}^T \gamma^t R(s_t, a_t)$ is the discounted return of trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$. By the chain rule of probability, the trajectory density factors as

$$p(\tau;\theta) = p(s_0) \cdot p(a_0 \mid s_0) \cdot p(s_1 \mid s_0, a_0) \cdot p(a_1 \mid s_0, a_0, s_1) \cdots p(a_T \mid s_0, a_0, \ldots, s_T).$$

Each conditional factor admits one of two Markov simplifications. First, the policy is Markovian: the action at step $t$ depends only on the current state, so

$$p(a_t \mid s_0, a_0, \ldots, s_t; \theta) = \pi_\theta(a_t \mid s_t).$$

Second, the environment is Markovian: the next state depends only on the current state-action pair, so

$$p(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t) = P(s_{t+1} \mid s_t, a_t).$$

Applying both simplifications at every step and collecting terms,

<span id="eq-traj-prob"></span>

$$p(\tau;\theta) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \cdot \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t). \tag{2}$$

Differentiating [(1)](#eq-J-integral) with respect to $\theta$ and noting that $G(\tau)$ does not depend on $\theta$,

$$
\begin{align*}
\nabla_\theta J(\pi_\theta)
&= \nabla_\theta \int p(\tau;\theta) G(\tau) \mathrm{d}\tau \\
&= \int \nabla_\theta\left[p(\tau;\theta) G(\tau)\right] \mathrm{d}\tau \\
&= \int \nabla_\theta p(\tau;\theta) \cdot G(\tau) \mathrm{d}\tau,
\end{align*}
$$

where the second equality interchanges gradient and integral (justified by the Leibniz rule, since $p(\tau;\theta)$ is smooth in $\theta$ and $G(\tau)$ is bounded under bounded rewards and $\gamma < 1$), and the third uses $\nabla_\theta G(\tau) = 0$.

**Step 2: Log-derivative trick.** Direct evaluation of $\nabla_\theta p(\tau;\theta)$ is intractable because [(2)](#eq-traj-prob) contains the unknown environment dynamics $P$. The identity $\nabla_\theta p(\tau;\theta) = p(\tau;\theta) \nabla_\theta \log p(\tau;\theta)$, obtained by rearranging $\nabla_\theta \log p = (\nabla_\theta p) / p$, converts the integral into an expectation:

<span id="eq-log-deriv"></span>

$$
\begin{align*}
\nabla_\theta J(\pi_\theta)
&= \int \nabla_\theta p(\tau;\theta) \cdot G(\tau) \mathrm{d}\tau \\
&= \int p(\tau;\theta) \nabla_\theta \log p(\tau;\theta) \cdot G(\tau) \mathrm{d}\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log p(\tau;\theta) \cdot G(\tau)\right]. \tag{3}
\end{align*}
$$

Taking the log of [(2)](#eq-traj-prob) converts the product into a sum,

$$
\begin{align*}
\log p(\tau;\theta)
&= \log p(s_0) + \log \prod_{t=0}^T \pi_\theta(a_t \mid s_t) + \log \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t) \\
&= \log p(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t).
\end{align*}
$$

Differentiating with respect to $\theta$ and noting that $p(s_0)$ and $P$ do not depend on $\theta$,

<span id="eq-log-grad"></span>

$$
\begin{align*}
\nabla_\theta \log p(\tau;\theta)
&= \underbrace{\nabla_\theta \log p(s_0)}_{=0} + \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) + \underbrace{\sum_{t=0}^{T-1} \nabla_\theta \log P(s_{t+1} \mid s_t, a_t)}_{=0} \\
&= \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t). \tag{4}
\end{align*}
$$

This is the central cancellation that makes policy gradient methods model-free: the gradient of $J$ depends only on the policy log-likelihood, not on the dynamics $P$. Substituting [(4)](#eq-log-grad) into [(3)](#eq-log-deriv) gives the **trajectory-form policy gradient**:

<span id="eq-traj-pg"></span>

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\left(\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right) G(\tau)\right]. \tag{5}$$

**Step 3: Causality.** Expanding the product in [(5)](#eq-traj-pg) and writing out $G(\tau) = \sum_{t'=0}^T \gamma^{t'} R(s_{t'}, a_{t'})$,

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \sum_{t'=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \gamma^{t'} R(s_{t'}, a_{t'})\right].$$

Splitting the inner sum at $t' = t$,

<span id="eq-pg-split"></span>

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left(\sum_{t'=0}^{t-1} \gamma^{t'} R(s_{t'}, a_{t'}) + \sum_{t'=t}^T \gamma^{t'} R(s_{t'}, a_{t'})\right)\right]. \tag{6}$$

The claim is that the first inner sum (the past rewards) contributes zero in expectation. Intuitively, an action $a_t$ sampled at step $t$ cannot causally influence rewards already collected at earlier steps $t' < t$. Formally, the **score function** $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ has mean zero under $\pi_\theta(\cdot \mid s_t)$ (a standard identity, proved below), so any quantity determined before $a_t$ is sampled multiplies it to zero in expectation.

Fix any pair $(t, t')$ with $t' < t$. The reward $R(s_{t'}, a_{t'})$ is a deterministic function of $(s_{t'}, a_{t'})$, both of which lie in the prefix $\tau_{<t} := (s_0, a_0, \ldots, s_{t-1}, a_{t-1}, s_t)$. Neither factor in the integrand depends on the post-action variables $(s_{t+1}, a_{t+1}, \ldots, s_T, a_T)$, so those variables marginalise out trivially. Writing the joint density of $(\tau_{<t}, a_t)$ as $p(\tau_{<t};\theta) \cdot \pi_\theta(a_t \mid s_t)$,

<span id="eq-cross-term"></span>

$$
\begin{align*}
&\mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \gamma^{t'} R(s_{t'}, a_{t'})\right] \\
&\quad= \int p(\tau_{<t};\theta) \pi_\theta(a_t \mid s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \gamma^{t'} R(s_{t'}, a_{t'}) \mathrm{d}\tau_{<t} \mathrm{d}a_t \\
&\quad= \int p(\tau_{<t};\theta) \gamma^{t'} R(s_{t'}, a_{t'}) \left[\int \pi_\theta(a_t \mid s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \mathrm{d}a_t\right] \mathrm{d}\tau_{<t} \\
&\quad= \mathbb{E}_{\tau_{<t}}\left[\gamma^{t'} R(s_{t'}, a_{t'}) \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]\right]. \tag{7}
\end{align*}
$$

The inner expectation in [(7)](#eq-cross-term) is

<span id="eq-score-zero"></span>

$$
\begin{align*}
\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
&= \sum_a \pi_\theta(a \mid s_t) \cdot \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} \\
&= \sum_a \nabla_\theta \pi_\theta(a \mid s_t) \\
&= \nabla_\theta \sum_a \pi_\theta(a \mid s_t) \\
&= \nabla_\theta 1 = 0, \tag{8}
\end{align*}
$$

where the first equality applies the log-derivative identity $\nabla_\theta \log \pi_\theta(a \mid s_t) = \nabla_\theta \pi_\theta(a \mid s_t) / \pi_\theta(a \mid s_t)$, and the third exchanges the sum and the gradient (valid for finite action spaces by linearity, and for continuous action spaces under standard regularity conditions for differentiating under the integral). Substituting this result into [(7)](#eq-cross-term) gives zero. Every $t' < t$ term in [(6)](#eq-pg-split) therefore vanishes, leaving only the $t' \geq t$ contributions, which form the **reward-to-go** from step $t$:

<span id="eq-rtg"></span>

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^T \gamma^{t'} R(s_{t'}, a_{t'})\right]. \tag{9}$$

Factor $\gamma^t$ out of the inner sum,

$$\sum_{t'=t}^T \gamma^{t'} R(s_{t'}, a_{t'}) = \gamma^t \sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}).$$

Recall the definition of the action-value function from [Section 1.1](#rl-problem), stated as the expected discounted return when starting at $(s, a)$ at time zero and following $\pi$ thereafter,

$$Q^{\pi}(s, a) = \mathbb{E}\left[\sum_{k=0}^{T} \gamma^k R(s_k, a_k) \middle| s_0 = s, a_0 = a, \pi\right].$$

To apply this definition inside [(9)](#eq-rtg), where the reward-to-go is indexed from an arbitrary time step $t$ within a longer trajectory, two consequences of the Markov property are needed.

First, **prefix independence**. Under Markov dynamics $P(s_{t+1} \mid s_t, a_t)$ and a Markov policy $\pi_\theta(a_t \mid s_t)$, the conditional distribution of the future $(s_{t+1}, a_{t+1}, \ldots, s_T, a_T)$ given the full prefix $\tau_{<t} \cup \{a_t\}$ depends only on $(s_t, a_t)$. Therefore,

$$\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}) \middle| \tau_{<t}, a_t\right] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}) \middle| s_t, a_t\right].$$

The full-prefix conditioning collapses to conditioning on $(s_t, a_t)$ alone.

Second, **time-homogeneity**. Both $P$ and $\pi_\theta$ are stationary, with no explicit dependence on the time index. The conditional distribution of $(s_{t+1}, a_{t+1}, \ldots, s_T, a_T)$ given $(s_t = s, a_t = a)$ therefore matches the distribution of an independent roll-out $(s_1, a_1, \ldots, s_{T-t}, a_{T-t})$ given $(s_0 = s, a_0 = a)$. Re-indexing the inner sum by $k = t' - t$,

$$\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}) \middle| s_t = s, a_t = a\right] = \mathbb{E}\left[\sum_{k=0}^{T-t} \gamma^k R(s_k, a_k) \middle| s_0 = s, a_0 = a, \pi_\theta\right],$$

which coincides with $Q^{\pi_\theta}(s, a)$ in the infinite-horizon limit and is taken as the working definition in the finite-horizon case (the remaining-horizon distinction is suppressed throughout). Combining the two properties yields the identity used in the next step,

<span id="eq-Q-identity"></span>

$$Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}) \middle| s_t, a_t\right]. \tag{10}$$

Without prefix independence, the right-hand side of [(10)](#eq-Q-identity) would still depend on the entire trajectory history. Without time-homogeneity, it would still depend on the absolute time index $t$. Together, the two reduce the reward-to-go conditional expectation to a function of $(s_t, a_t)$ alone.

For each $t$, apply the law of total expectation to the corresponding term in [(9)](#eq-rtg) by conditioning on $(s_t, a_t)$, then substitute [(10)](#eq-Q-identity):

$$
\begin{align*}
&\mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \gamma^t \sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'})\right] \\
&\quad= \mathbb{E}_{(s_t, a_t)}\left[\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}) \middle| s_t, a_t\right]\right] \\
&\quad= \mathbb{E}_{(s_t, a_t)}\left[\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right],
\end{align*}
$$

where the outer expectation $\mathbb{E}_{(s_t, a_t)}$ is taken under the on-policy marginal distribution of $(s_t, a_t)$ induced by $\pi_\theta$. Summing over $t$,

<span id="eq-pg-per-step"></span>

$$\nabla_\theta J(\pi_\theta) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t)}\left[\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]. \tag{11}$$

Each per-step term in [(11)](#eq-pg-per-step) is an expectation under a different marginal. To re-fold them into a single trajectory expectation, note that the integrand at step $t$, namely

$$f_t(s_t, a_t) := \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t),$$

depends only on $(s_t, a_t)$, since $Q^{\pi_\theta}(s_t, a_t)$ has its future variables already integrated out. The trajectory expectation of $f_t$ therefore marginalises down to the $(s_t, a_t)$ marginal,

<span id="eq-marg-fold"></span>

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_\theta}[f_t(s_t, a_t)]
&= \int p(\tau;\theta) f_t(s_t, a_t) \mathrm{d}\tau \\
&= \int f_t(s_t, a_t) \underbrace{\left[\int p(\tau;\theta) \mathrm{d}\tau_{\setminus (s_t, a_t)}\right]}_{p_t(s_t, a_t)} \mathrm{d}s_t \mathrm{d}a_t \\
&= \mathbb{E}_{(s_t, a_t)}[f_t(s_t, a_t)], \tag{12}
\end{align*}
$$

where $p_t(s_t, a_t)$ is the on-policy marginal at step $t$. Substituting [(12)](#eq-marg-fold) into [(11)](#eq-pg-per-step) and pulling the sum inside by linearity,

$$
\begin{align*}
\nabla_\theta J(\pi_\theta)
&= \sum_{t=0}^T \mathbb{E}_{(s_t, a_t)}[f_t(s_t, a_t)] \\
&= \sum_{t=0}^T \mathbb{E}_{\tau \sim \pi_\theta}[f_t(s_t, a_t)] \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T f_t(s_t, a_t)\right],
\end{align*}
$$

which gives the **Q-form policy gradient**:

<span id="eq-pg-Q"></span>

$$\boxed{\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]} \tag{13}$$

**Step 4: Baseline subtraction.** The advantage form of the policy gradient theorem follows from the fact that any function of the state alone may be subtracted from the per-step coefficient $Q^{\pi_\theta}(s_t, a_t)$ in [(13)](#eq-pg-Q) without changing the gradient in expectation. The motivation is variance reduction: replacing $Q^{\pi_\theta}$ with a centred quantity reduces the variance of the sample-based estimator without biasing it. The key lemma is that, for any function $b: S \to \mathbb{R}$ and any $t$, the per-step expectation of the score function weighted by $b(s_t)$ vanishes:

<span id="eq-baseline-zero"></span>

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_\theta}\left[\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t)\right]
&= \mathbb{E}_{(s_t, a_t)}\left[\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t)\right] \\
&= \mathbb{E}_{s_t}\left[\gamma^t b(s_t) \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]\right] \\
&= \mathbb{E}_{s_t}\left[\gamma^t b(s_t) \cdot 0\right] = 0. \tag{14}
\end{align*}
$$

The first equality applies the marginalisation identity [(12)](#eq-marg-fold), since the integrand depends only on $(s_t, a_t)$. The second decomposes the joint marginal as $p_t(s_t, a_t) = p_t(s_t) \pi_\theta(a_t \mid s_t)$ and pulls $b(s_t)$ out of the inner expectation, where it is constant with respect to $a_t$. The third applies the score-function identity [(8)](#eq-score-zero) established in Step 3.

Since [(14)](#eq-baseline-zero) holds for every $t$, summing over $t$ also gives zero. Subtracting this zero sum from [(13)](#eq-pg-Q) and combining the two trajectory expectations by linearity,

$$
\begin{align*}
\nabla_\theta J(\pi_\theta)
&= \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right] \\
&\quad- \underbrace{\sum_{t=0}^T \mathbb{E}_{\tau \sim \pi_\theta}\left[\gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot b(s_t)\right]}_{=0} \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \big(Q^{\pi_\theta}(s_t, a_t) - b(s_t)\big)\right].
\end{align*}
$$

This identity holds for any baseline $b: S \to \mathbb{R}$. The mean of the estimator is unchanged by the choice of $b$, but the variance is not. To see why subtraction reduces variance, write the per-step estimator $X_t := \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (Q^{\pi_\theta}(s_t, a_t) - b(s_t))$ and decompose its variance as $\text{Var}[X_t] = \mathbb{E}[X_t^2] - (\mathbb{E}[X_t])^2$. The first moment $\mathbb{E}[X_t]$ does not depend on $b$ (by [(14)](#eq-baseline-zero) applied with the negative of any constant shift), so only the second moment responds to the choice of baseline:

$$\mathbb{E}[X_t^2] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\|\nabla_\theta \log \pi_\theta(a_t \mid s_t)\|^2 \cdot (Q^{\pi_\theta}(s_t, a_t) - b(s_t))^2\right].$$

The score-norm factor does not depend on $b$ either. Minimising the second moment therefore reduces to making $(Q - b)^2$ small on average. Concretely, if returns are large in magnitude, say $|Q^{\pi_\theta}| \sim 1000$, then a single-sample estimator with $b = 0$ multiplies the score function by a factor of size $\sim 10^6$ inside the second moment, producing wild swings between samples. Subtracting any reasonable approximation of $Q$ recentres the factor near zero. For example, $V^{\pi_\theta}(s_t) = \mathbb{E}_{a_t}[Q^{\pi_\theta}(s_t, a_t)]$ matches the state-dependent mean of $Q$, so $(Q - V)^2$ is bounded by the action-to-action spread of $Q$ rather than its absolute scale.

The strictly variance-minimising baseline under the current $\pi_\theta$ is the score-squared-weighted Q-average,

$$b^*(s_t) = \frac{\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\left[\|\nabla_\theta \log \pi_\theta(a_t \mid s_t)\|^2  Q^{\pi_\theta}(s_t, a_t)\right]}{\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)}\left[\|\nabla_\theta \log \pi_\theta(a_t \mid s_t)\|^2\right]},$$

which weights $Q$ across actions by the score norm. The unweighted average $V^{\pi_\theta}(s_t)$ coincides with $b^*$ only when the score norm is constant in $a_t$, which is generally false. In practice $V^{\pi_\theta}$ is preferred because it is far easier to estimate (only an expected return, not a score-weighted return), and the residual gap to $b^*$ is small relative to other sources of estimator noise.

Choosing $b(s_t) = V^{\pi_\theta}(s_t)$ replaces $Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$ with the advantage $A^{\pi_\theta}(s_t, a_t)$:

<span id="eq-pg-advantage"></span>

$$\boxed{\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right].} \tag{15}$$

Equation [(15)](#eq-pg-advantage) is the **policy gradient theorem** in its advantage form. The advantage acts as a directional signal under gradient ascent: the per-step update $\theta \leftarrow \theta + \eta  \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t)$ moves $\theta$ along the direction in which $\pi_\theta(a_t \mid s_t)$ increases, scaled by the sign and magnitude of $A^{\pi_\theta}$. When $A^{\pi_\theta}(s_t, a_t) > 0$ the probability $\pi_\theta(a_t \mid s_t)$ is raised, reinforcing the action. When $A^{\pi_\theta}(s_t, a_t) < 0$ the negative scalar flips the direction and the probability is lowered, suppressing the action. Actions that beat the policy's own expected return are pushed up, and those that underperform are pushed down.

In practice almost all implementations drop $\gamma^t$ from the per-step coefficient in [(15)](#eq-pg-advantage), weighting every timestep equally when aggregating the gradient across a trajectory. This is a biased estimator of $\nabla_\theta J$: later timesteps are overweighted relative to what the discounted objective requires, so the policy converges to the maximiser of a slightly different objective. The bias is empirically benign because $\gamma$ is typically close to 1, so $\gamma^t$ deviates little from 1 over a finite episode. In the episodic RLHF setting, where each episode is a single self-contained response with no meaningful continuation beyond it, $\gamma = 1$ is standard, $\gamma^t = 1$ for all $t$, and the question does not arise.

Equation [(15)](#eq-pg-advantage) admits a direct sample-based estimator: collect trajectories under $\pi_\theta$, estimate $A^{\pi_\theta}(s_t, a_t)$ at each step, and average $\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t$ over the batch. The log-probability gradient is identical to the gradient used in supervised cross-entropy training, which makes the implementation straightforward in any deep learning framework.

The **REINFORCE algorithm** is the simplest such estimator. It estimates $Q^{\pi_\theta}(s_t, a_t)$ by the empirical return $\sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'})$, which is unbiased since its expectation equals $Q^{\pi_\theta}(s_t, a_t)$ by definition of [(10)](#eq-Q-identity), but suffers from high variance because a single trajectory's return is a noisy proxy for the expected return. REINFORCE may be used with no baseline or with a simple running mean, i.e. an incremental unweighted average of total episode returns updated as $\bar{R}_n = \bar{R}_{n-1} + \frac{1}{n}(R_n - \bar{R}_{n-1})$ after each episode $n$. The running mean is state-independent and provides modest variance reduction at negligible cost, but because it ignores which state the agent is in, it falls well short of a state-dependent baseline such as $V^{\pi_\theta}(s_t)$. **Actor-critic** methods close this gap by learning a separate value network $V_\phi(s) \approx V^{\pi_\theta}(s)$ as a state-dependent baseline, giving the advantage estimate $\hat{A}_t = \hat{Q}_t - V_\phi(s_t)$. The name reflects the two roles: the policy $\pi_\theta$ is the *actor* (it selects actions), and $V_\phi$ is the *critic* (it evaluates the current state and provides the baseline signal that centres the actor's gradient update). The variance reduction from a well-fit critic is the primary reason actor-critic methods dominate REINFORCE in modern implementations, including the PPO-based RLHF pipeline covered in [Section 2.1.2](#ppo-clip).

#### 2.1.2. Proximal Policy Optimisation {: #ppo-clip}

Vanilla policy gradient updates are sensitive to the step size. A single large update can shift the policy far from the distribution under which the rollouts were collected, making the gradient estimate meaningless and, in the worst case, collapsing the policy to a degenerate distribution (e.g. always outputting the same token). Once collapsed, the policy produces no useful gradient signal and cannot recover. **Trust region** methods address this by constraining how much the policy is allowed to change in a single update. Trust Region Policy Optimisation (TRPO) enforces this constraint exactly by solving a constrained optimisation problem at every step, which is expensive. Proximal Policy Optimisation (PPO) is a first-order approximation that achieves a similar effect through a simple clipping heuristic, without any constrained solver. The same heuristic delivers a second benefit that matters just as much in practice: it lets a single batch of expensive rollouts be reused for several gradient epochs, which is the property developed below.

**From policy gradient to a local approximation:**

The policy gradient in [(15)](#eq-pg-advantage) is an expectation over trajectories drawn from the *current* policy $\pi_\theta$, which makes it strictly on-policy: each batch of rollouts is valid for exactly one gradient step. After a single update $\theta$ has moved, the batch is now sampled from the old policy rather than the new one, and must be discarded in favour of fresh rollouts. This is wasteful, because collecting rollouts is the dominant cost of training. In the RLHF setting each rollout is a full generation from a large language model, so extracting only one update per batch is prohibitively expensive.

The goal is therefore to reuse one batch, collected under a fixed behaviour policy $\pi_{\theta_\text{old}}$, for several gradient epochs. The obstacle is that the batch is off-policy with respect to the updated $\pi_\theta$, so the on-policy gradient of [(15)](#eq-pg-advantage) no longer applies. Importance sampling corrects for this distribution mismatch by reweighting each sample with the probability ratio

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)},
$$

which equals $1$ when $\theta = \theta_\text{old}$. The importance-sampling objective is

$$
\mathcal{L}^{\text{IS}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}}\left[\sum_{t=0}^T \gamma^t r_t(\theta) \hat{A}_t \right],
$$

where the expectation is over trajectories $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ sampled from $\pi_{\theta_\text{old}}$, and $\hat{A}_t$ is an empirical estimate of $A^{\pi_{\theta_\text{old}}}(s_t, a_t)$, typically computed with Generalised Advantage Estimation (GAE) (see [Section 2.1.3](#gae)). In implementations, the outer expectation and inner sum are flattened: each $(s_t, a_t)$ from the rollout buffer is treated as a single sample, and $\mathbb{E}_t[\cdot]$ in the literature is shorthand for the empirical average over this flattened collection.

**Gradient equivalence at $\theta = \theta_\text{old}$:**

To see that $\mathcal{L}^{\text{IS}}$ is a valid local approximation of $J$ at $\theta_\text{old}$, differentiate it and evaluate at $\theta = \theta_\text{old}$. Using $\nabla_\theta r_t(\theta) = r_t(\theta) \nabla_\theta \log \pi_\theta(a_t \mid s_t)$,

$$
\nabla_\theta \mathcal{L}^{\text{IS}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}}\left[\sum_{t=0}^T \gamma^t r_t(\theta) \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right].
$$

Setting $\theta = \theta_\text{old}$, every $r_t(\theta_\text{old}) = 1$ and $\hat{A}_t$ estimates $A^{\pi_{\theta_\text{old}}}(s_t, a_t)$, so

$$
\nabla_\theta \mathcal{L}^{\text{IS}}(\theta)\Big|_{\theta = \theta_\text{old}} = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}}\left[\sum_{t=0}^T \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big|_{\theta_\text{old}} \cdot A^{\pi_{\theta_\text{old}}}(s_t, a_t) \right].
$$

The right-hand side is exactly $\nabla_\theta J(\pi_\theta)\big|_{\theta = \theta_\text{old}}$ from the policy gradient theorem in [(15)](#eq-pg-advantage): the sampling distribution $\pi_{\theta_\text{old}}$ in the expectation now matches the policy at which the gradient is evaluated, and the advantage is taken under the same policy. The gradients therefore agree at $\theta_\text{old}$, but $\mathcal{L}^{\text{IS}}$ is only a local approximation: as $\theta$ moves away from $\theta_\text{old}$, the distribution mismatch between $\pi_\theta$ and $\pi_{\theta_\text{old}}$ grows, and $\mathcal{L}^{\text{IS}}$ becomes an increasingly unreliable estimate of $J$.

**Clipped surrogate objective:**

Since $\mathcal{L}^{\text{IS}}$ is only a local approximation, it is only reliable while $\theta$ stays close to $\theta_\text{old}$: when $r_t$ deviates far from $1$, the importance-weighted estimate has high variance and may drive the policy into a degenerate region. PPO addresses this by clipping the ratio:

<span id="eq-ppo-clip"></span>

$$
\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right],} \tag{16}
$$

where $\varepsilon$ is a small constant (typically 0.1 or 0.2) and $\mathbb{E}_t[\cdot]$ denotes the empirical average over sampled timesteps from $\pi_{\theta_\text{old}}$-rollouts, as introduced above. Note that [(16)](#eq-ppo-clip) drops the per-step factor $\gamma^t$ that appears in $\mathcal{L}^{\text{IS}}$ and in the policy gradient theorem [(15)](#eq-pg-advantage): the flat average $\mathbb{E}_t[\cdot]$ weights every timestep equally. This is the standard implementation convention discussed in [Section 2.1.1](#211-the-policy-gradient-theorem), and the distinction between the two equations is exactly that one. Equation [(15)](#eq-pg-advantage) is theoretically grounded, retaining $\gamma^t$ so that the gradient of $\mathcal{L}^{\text{IS}}$ matches it exactly at $\theta_\text{old}$, whereas [(16)](#eq-ppo-clip) is the practical objective actually optimised. The resulting bias is benign because $\gamma$ is close to 1, and it vanishes entirely in the episodic RLHF setting where $\gamma = 1$.

**How the clip works:**

Consider the two cases separately.

- **Positive advantage** ($\hat{A}_t > 0$, the action was better than the baseline). The objective rewards increasing $r_t$. The clip caps the unclipped term at $r_t = 1 + \varepsilon$, so once the new policy has raised the probability of $a_t$ by a factor of $1 + \varepsilon$ relative to $\pi_{\theta_\text{old}}$, no further gradient is provided. Beyond that point, the clipped term becomes constant in $\theta$ and its gradient vanishes.
- **Negative advantage** ($\hat{A}_t < 0$, the action was worse than the baseline). The objective rewards decreasing $r_t$. The clip caps the unclipped term at $r_t = 1 - \varepsilon$, so once the probability of $a_t$ has been reduced by a factor of $1 - \varepsilon$, no further gradient is provided in the same direction.

The outer $\min$ is what makes the heuristic sound. If the optimiser is moving in the right direction (improving the objective), the clip removes the gradient once the trust region is crossed. However, if a step has already pushed $r_t$ outside the trust region in the wrong direction (e.g. $r_t > 1 + \varepsilon$ when $\hat{A}_t < 0$, so the policy is now assigning more probability to a bad action), $\min$ selects the unclipped term, which still has a non-zero gradient pulling $r_t$ back. The clip therefore only suppresses gradients when doing so is safe, and never lets an out-of-region policy go uncorrected.

**Full PPO objective:**

In practice the policy update is combined with a value-function regression loss and an entropy bonus:

<span id="eq-ppo-full"></span>

$$
\boxed{\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 H[\pi_\theta],} \tag{17}
$$

where $\mathcal{L}^{\text{VF}}$ is the squared error between the critic's value estimate and the empirical return, $H[\pi_\theta]$ is the policy entropy (added to encourage exploration), and $c_1, c_2$ are scalar coefficients.

$\mathcal{L}(\theta)$ is a surrogate *objective*, not a loss in the minimisation sense: all three terms are oriented so that larger values correspond to better policies. Training therefore solves

$$
\max_{\theta}\ \mathcal{L}(\theta),
$$

via stochastic gradient *ascent* on $\nabla_\theta \mathcal{L}(\theta)$ (equivalently, descent on $-\mathcal{L}(\theta)$).

**Why PPO matters for RLHF:**

PPO is the workhorse optimiser of the classical RLHF pipeline (see [Section 5](phase02.md#ppo-rlhf-loop)), for two reasons.

- The clip provides cheap, stable on-policy updates from minibatch data, which matters when each rollout (a full generation from a large language model) is expensive.
- In RLHF, PPO's trust-region behaviour is reinforced by an explicit Kullback–Leibler (KL) divergence penalty against the supervised fine-tuned reference policy, added to the per-token reward. This second KL term is conceptually distinct from the clip (it constrains drift from the *reference* policy rather than from the *rollout* policy), but both share the same goal: preventing the policy from straying into regions where the reward model is unreliable, which is the proximate cause of reward hacking.

Understanding the clip is therefore a prerequisite for diagnosing two failure modes in RLHF: instability from overly aggressive policy updates, and reward hacking from drift away from the reference distribution.

#### 2.1.3. Generalised Advantage Estimation {: #gae}

In practice, the advantage is not computed from full Monte Carlo returns (too high variance) or from a single one-step TD error (too high bias) but from a weighted combination of one-step TD errors via **Generalised Advantage Estimation (GAE)**:

<span id="eq-gae"></span>

$$
\boxed{\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l},} \tag{18}
$$

where $R_t = R(s_t, a_t)$ is the observed reward and $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$ is a one-step TD residual. The hyperparameter $\lambda \in [0, 1]$ interpolates between high-bias/low-variance ($\lambda = 0$, pure TD) and low-bias/high-variance ($\lambda = 1$, pure Monte Carlo). Values of $\lambda = 0.95$ and $\gamma = 0.99$ are robust defaults.

**Intuition: GAE as a geometric mixture of $n$-step advantage estimates.** To see why [(18)](#eq-gae) is a weighted combination of one-step TD errors, define the $n$-step advantage estimate as the sum of the first $n$ one-step TD residuals:

$$
\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \underbrace{R_t + \gamma R_{t+1} + \cdots + \gamma^{n-1} R_{t+n-1}}_{n \text{ actual rewards}} + \underbrace{\gamma^n V(s_{t+n}) - V(s_t)}_{\text{bootstrap from step } n}.
$$

Using $n$ real rewards before bootstrapping reduces bias (less reliance on the potentially inaccurate $V$) at the cost of higher variance (more stochastic reward terms). GAE mixes all such estimates with geometrically decaying weights (this form holds for $\lambda \in [0, 1)$, while the $\lambda = 1$ endpoint is read directly from equation [(18)](#eq-gae)):

$$
\hat{A}_t^{\text{GAE}} = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}.
$$

To see this collapses to [(18)](#eq-gae), swap the order of summation: $\delta_{t+l}$ contributes to every $\hat{A}_t^{(n)}$ with $n > l$, so its total weight is $(1-\lambda)\sum_{n=l+1}^{\infty}\lambda^{n-1}\cdot\gamma^l = \lambda^l \gamma^l = (\gamma\lambda)^l$, recovering $\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$. At $\lambda = 1$ the $(1-\lambda)$ formula yields $0 \cdot \infty$ (the sum diverges), so the extreme case must be read from [(18)](#eq-gae) directly, as done below. The same identity, expressed as a TD($\lambda$) return rather than an advantage, is derived in full in [Appendix 1.5](appendix.md#15-n-step-td-error).

**$\lambda$ at the extremes.**

- $\lambda = 0$: only the $l=0$ term in [(18)](#eq-gae) survives, giving $\hat{A}_t = \delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$. This is the pure one-step TD estimate. It bootstraps immediately from $V$, so it carries $V$'s bias but accumulates only one reward's worth of variance.

- $\lambda = 1$: all terms survive with weights $\gamma^l$. Substituting the definition $\delta_{t+l} = R_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ and separating the reward terms from the value terms,

$$
\begin{align*}
\hat{A}_t^{\text{GAE}(\gamma,1)} &= \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \\
&= \sum_{l=0}^{\infty} \gamma^l \big(R_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\big) \\
&= \sum_{l=0}^{\infty} \gamma^l R_{t+l} + \sum_{l=0}^{\infty} \big(\gamma^{l+1} V(s_{t+l+1}) - \gamma^l V(s_{t+l})\big) \\
&= \sum_{l=0}^{\infty} \gamma^l R_{t+l} - V(s_t) \\
&= G_t - V(s_t).
\end{align*}
$$

The second sum telescopes: the $\gamma^{l+1} V(s_{t+l+1})$ term at index $l$ cancels the $-\gamma^{l'} V(s_{t+l'})$ term at index $l' = l+1$, so every value term cancels except the boundary term $-\gamma^0 V(s_t) = -V(s_t)$ (using $\gamma^l V(s_{t+l}) \to 0$ as $l \to \infty$). Here $G_t$ is the **(discounted) Monte Carlo return-to-go** from step $t$,

<span id="eq-return-def"></span>

$$
G_t := \sum_{l=0}^{\infty} \gamma^l R_{t+l}. \tag{19}
$$

The resulting estimate $G_t - V(s_t)$ is the **Monte Carlo advantage**, unbiased (no $V$ appears in the lookahead, only as the baseline $V(s_t)$) but with high variance because $G_t$ sums infinitely many stochastic reward terms.

**The return $G_t$ and its relationship to $Q$ and $V$.** Equation [(10)](#eq-Q-identity) defined $Q^{\pi_\theta}$ as an *expected* return. The $G_t$ above is a single *realised* return collected along one trajectory, so the two value functions of [Section 1.1](#rl-problem) are recovered by conditioning that one sample on different information,

<span id="eq-G-QV"></span>

$$
Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}[G_t \mid s_t, a_t], \qquad V^{\pi_\theta}(s_t) = \mathbb{E}[G_t \mid s_t]. \tag{20}
$$

Fixing the action $a_t$ gives $Q$, while averaging over the policy's action gives $V$, consistent with the identity $V^{\pi_\theta}(s_t) = \mathbb{E}_{a_t \sim \pi_\theta}[Q^{\pi_\theta}(s_t, a_t)]$ from [Section 1.1](#rl-problem). This is why the $\lambda = 1$ estimate $G_t - V(s_t)$ is unbiased for the advantage: by [(20)](#eq-G-QV) the return $G_t$ is a one-sample draw of $Q^{\pi_\theta}(s_t, a_t)$, and $V(s_t)$ is the baseline, so their difference samples $A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$.

Intermediate $\lambda$ blends both: the weight on the $n$-step estimate decays as $\lambda^{n-1}$, so short-horizon estimates (low variance, higher bias) dominate while long-horizon ones are exponentially discounted.

**Implementation note: computing GAE by a backward recursion.** Equation [(18)](#eq-gae) is written as an infinite forward sum, but it is never evaluated that way. It is computed by a single backward pass, the form used in [Algorithm 4](#alg-ppo). Pull the $l = 0$ term out of [(18)](#eq-gae) and re-index the remainder with $m = l - 1$:

$$
\begin{align*}
\hat{A}_t^{\text{GAE}(\gamma,\lambda)}
&= \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} \\
&= \delta_t + \sum_{l=1}^{\infty} (\gamma\lambda)^l \delta_{t+l} \\
&= \delta_t + \gamma\lambda \sum_{m=0}^{\infty} (\gamma\lambda)^{m} \delta_{(t+1)+m} \\
&= \delta_t + \gamma\lambda \hat{A}_{t+1}^{\text{GAE}(\gamma,\lambda)},
\end{align*}
$$

where the inner sum is exactly [(18)](#eq-gae) evaluated at step $t+1$. This gives the recurrence $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$. In a finite rollout of length $T$ the sum truncates at $l = T - t$; seeding the recursion with $\hat{A}_{T+1} = 0$ and sweeping $t = T, T-1, \ldots, 0$ reproduces the truncated sum exactly,

$$
\hat{A}_T = \delta_T, \qquad \hat{A}_{T-1} = \delta_{T-1} + \gamma\lambda \delta_T, \qquad \ldots, \qquad \hat{A}_t = \sum_{l=0}^{T-t}(\gamma\lambda)^l \delta_{t+l}.
$$

Each step prepends one TD residual and rescales the accumulated tail by $\gamma\lambda$, computing every $\hat{A}_t$ in one $O(T)$ pass rather than the $O(T^2)$ cost of summing each timestep independently. The boundary value $\hat{A}_{T+1} = 0$ encodes the episode edge: there is no advantage to accumulate beyond the final step, with $V(s_{T+1}) = 0$ for a true terminal state or the bootstrap value $V(s_{T+1})$ if the episode was merely truncated at the horizon.

**Advantage standardisation.** In practice, $\hat{A}_t$ is standardised to zero mean and unit variance across the rollout batch before being used in the policy gradient update. The two operations have different justifications. Subtracting the batch mean is theoretically neutral: any state-independent constant may be subtracted from the advantage without changing the expected gradient, by the baseline argument in [Section 2.1.1](#eq-pg-advantage). Dividing by the batch standard deviation has no formal justification but prevents large reward magnitudes from producing disproportionately large gradient steps, acting as a per-batch adaptive scaling of the learning rate. This standardisation is not part of the formal GAE or PPO theory, but it is a near-universal implementation default.

**A note on TD error targets.** In [Section 1.1.1](#q-learning-dqn) the TD error was defined on $Q$, namely $\delta_t = R_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$, whereas the residual here is defined on $V$, namely $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$. This is not a conflict: TD is a general bootstrapping technique that applies to any value function, because each of $Q$ and $V$ obeys its own Bellman recursion in the immediate reward $R_t$. The choice of target depends on the algorithm. $Q$-based TD suits off-policy control (Q-learning, DQN), where $\max_{a'} Q$ selects the greedy action without a separate value network, while $V$-based TD suits actor-critic methods (e.g. Advantage Actor-Critic (A2C), PPO), where a separate policy handles action selection and the critic need only estimate state values. The advantage $A$ is never bootstrapped directly; it is recovered from $V$-based TD residuals through GAE, as the $\lambda = 0$ case $\hat{A}_t = \delta_t$ makes explicit. [Appendix 1.2](appendix.md#12-why-value-functions-satisfy-a-td-recursion) derives why $V$ obeys the same one-step recursion as $Q$ and shows how a $V$-based TD residual doubles as a one-sample advantage estimate.

#### 2.1.4. Putting It Together: The PPO Algorithm {: #ppo-algorithm}

The three subsections above are the three ingredients of one training loop. The **policy gradient theorem** ([Section 2.1.1](#eq-pg-advantage)) establishes the gradient [(15)](#eq-pg-advantage): push up the log-probability of actions with positive advantage, push down those with negative advantage. As shown in [(17)](#eq-ppo-full), **PPO** ([Section 2.1.2](#ppo-clip)) replaces the raw gradient with the clipped surrogate $\mathcal{L}^{\text{CLIP}}$ (as defined in [(16)](#eq-ppo-clip)) so that each update stays inside a trust region, which is what permits reusing one batch of rollouts for several gradient epochs. **GAE** ([Section 2.1.3](#gae)) supplies the advantage estimate $\hat{A}_t$ that both of the above require, trading off bias and variance through $\lambda$ (see [(18)](#eq-gae)). The actor-critic structure underpins all three: a value network $V_\phi$ acts as the baseline that makes the advantage well-conditioned (the variance-reduction argument from Step 4 of [Section 2.1.1](#eq-pg-advantage)) and supplies the $V(s)$ terms inside the TD residuals.

<figure id="alg-ppo" style="text-align: center;" markdown="1">
<div style="border: 1px solid #ccc; display: inline-block; text-align: left; padding: 1em; font-family: monospace;" markdown="1">

**Input:**<br>
$\theta$: initial policy network parameters<br>
$\phi$: initial value network parameters<br>
$\varepsilon$: clip ratio<br>
$K$: number of optimisation epochs per rollout batch<br>
$M$: minibatch size<br>
$\gamma \in (0,1]$: discount factor<br>
$\lambda \in [0,1]$: GAE parameter<br>
$c_1, c_2$: value loss and entropy bonus coefficients<br>
$\eta$: learning rate

**Output:**<br>
$\theta$: trained policy parameters

**repeat until** converged:<br>
$\quad$ *// 1. Collect rollouts with the current (frozen) policy*<br>
$\quad$ $\theta_\text{old} \leftarrow \theta$<br>
$\quad$ Run $\pi_{\theta_\text{old}}$ in the environment to collect a batch of transitions $\{(s_t, a_t, R_t)\}$<br>
$\quad$ Record old log-probs: $\log \pi_{\theta_\text{old}}(a_t \mid s_t)$<br>
$\quad$ *// 2. Estimate advantages and value targets via GAE ([Section 2.1.3](#gae))*<br>
$\quad$ **for each** $t$:<br>
$\quad\quad$ $\delta_t \leftarrow R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ *// TD residual*<br>
$\quad$ $\hat{A}_{T+1} \leftarrow 0$<br>
$\quad$ **for** $t = T, T-1, \ldots, 0$:<br>
$\quad\quad$ $\hat{A}_t \leftarrow \delta_t + \gamma\lambda \hat{A}_{t+1}$ *// backward GAE recurrence*<br>
$\quad\quad$ $\hat{G}_t \leftarrow \hat{A}_t + V_\phi(s_t)$ *// value target (return estimate), explained below*<br>
$\quad$ Standardise $\hat{A}_t$ to zero mean and unit variance across the batch *// implementation trick, see [Section 2.1.3](#gae)*<br>
$\quad$ *// 3. Optimise for $K$ epochs over the same batch (clip keeps off-policy reuse inside the trust region)*<br>
$\quad$ **for** epoch $= 1, \ldots, K$:<br>
$\quad\quad$ **for each** minibatch of size $M$:<br>
$\quad\quad\quad$ $r_t(\theta) \leftarrow \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ *// probability ratio, [Section 2.1.2](#ppo-clip)*<br>
$\quad\quad\quad$ $\mathcal{L}^\text{CLIP} \leftarrow \mathrm{mean}\left(\min\left(r_t(\theta)\hat{A}_t, \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right)$<br>
$\quad\quad\quad$ $\mathcal{L}^\text{VF} \leftarrow \mathrm{mean}\left((V_\phi(s_t) - \hat{G}_t)^2\right)$<br>
$\quad\quad\quad$ $H \leftarrow \mathrm{mean}\left(\mathrm{entropy}[\pi_\theta(\cdot \mid s_t)]\right)$<br>
$\quad\quad\quad$ $\mathcal{L}(\theta,\phi) \leftarrow \mathcal{L}^\text{CLIP} - c_1 \mathcal{L}^\text{VF} + c_2 H$<br>
$\quad\quad\quad$ $(\theta, \phi) \leftarrow (\theta, \phi) + \eta \nabla_{\theta,\phi} \mathcal{L}(\theta,\phi)$ *// gradient ascent*

**return** $\theta$

</div>
<figcaption>Algorithm 4: PPO training loop with clipped surrogate objective, GAE advantage estimation, and actor-critic architecture.</figcaption>
</figure>

**The value target $\hat{G}_t$.** The critic $V_\phi$ is trained by regression, so it needs a target, an estimate of $V^{\pi_\theta}(s_t)$. The advantage already produced by GAE supplies one at no extra cost. Rearranging the advantage definition as $Q = A + V$ and reading it through [(20)](#eq-G-QV), 'advantage plus baseline' recovers a return estimate,

$$
\hat{G}_t := \hat{A}_t + V_\phi(s_t).
$$

The hat denotes an estimate of the like-named quantity, so $\hat{G}_t$ estimates the return $G_t$ of [(19)](#eq-return-def), not the immediate reward $R_t$ (the symbol $R$ is reserved for the immediate reward throughout, which is why the value target is denoted $\hat{G}$ rather than $\hat{R}$). Conditioned on the sampled action its mean is $Q^{\pi_\theta}(s_t, a_t)$, and averaged over the policy's actions across the batch it is $V^{\pi_\theta}(s_t)$ by [(20)](#eq-G-QV), which is exactly what least-squares regression drives $V_\phi(s_t)$ towards. Unrolling the backward recursion of [Section 2.1.3](#gae) shows $\hat{G}_t$ is the TD($\lambda$) return: at $\lambda = 0$ it reduces to the one-step target $R_t + \gamma V_\phi(s_{t+1})$, and at $\lambda = 1$ to the full Monte Carlo return $G_t$. The value-target line of [Algorithm 4](#alg-ppo) forms $\hat{G}_t$ from the unstandardised advantage and regresses $V_\phi$ towards it through the value loss $\mathcal{L}^{\text{VF}}$.

The combined objective $\mathcal{L} = \mathcal{L}^{\text{CLIP}} - c_1 \mathcal{L}^{\text{VF}} + c_2 H$ ascends the clipped surrogate, descends the squared value error (the $-c_1$ sign turns the maximisation into a minimisation of the critic's fit), and adds the entropy bonus to keep the policy exploratory. After the $K$ epochs the batch is discarded, $\theta_\text{old}$ is refreshed to the updated $\theta$, and the loop repeats with fresh on-policy rollouts. This is the exact actor-critic skeleton reused in the RLHF pipeline, where the environment reward $R_t$ is supplied by a reward model and a KL-to-reference penalty plays the role that the clip plays here.

### 2.2. Intuitions for Debugging Proximal Policy Optimisation

The following diagnostic checks cover the majority of PPO failures encountered in practice.

- **Clip fraction too high.** If more than 20–30% of the ratio $r_t$ is being clipped in a given update batch, the policy is trying to change too fast. Reduce the learning rate or reduce the number of PPO epochs per rollout.
- **Value function loss not decreasing.** The critic is failing to fit. This is often due to poor normalisation of the returns or advantages. Always standardise advantages to zero mean and unit variance within a mini-batch.
- **Entropy collapsing to zero.** The policy has become deterministic. This usually means the reward signal is too strong relative to the entropy bonus. Increase $c_2$ or reduce the reward scale.
- **KL divergence spiking.** In the RLHF context, a spike in the KL between the policy and the reference model is the primary signal of reward hacking or training instability. It warrants immediate inspection of the reward model's scores on the current policy's outputs.
- **Reward increases but true performance degrades.** This is reward hacking. The policy has found a distributional artefact in the reward model.

### 2.3. Suggested Reading

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms.* Read before the project. It is the primary reference for [Section 2.1.2](#ppo-clip).
- Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* Read alongside [Section 2.1.3](#gae) for the full technical derivation of GAE.

#### 2.3.1. Schulman et al. (2017): *Proximal Policy Optimization Algorithms*

**Overview**

Schulman et al. (2017) introduced PPO, a family of policy gradient methods designed to be simpler and more sample-efficient than TRPO while retaining its stability. The paper proposed two variants: a KL-penalty version, which adds an adaptive penalty on the KL divergence between successive policies, and the clipped surrogate objective (CLIP) described in [Section 2.1.2](#ppo-clip). The CLIP variant became the standard and is the algorithm referred to as 'PPO' throughout this course.

**Problem**

TRPO achieves stable policy updates but requires solving a constrained optimisation problem at each step, making it computationally expensive and difficult to combine with architectures that share parameters between the policy and value function. Vanilla policy gradient methods are cheaper but sensitive to step size: a single large gradient step can destroy a well-performing policy, and no reliable mechanism exists to detect this until training has already diverged.

**Key Contributions**

The central contribution is the CLIP objective (see [(16)](#eq-ppo-clip)), which constrains the probability ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ to $[1-\varepsilon, 1+\varepsilon]$. By taking the minimum of the clipped and unclipped terms, the objective removes the incentive to move $r_t$ outside the trust region in the direction favoured by the advantage (in either sign), and continues to provide a corrective gradient when the ratio has already moved outside the region in the wrong direction. A secondary contribution is the combined objective in [(17)](#eq-ppo-full), which folds the value function regression loss and an entropy bonus into a single scalar that can be maximised with a standard first-order optimiser over multiple epochs on the same rollout batch, avoiding the expense of a conjugate gradient solver.

**Results**

PPO was evaluated on MuJoCo continuous-control tasks and 49 Atari 2600 games. On continuous control, the CLIP objective with $\varepsilon = 0.2$ scored best among the tested surrogate variants, outperforming the adaptive KL-penalty version of PPO as well as TRPO, the cross-entropy method (CEM), and A2C baselines. On Atari, PPO beat A2C on the large majority of games and was competitive with Actor-Critic with Experience Replay (ACER): ACER reached higher final scores on more games, while PPO led on early learning speed despite being far simpler to implement. Between three and ten optimisation epochs per rollout batch were reported as robust defaults across tasks.

**Significance**

PPO replaced TRPO as the default policy optimiser for large-scale RL applications, including the classical RLHF pipeline (see [Section 5](phase02.md#ppo-rlhf-loop)). Its prevalence is due to three properties: it requires only first-order gradients, it is compatible with shared actor-critic architectures, and its clipping heuristic requires little per-task tuning in practice.

#### 2.3.2. Schulman et al. (2016): *High-Dimensional Continuous Control Using Generalized Advantage Estimation*

**Overview**

Schulman et al. (2016) introduced GAE, a family of advantage estimators parameterised by $\lambda \in [0, 1]$ that interpolates between high-bias/low-variance and low-bias/high-variance estimates. The paper combined GAE with TRPO and demonstrated high-dimensional continuous locomotion on MuJoCo. The technical derivation of GAE is covered in [Section 2.1.3](#gae).

**Problem**

Advantage estimation in policy gradient methods faces a fundamental bias-variance tradeoff. Monte Carlo returns are unbiased but have high variance because they sum many stochastic reward terms. One-step TD estimates are low-variance but biased, since they bootstrap from the value function $V$. Prior work provided no principled way to combine the two.

**Key Contributions**

The central contribution is the GAE estimator (see [(18)](#eq-gae)),

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l},$$

which is equivalent to an exponentially-weighted mixture of $n$-step advantage estimates with weights decaying as $\lambda^{n-1}$. At $\lambda = 0$, only the one-step TD residual $\delta_t$ contributes, giving the pure TD advantage. At $\lambda = 1$, all terms contribute with weights $\gamma^l$, the sum telescopes, and the estimator reduces to the Monte Carlo advantage $G_t - V(s_t)$. Intermediate values of $\lambda$ blend both extremes, with shorter-horizon estimates dominating due to exponential discounting. The paper's second contribution is a trust-region procedure for fitting the value function itself, constraining how far the value estimates can move on each batch. Since GAE's TD residuals bootstrap from $V$, stabilising the value fit stabilises every advantage estimate built on it.

**Results**

GAE combined with TRPO was evaluated on cart-pole and on challenging 3D locomotion tasks: learning running gaits for bipedal and quadrupedal simulated robots, and getting a 3D biped to stand up from the ground, with neural network policies mapping raw kinematics directly to joint torques. Intermediate values of $\lambda$ (approximately 0.92 to 0.98) consistently outperformed both $\lambda = 0$ (pure TD) and $\lambda = 1$ (Monte Carlo), confirming the benefit of the intermediate bias-variance tradeoff. The locomotion results were among the first of this dimensionality learnt directly with model-free policy gradient methods, without hand-crafted policy representations.

**Significance**

GAE is the standard advantage estimator in virtually every modern actor-critic implementation, including PPO (see [Section 2.1.3](#gae) and [Algorithm 4](#alg-ppo)). Its significance lies not only in the estimator itself but in framing advantage estimation as a bias-variance tradeoff controlled by a single interpretable parameter $\lambda$, giving practitioners a principled knob to tune rather than an ad hoc design choice.

### 2.4. Hands-on Project: Proximal Policy Optimisation on LunarLander

**Objective.** Implement PPO from scratch using PyTorch and solve `LunarLanderContinuous-v3`. This environment requires a Gaussian policy head and is a direct warm-up for the actor-critic structure used in RLHF.

**Setup.**

```bash
cd /Volumes/ML_Workspace/projects/rl-foundations
uv add "gymnasium[box2d]" tensorboard
```

**Implementation outline.**

1. Define an `ActorCritic` network with a shared trunk (two-layer MLP), a Gaussian policy head (outputs mean $\mu$ and log-std $\log \sigma$ for each action dimension), and a scalar value head.
2. Implement the rollout buffer: collect $N = 2048$ steps of interaction and compute GAE advantages and returns.
3. Implement the PPO update: iterate $K = 10$ epochs over the buffer in mini-batches of size 64, computing $\mathcal{L}^{\text{CLIP}}$, $\mathcal{L}^{\text{VF}}$, and entropy, and clipping gradient norms to 0.5.
4. Log the clip fraction, entropy, value loss, and episodic return to TensorBoard under `./results/lunarlander_ppo/tb/<config_label>/`, then inspect with `tensorboard --logdir ./results/lunarlander_ppo/tb`.

**What to observe.** LunarLander should reach a mean return of approximately 200 within 1,000,000 steps on CPU (fast on M4 Pro, expected to complete in under 30 minutes). Plot the clip fraction and entropy alongside reward. If the clip fraction spikes above 0.4 early in training, the learning rate is too high.
