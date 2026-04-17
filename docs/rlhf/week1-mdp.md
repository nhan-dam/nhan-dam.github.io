# 1. Week 1 – MDPs and Bellman Equations

## 1.1. The Markov Decision Process

A Markov decision process (MDP) is a mathematical framework for sequential decision-making under uncertainty. It is defined by a tuple \((S, A, P, R, \gamma)\):

- \(S\) — the state space.
- \(A\) — the action space.
- \(P(s' \mid s, a)\) — the transition probability of reaching state \(s'\) after taking action \(a\) in state \(s\).
- \(R(s, a)\) — the reward function.
- \(\gamma \in [0, 1)\) — the discount factor, which controls how much future rewards are weighted relative to immediate ones.

The 'Markov' property means the next state depends only on the current state and action, not on the history.

---

## 1.2. Value Functions

### 1.2.1. State-Value Function

The state-value function \(V^\pi(s)\) measures the expected return starting from state \(s\) and following policy \(\pi\):

\[
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \;\Big|\; s_0 = s \right]
\]

### 1.2.2. Action-Value Function

The action-value function \(Q^\pi(s, a)\) additionally conditions on the first action taken:

\[
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \;\Big|\; s_0 = s,\, a_0 = a \right]
\]

The two are related by \(V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)\).

---

## 1.3. Bellman Equations

The Bellman equations express a recursive consistency condition that any valid value function must satisfy.

### 1.3.1. Bellman Expectation Equation

\[
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]
\]

This says: the value of a state is the expected immediate reward plus the discounted value of the next state, averaged over both the policy and the environment dynamics.

### 1.3.2. Bellman Optimality Equation

The optimal value function \(V^*(s)\) satisfies:

\[
V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a) + \gamma V^*(s') \right]
\]

The optimal policy can be recovered greedily: \(\pi^*(s) = \arg\max_a Q^*(s, a)\).

---

## 1.4. Key Takeaways

- An MDP is the standard model for RL; understanding it precisely is a prerequisite for everything else.
- Value functions quantify 'how good' a state (or state-action pair) is under a given policy.
- The Bellman equations are the foundation of almost all RL algorithms — both dynamic programming methods (value iteration, policy iteration) and modern deep RL.
- The discount factor \(\gamma\) is not just a mathematical convenience; it encodes a preference for sooner rewards and ensures convergence in infinite-horizon problems.

---

## 1.5. References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, Ch. 3.
