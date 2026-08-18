# Appendix: Temporal-Difference Error and Bootstrapping

> Created on: 28 May 2026
>
> Updated on: 28 May 2026

## 1. Temporal-Difference Error {: #appendix-td-error}

### 1.1. General Definition

The 'temporal-difference' (TD) error is the signed difference between a bootstrapped one-step estimate of return and the current value estimate. It serves as the learning signal in all TD methods.

The Bellman equation states that value satisfies a recursive relationship: the return from the current step equals the immediate reward plus the discounted return from the next step. Applying this recursion whilst substituting the current value estimate for the unknown future return gives the TD error,

$$
\delta_t = \underbrace{R_t + \gamma \hat{V}(\text{next})}_{\text{TD target}} - \underbrace{\hat{V}(\text{now})}_{\text{current estimate}},
$$

where $\hat{V}$ denotes the value estimate being learned and $\gamma \in [0, 1]$ is the discount factor. A non-zero $\delta_t$ indicates that the estimate is inconsistent with itself after one step of real experience. The standard update rule adjusts the estimate by $\alpha \delta_t$, where $\alpha$ is the learning rate.

### 1.2. Why Value Functions Satisfy a TD Recursion

[Appendix 1.1](#11-general-definition) states that value 'satisfies a recursive relationship' but does not derive it. This subsection derives the recursion for the state-value function $V$ and shows why the same one-step reward $R_t$ that defines the action-value function $Q$ also governs $V$. The point is easy to take for granted because $Q$'s dependence on $R_t$ is the more obvious of the two.

**The recursion for $V$.** The return $G_t = \sum_{l=0}^{\infty} \gamma^l R_{t+l}$ (defined in [Section 2.1.3](phase01.md#gae), [(19)](phase01.md#eq-return-def)) splits into the immediate reward and the discounted return from $t+1$,

$$
G_t = R_t + \gamma G_{t+1}.
$$

Condition on $S_t = s$ and take expectations, using the definition $V^\pi(s) = \mathbb{E}[G_t \mid S_t = s]$,

$$
V^\pi(s) = \mathbb{E}[R_t \mid S_t = s] + \gamma \mathbb{E}[G_{t+1} \mid S_t = s].
$$

The second term is resolved by the **law of total expectation** (the 'tower rule'), which states that $\mathbb{E}[X \mid S_t] = \mathbb{E}\big[\mathbb{E}[X \mid S_t, S_{t+1}] \mid S_t\big]$ for any $X$, i.e. averaging the finer conditioning on $S_{t+1}$ back over $S_{t+1}$ recovers the coarser expectation. Applying it with $X = G_{t+1}$,

$$
\mathbb{E}[G_{t+1} \mid S_t = s] = \mathbb{E}_{S_{t+1}}\big[ \mathbb{E}[G_{t+1} \mid S_{t+1}, S_t = s] \mid S_t = s \big] = \mathbb{E}_{S_{t+1}}\big[ V^\pi(S_{t+1}) \mid S_t = s \big].
$$

The inner conditional expectation collapses to $V^\pi(S_{t+1})$ for two reasons: by the Markov property the future depends on the past only through $S_{t+1}$, so the extra conditioning on $S_t$ drops out, and $\mathbb{E}[G_{t+1} \mid S_{t+1}]$ is by definition the value function one step later. Substituting back,

$$
V^\pi(s) = \mathbb{E}_{S_{t+1}}\big[ R_t + \gamma V^\pi(S_{t+1}) \mid S_t = s \big].
$$

This is the **Bellman equation for $V$**. Its structure is identical to the Bellman equation for $Q$,

$$
Q^\pi(s, a) = \mathbb{E}\big[ R_t + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \big].
$$

In both cases value equals expected immediate reward plus discounted next value. The only difference is the conditioning: $Q$ fixes the action $A_t = a$ and bootstraps from the next state-action value, whereas $V$ averages over $A_t \sim \pi$ and bootstraps from the next state value. Both relate value to the single reward $R_t$ through the same unrolling of the return, so the dependence of $V$ on $R_t$ is no less direct than that of $Q$. Replacing the unknown $V^\pi$ with the current estimate $V$ and dropping the expectation in favour of one sampled transition yields the TD target $R_t + \gamma V(S_{t+1})$ and the TD error $\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$ of [Appendix 1.1](#11-general-definition).

**The dual reading of $R_t + \gamma V(S_{t+1})$.** The same expression carries two interpretations depending on what is held fixed, and this is what links the $V$-based TD error to the advantage function $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ used in policy gradient methods (see [Section 1.1](phase01.md#rl-problem)).

- Averaged over the action the policy would take, $R_t + \gamma V(S_{t+1})$ is the one-step TD target for $V(S_t)$, i.e. it bootstraps the *state* value.
- Holding the action actually taken fixed, $R_t + \gamma V(S_{t+1})$ is a single-sample estimate of $Q^\pi(S_t, A_t)$, since $Q^\pi(s, a) = \mathbb{E}[R_t + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a]$ is the same Bellman relation conditioned on the action.

Reading it the second way and subtracting the baseline $V(S_t)$ gives

$$
\delta_t = \underbrace{R_t + \gamma V(S_{t+1})}_{\text{one-sample estimate of } Q^\pi(S_t, A_t)} - \underbrace{V(S_t)}_{\text{baseline}} \approx Q^\pi(S_t, A_t) - V^\pi(S_t) = A^\pi(S_t, A_t).
$$

The TD residual on $V$ is therefore a one-sample estimate of the advantage. This is precisely the $\lambda = 0$ case of Generalised Advantage Estimation (see [Section 2.1.3](phase01.md#gae)), and it explains how GAE constructs an advantage estimate entirely from $V$-based TD residuals without ever forming a separate $Q$ network.

### 1.3. Instantiations

The general form specialises based on which value function is being learned.

**State-value function $V$.**

$$
\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t).
$$

**Action-value function $Q$.** Three variants arise depending on how the next-step estimate is formed.

| Variant | TD target | Next-step estimate |
|---|---|---|
| SARSA | $R_t + \gamma Q(S_{t+1}, A_{t+1})$ | Action actually taken (on-policy). |
| Q-learning | $R_t + \gamma \max_a Q(S_{t+1}, a)$ | Greedy action (off-policy). |
| Expected SARSA | $R_t + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a)$ | Policy-weighted average (on-policy). |

In every variant the TD error has the same shape: one step of real reward followed by a bootstrapped estimate of the remainder, minus the current estimate.

### 1.4. Semi-gradient Note

When $\hat{V}$ is parameterised (e.g. a neural network), TD methods treat the bootstrap target as a constant during the gradient step, i.e. they do not differentiate through $\hat{V}(\text{next})$. This is the 'semi-gradient' property, which distinguishes TD updates from true gradient descent on a fixed target such as the observed Monte Carlo return.

### 1.5. $n$-step TD Error

The 1-step TD error generalises to $n$ steps by accumulating real rewards over $n$ transitions before bootstrapping. The '$n$-step return' from time $t$ is

$$
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k} + \gamma^n \hat{V}(S_{t+n}),
$$

where the summation covers $n$ observed rewards and $\gamma^n \hat{V}(S_{t+n})$ is the bootstrapped tail. The '$n$-step TD error' is then

$$
\delta_t^{(n)} = G_t^{(n)} - \hat{V}(S_t).
$$

Setting $n = 1$ recovers the standard TD error from [Appendix 1.1](#11-general-definition). As $n \to \infty$, the bootstrap term vanishes (for $\gamma \lt 1$) and $G_t^{(n)}$ converges to the full Monte Carlo return $G_t$.

The parameter $n$ therefore acts as a dial between the two extremes described in [Appendix 2.2](#22-trade-off-with-monte-carlo): larger $n$ reduces bias (more real reward signal, less dependence on the imperfect $\hat{V}$) at the cost of higher variance (more stochastic steps in the target). '$\text{TD}(\lambda)$' extends this idea by replacing the single choice of $n$ with an exponentially weighted average over all $n$-step returns,

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_t^{(n)} + \lambda^{T-t-1} G_t,
$$

where $T$ is the episode termination step and $G_t$ is the full Monte Carlo return from $t$. Setting $\lambda = 0$ zeroes out all but the $n=1$ term, recovering 1-step TD. Setting $\lambda = 1$ zeroes out the entire sum via the $(1-\lambda)$ prefactor, leaving $\lambda^{T-t-1} G_t = G_t$, i.e. the full Monte Carlo return.

Subtracting $V(S_t)$ from both sides yields an equivalent form: a weighted sum of 1-step TD errors,

$$
G_t^\lambda - V(S_t) = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l},
$$

where $\delta_{t+l} = R_{t+l} + \gamma V(S_{t+l+1}) - V(S_{t+l})$. The derivation proceeds in two steps.

**Step 1: telescoping identity.** For any $n \geq 1$, substituting the definition of $\delta$ into $\sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$ gives

$$
\sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \sum_{l=0}^{n-1} \gamma^l R_{t+l} + \sum_{l=0}^{n-1} \gamma^{l+1} V(S_{t+l+1}) - \sum_{l=0}^{n-1} \gamma^l V(S_{t+l}).
$$

Reindexing the middle sum as $\sum_{l=1}^{n} \gamma^l V(S_{t+l})$, the last two sums telescope, leaving only boundary terms:

$$
\sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \underbrace{\sum_{l=0}^{n-1} \gamma^l R_{t+l} + \gamma^n V(S_{t+n})}_{G_t^{(n)}} - V(S_t) = G_t^{(n)} - V(S_t).
$$

**Step 2: swap summation order.** Since $(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} = 1$, subtracting $V(S_t)$ from $G_t^\lambda$ gives

$$
G_t^\lambda - V(S_t) = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\bigl[G_t^{(n)} - V(S_t)\bigr].
$$

Substituting the identity from Step 1 and swapping the order of summation (the term $\gamma^l \delta_{t+l}$ appears in the inner sum for every $n \gt l$):

$$
\begin{align*}
G_t^\lambda - V(S_t) &= (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{l=0}^{n-1}\gamma^l\delta_{t+l} \\
&= (1-\lambda)\sum_{l=0}^{\infty}\gamma^l\delta_{t+l}\sum_{n=l+1}^{\infty}\lambda^{n-1} \\
&= (1-\lambda)\sum_{l=0}^{\infty}\gamma^l\delta_{t+l} \cdot \frac{\lambda^l}{1-\lambda} \\
&= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},
\end{align*}
$$

where the geometric series $\sum_{n=l+1}^{\infty}\lambda^{n-1} = \lambda^l/(1-\lambda)$ was used. This is the form used in Generalised Advantage Estimation (GAE): the advantage estimate $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$ is the TD($\lambda$) error in advantage form.

---

## 2. Bootstrapping

### 2.1. Definition in RL

'Bootstrapping' means using the agent's own current value estimate as a substitute for the true future return, rather than waiting to observe it directly. The name is a metaphor: the agent builds on an estimate it is simultaneously trying to improve.

Concretely, instead of observing the full discounted return

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots,
$$

the agent approximates it as

$$
G_t \approx R_t + \gamma \hat{V}(S_{t+1}),
$$

replacing the unobserved tail with a single stored value. This approximation is the source of the TD target defined in [Appendix 1.1](#11-general-definition).

### 2.2. Trade-off with Monte Carlo

The alternative to bootstrapping is the Monte Carlo (MC) approach: wait until the episode terminates, then use the actual observed return $G_t$ as the learning target. The two approaches have complementary properties.

| Property | TD (bootstrapping) | Monte Carlo |
|---|---|---|
| Bias | Biased (errors in $\hat{V}$ propagate). | Unbiased. |
| Variance | Lower (one random step of noise). | Higher (full trajectory of noise). |
| Online learning | Yes (updates mid-episode). | No (requires episode completion). |

The defining distinction between the two families of RL algorithms is that TD methods bootstrap, while MC methods do not.

### 2.3. Disambiguation: Statistical Bootstrap

The term 'bootstrap' has a distinct meaning in statistical machine learning. In statistics, 'bootstrapping' refers to the resampling technique introduced by Efron (1979): given a dataset of $n$ observations, draw $n$ samples with replacement repeatedly to approximate the sampling distribution of a statistic. This is unrelated to RL bootstrapping beyond the shared metaphor of making do with available data rather than acquiring more.
