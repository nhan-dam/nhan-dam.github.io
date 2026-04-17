# 1. RLHF — Conceptual Overview

---

# 2. The Core Problem

Standard supervised fine-tuning maps input to a single correct output. But for open-ended tasks like summarisation, **there is no single correct answer** — only human preferences. RLHF reframes the problem: instead of 'what is the right answer?', ask 'what do humans prefer?'

---

# 3. The Three-Stage Pipeline

## 3.1. Stage 1 — Build a Preference Dataset

1. Use the base LLM to generate **multiple completions** for the same prompt.
2. Show human labellers **two completions side-by-side** and ask which they prefer (pairwise comparison is preferable to absolute rating scales, which are too subjective and vary across people).
3. The result is a dataset of `(prompt, chosen_completion, rejected_completion)` triplets.

> ⚠️ Key nuance: this dataset captures the preferences of *your specific labellers*, not human preference in general. Defining your **alignment criteria** upfront (e.g. more helpful, less toxic, more concise) is critical.

---

## 3.2. Stage 2 — Train a Reward Model

The reward model is itself an LLM, repurposed as a **regression model** — it takes a `(prompt, completion)` pair as input and outputs a single scalar score (higher = better aligned with human preferences).

### 3.2.1. The Loss Function

The reward model is trained using a **pairwise ranking loss** derived from the Bradley-Terry model for pairwise comparisons. Given a triplet `(prompt x, chosen response y_w, rejected response y_l)`, the loss is:

```
L = -log( σ( r(x, y_w) - r(x, y_l) ) )
```

where `r(x, y)` is the scalar score the reward model outputs for a given prompt-completion pair, and `σ` is the sigmoid function.

### 3.2.2. Why Not Maximise the Raw Score Difference?

A simpler alternative would be `L = -(r(y_w) - r(y_l))` — directly maximising the gap between chosen and rejected scores. This fails for two reasons.

First, the loss is **unbounded below**. The optimiser can always make the loss more negative by widening the gap further, with no natural stopping point. There is no sense of 'confident enough' — gradients never diminish as the model becomes more certain, which makes training unstable.

Second, the gradients are **constant**. The gradient of the raw difference with respect to the output scores is always ±1, regardless of how confidently right or wrong the model is. A model that barely separates a pair (`r(y_w) = 0.51, r(y_l) = 0.50`) receives exactly the same gradient update as one that separates them decisively (`r(y_w) = 100, r(y_l) = -100`). The network cannot tell from the gradient signal alone whether it should be more or less confident.

### 3.2.3. What the Log-Sigmoid Does

The sigmoid squashes the score difference into a probability between 0 and 1 — specifically, the model's estimated probability that `y_w` is preferred over `y_l`. Taking the log then gives the loss two important properties.

The loss is now **bounded**: as the gap grows large and positive, `σ` approaches 1 and `log(σ)` approaches 0. Training naturally saturates once the model is confident, and gradients shrink towards zero — a stable, well-behaved objective.

The gradients are now **adaptive**: the gradient of `L` with respect to `r(y_w)` works out to `-(1 - σ(r(y_w) - r(y_l)))`. When the model is already confident, the sigmoid is close to 1, so the gradient is nearly zero — a small nudge. When the model is wrong or uncertain, the sigmoid is near 0 or 0.5, the gradient is large, and the network receives a strong corrective signal. Training effort is automatically concentrated where it is most needed.

> 💡 The Bradley-Terry framing is a natural fit for preference data: the reward model learns a relative 'strength' for each completion — meaningful only in comparison to other completions, never on an absolute scale. This mirrors exactly how human labellers produce the data, always expressing relative rather than absolute judgements.

---

## 3.3. Stage 3 — Fine-tune with RL (PPO)

The RL framing maps naturally onto the LLM setting:

| RL Concept | RLHF Equivalent |
|---|---|
| Agent | The LLM being tuned. |
| Policy | LLM weights (maps state → action). |
| State | Current context (prompt + tokens generated so far). |
| Action | Generating the next token. |
| Reward | Score from the reward model. |

The training loop proceeds as follows:

1. Sample a prompt from a **prompt-only dataset**.
2. The LLM generates a completion.
3. The reward model scores the `(prompt, completion)` pair.
4. Update LLM weights via **PPO** (Proximal Policy Optimisation).
5. Repeat — the policy gradually produces more preferred outputs.

> ⚠️ A **KL-divergence penalty** is added in practice to prevent the tuned model from drifting too far from the base model (an important detail glossed over in the lecture).

---

# 4. Efficient Training: PEFT

Full fine-tuning updates all model weights, which is expensive for large models. **Parameter-Efficient Fine-Tuning (PEFT)** instead trains only a small subset of (or entirely new) parameters. The key benefits are:

- Much faster training.
- Simpler serving: one base model with swappable adapter weights per use case.

The course uses a PEFT approach when tuning LLaMA 2.

---

# 5. Summary

> Collect human preference comparisons → train a reward model to score outputs → use PPO to optimise the LLM towards high-reward completions, efficiently via PEFT.
