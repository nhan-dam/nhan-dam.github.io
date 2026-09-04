# Reinforcement Learning from Human Feedback: A Course for Machine Learning Practitioners

> Created on: 22 April 2026
>
> Updated on: 4 September 2026

This course is written for a machine learning (ML) practitioner with a solid background in supervised learning, neural network training, and standard deep learning tooling, but with no prior exposure to reinforcement learning (RL). This page is the way in. It explains why the course has the modules it has and in that order, by following one chain of reasoning from the original problem to the methods that address it, and it records what was built and run along the way.

## 1. Scope and Audience

The goal is threefold: to build sufficient theoretical understanding to reason about why RL and reinforcement learning from human feedback (RLHF) behave as they do, to develop the debugging intuitions needed to diagnose training failures, and to produce working, non-trivial implementations on a single workstation.

The material is divided into three phases plus a cross-cutting reference. Phase 1 covers the RL foundations that underpin RLHF. Phase 2 covers the classical three-step RLHF pipeline. Phase 3 covers the modern alternatives, both the offline preference methods that remove the reinforcement learning entirely and the critic-free methods that keep it. A debugging appendix collects failure signals and remedies across all three. Most modules carry a theory section, a set of design intuitions aimed at debugging and training stability, and a hands-on project with concrete implementation guidance for local hardware.

The discussion on this page is deliberately high level. Every claim is developed properly on the phase page that owns it, and every implementation has its own project report. [Section 5](#5-the-experiments) records what was actually built and run, separately from the argument, so that the chain stands on theory alone.

---

## 2. The Problem

A pre-trained language model predicts likely text. It does not produce *useful* text, and prompted with a question it is as willing to continue with another question as to answer one, because both are plausible continuations of web text.

The behaviour wanted instead, i.e. following an instruction, being helpful, declining to be harmful, cannot be written down as a differentiable loss over tokens. There is no target string to compare against, and for most prompts many different responses are acceptable while many others are not. Supervised learning needs a label. Preference is not a label.

Everything that follows is an attempt to close that gap, and each attempt exposes the difficulty that motivates the next.

---

## 3. The Chain

### 3.1. Module 1: The Reinforcement Learning Problem

**The problem.** If the target cannot be labelled but the outcome can be scored, the task is no longer supervised learning. It becomes sequential decision-making under a scalar signal, and that needs a formalism before it needs an algorithm.

**The method.** Model the task as a Markov decision process, i.e. states, actions, transitions, and a reward, with behaviour described by a policy and the goal expressed as the expected discounted return. Within that frame, value-based methods learn an action-value function scoring each action in each state, then read a policy off it by acting greedily.

**What it achieves.** A precise statement of what it means to learn from a score rather than a label, and a working method for problems whose actions are few and discrete.

**What it leaves open.** Acting greedily means comparing every available action, which is only possible when they can be enumerated. A language model chooses from the whole vocabulary at every step, and it has to stay stochastic rather than always emitting the single highest-scoring token. Deriving a policy from a value function is the wrong shape for this task.

### 3.2. Module 2: Policy Gradients and Proximal Policy Optimisation

**The problem.** Optimise the policy directly, without enumerating actions and without giving up stochasticity.

**The method.** The policy gradient theorem gives the gradient of the return with respect to the policy's own parameters, so the policy is trained rather than derived. Two corrections make that practical. The raw estimator has ruinous variance, so a baseline is subtracted to leave an advantage, which is estimated by Generalised Advantage Estimation (GAE). A single large step can destroy a working policy, because the data was collected under the previous one, so the update is confined to a trust region, which in its practical form is Proximal Policy Optimisation (PPO) and its clipped surrogate.

**What it achieves.** A stable optimiser that needs nothing but states, actions, and a scalar reward. It never needs to know what the *right* action was, which is exactly the property the original problem demands.

**What it leaves open.** That interface has to be supplied by something. Nothing so far says what a state, an action, or an episode is when the task is generating text, and nothing supplies the reward.

**Why this module rests on the last.** The advantage that Module 2's estimator consumes is defined in terms of Module 1's value function, and the return it differentiates is Module 1's objective. Module 2 is not an alternative to Module 1 but a second route to the same target, taken because the first route does not survive a vocabulary-sized action space.

### 3.3. Module 3: Supervised Fine-Tuning

**The problem.** Before preference can be optimised, the model has to be capable of instruction-following at all. A base model is too far from the target behaviour for a reward signal to find anything worth reinforcing.

**The method.** Fine-tune on curated (prompt, response) demonstrations with ordinary cross-entropy, masked so that only response tokens contribute. Low-Rank Adaptation (LoRA) makes this affordable on one workstation.

**What it achieves.** An instruction-following model. It also becomes the reference policy $\pi_{\text{ref}}$ that every later stage anchors to, and the initialisation for every later model.

**What it leaves open.** Two things, both structural rather than incidental. Imitation cannot exceed the quality of what it imitates, so the ceiling is the demonstration set. More fundamentally, the loss treats one reference response as correct and everything else as wrong, so it has no way to express that one acceptable answer is *better* than another acceptable answer. That comparison is exactly what the original problem was about.

### 3.4. Module 4: Reward Model Training

**The problem.** A signal is needed that ranks two acceptable responses, and it must come from human judgement without requiring a human in the loop at training time.

**The method.** Collect pairwise comparisons rather than absolute scores, because people rank far more consistently than they rate. Fit the Bradley–Terry model, which assumes the probability of preferring one response over another depends only on the difference of two latent scores, and identify that latent score with a network's output. The result is a scalar reward model $r_\phi$, built on the fine-tuned backbone so that it already understands the responses it is judging.

**What it achieves.** A differentiable, queryable stand-in for human preference. Preference has been converted into a number.

**What it leaves open.** Two limits, and the second is the one that shapes the rest of the course. First, a reward model judges but cannot generate, so on its own it improves nothing. Second, it is a proxy fitted to finite, noisy comparisons, so it agrees with true preference only over the region it was trained on. Optimising it hard therefore stops being a way of optimising preference and starts being a way of exploiting the fit, which is reward hacking. This is a property of using any learned proxy as an objective, not a defect of a particular model.

### 3.5. Module 5: The Full Reinforcement Learning from Human Feedback Training Loop

**The problem.** A judge exists and an optimiser exists. They have to be connected, and the connection has to be built so that the proxy is not exploited.

**The method.** Cast generation as the decision process Module 1 described, i.e. the state is the prompt plus the tokens so far, an action is the next token, and an episode is one response. The fine-tuned model becomes the policy, the reward model supplies a score at the end of the episode, and PPO does the updating. Because the reward model is trustworthy only near the distribution it was trained on, a Kullback–Leibler (KL) penalty against $\pi_{\text{ref}}$ is added, making distance from the reference a cost that a high score has to outweigh.

**What it achieves.** The first method in the chain that actually improves a policy against a preference signal. It also completes the pipeline, since each of the three stages consumes what the previous one produced.

**What it leaves open.** Three costs, all inherent to the design. It needs four models in memory at once, i.e. the policy, the frozen reference, the reward model, and a learned critic. It has two nested mechanisms restraining the policy at different anchors and timescales, i.e. PPO's clip against the rollout policy within a batch and the KL against the reference across the run, and both need tuning. Above all, the proxy problem of [Section 3.4](#34-module-4-reward-model-training) is now load-bearing, because the entire loop is an apparatus for maximising a fitted approximation.

> **Two levels, easily conflated.** The KL-shaped reward defines *what* is maximised. PPO's clipped surrogate defines *how* each gradient step maximises it. They are not competing objectives, and Module 5 states the relationship explicitly.

### 3.6. Module 6: Direct Preference Optimisation

**The problem.** The loop is expensive and delicate, and it reaches preference through two approximations stacked on each other, i.e. fit a reward model to comparisons, then optimise a policy against that fit.

**The method.** Observe that the KL-penalised objective Module 5 optimises numerically has an exact closed-form maximiser. Invert it, and any policy paired with the reference implicitly defines a reward function, namely the scaled log-ratio of their probabilities. Substitute that into the Bradley–Terry model of Module 4, where only a *difference* of rewards appears, so the intractable normalising term cancels because both responses share a prompt. What remains is an ordinary binary classification loss on preference pairs.

**What it achieves.** The reward model, the rollouts, and the critic all disappear. Two models remain and training is supervised, which makes it dramatically cheaper and far less sensitive to hyperparameters. The KL anchor has not been abandoned but absorbed, since the reference now sits inside the loss as a regulariser.

**What it leaves open.** It is off-policy. Training runs over a fixed dataset and the model never sees its own samples, so it can only align to whatever distribution the preference data represents, and it cannot improve by exploring. It also admits a failure mode with no analogue in Module 5, i.e. only the difference of implicit rewards appears in the loss, so the loss can be reduced by making both responses less likely provided the rejected one falls faster.

### 3.7. Module 7: Group Relative Policy Optimisation

**The problem.** Module 6 escaped PPO's cost by giving up on-policy learning. The natural question is whether the cost can be removed without that concession.

**The method.** Return to the loop and remove the critic instead. The critic exists only to provide a baseline that reduces gradient variance, and any state-dependent quantity serves. Sampling several responses to the *same* prompt makes their mean reward such a baseline, available at no training cost, and a response is scored by how it compares with its siblings. The KL penalty also moves out of the reward and into the loss, so it no longer distorts the advantages.

**What it achieves.** The critic, its optimiser state, its learning rate, and its failure mode are gone, while generation inside the loop and therefore on-policy learning are kept. It is cheaper than PPO despite generating more.

**What it leaves open.** Every token of a response receives the same advantage, so there is no credit assignment within a response. That costs little where the reward arrives only at the end, as it does here, and more where dense rewards genuinely exist. More importantly, a reward signal is still required, so the proxy problem returns unless the reward can be checked mechanically rather than learned.

---

## 4. Where the Chain Ends

The last step points outside the course. When correctness is decidable, e.g. mathematics with a known answer or code with a test suite, the learned reward model can be replaced by a checker, which is Reinforcement Learning with Verifiable Rewards. A checker cannot be hacked by confident prose and has no generalisation gap, which removes the difficulty that has shaped every step from [Section 3.4](#34-module-4-reward-model-training) onwards. It also does not apply to helpfulness or harmlessness, where there is nothing to check, which is the setting this course works in.

So the three routes are not a sequence of replacements but a stack. Supervised fine-tuning gives instruction-following, preference optimisation handles the subjective properties, and RL against verifiable rewards handles reasoning. Which one applies is decided by what kind of reward the task admits.

---

## 5. The Experiments

Every stage of the chain was implemented, not only read about. Each has its own project report, and each run is configuration-driven and hash-labelled so that a result can be traced back to the exact settings that produced it. The findings below are recorded for completeness. They are not part of the argument in [Section 3](#3-the-chain).

The foundations came first. Phase 1's material was implemented as four self-contained projects before any of it was used in anger, i.e. tabular Q-learning on Blackjack and on CartPole, a from-scratch deep Q-network on CartPole, and from-scratch PPO on LunarLander. The pipeline projects below then reuse that machinery rather than assuming it.

The pipeline itself fine-tunes Qwen2.5-0.5B with Low-Rank Adaptation throughout, on a single Apple Silicon workstation, using the Databricks Dolly-15k instruction set for supervised fine-tuning and the Anthropic HH-RLHF preference corpus for everything after it.

<figure id="tbl-runs" style="text-align: center;" markdown="1">

| Stage | Data | Status |
|---|---|---|
| Supervised fine-tuning | Dolly-15k demonstrations | Complete |
| Reward model | HH-RLHF preference pairs | Complete, plus a capacity study |
| PPO | HH-RLHF prompts | Complete |
| DPO | HH-RLHF preference pairs | Implemented, no run yet |
| GRPO | HH-RLHF prompts | Curriculum written, not implemented |

<figcaption>Table 1: Pipeline stages, their datasets, and the state of each.</figcaption>
</figure>

**Supervised fine-tuning.** The report's substantive contribution is an account of memory management across CUDA and Apple Silicon, which the three later stages all depend on and cross-reference.

**Reward model.** A capacity study established that the model's ceiling is set by the preference data rather than by adapter size, so more parameters do not help. Adversarial probing found the specific blind spots, i.e. responses that game the expected format and responses that are confidently wrong, which is a concrete instance of the proxy limitation described in [Section 3.4](#34-module-4-reward-model-training).

**PPO.** The held-out evaluation used paired per-prompt comparison with bootstrap confidence intervals, and found a resolvable but modest gain, with roughly a quarter of the improvement seen during training transferring to held-out prompts. A separate audit of the trainer's logged metrics found that post-end-of-sequence padding corrupts several of them, with a taxonomy of which aggregations survive it and which do not. Two negative results are reported rather than omitted, i.e. the win rate is not statistically resolvable, and a checkpoint sweep found no late-run decline that would justify early stopping.

**DPO and GRPO.** The DPO stage is implemented and its report is written except for the sections that need a run. GRPO exists as curriculum only. Neither carries results.

---

## 6. Reading Order

- **Phase 1, Reinforcement Learning Foundations.** Modules 1 and 2, i.e. Sections [3.1](#31-module-1-the-reinforcement-learning-problem) and [3.2](#32-module-2-policy-gradients-and-proximal-policy-optimisation). Read this first if reinforcement learning is unfamiliar, and skip it if it is not.
- **Phase 2, the Classical RLHF Pipeline.** Modules 3 to 5, i.e. the three stages that make up RLHF as originally published.
- **Phase 3, Modern Alignment Methods.** Module 6 onwards, i.e. the methods that simplify or replace the PPO stage.
- **The debugging appendix.** A cross-cutting reference of failure signals and remedies for every stage above, most useful once something is actually running.
- **The project reports.** One per implementation, each following the same structure from background and data through to reflections and next steps.

Each phase page opens with theory and closes with a hands-on project, so the curriculum and the reports can be read either in parallel or in sequence. For a reader who wants one page rather than the whole course, the full RLHF loop project is the most complete piece of the implementation work.
