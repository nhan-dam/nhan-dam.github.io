# Evaluating a Reinforcement-Learning-from-Human-Feedback Tuned Model

> Created on: 21 April 2026
>
> Updated on: 21 April 2026

## 1. Overview

The ultimate goal of the RLHF pipeline is not merely to run a training loop, but to produce a large language model (LLM) that performs a target task better than the original base model. Evaluation is therefore the critical closing step: it tells us whether the tuning actually worked, and by how much. This chapter covers the three broad families of evaluation strategy relevant to RLHF, explains what the training curves mean and how to interpret them, and then examines two advanced techniques, RLAIF and Auto Side-by-Side (AutoSxS), that push the frontier of scalable evaluation.

---

## 2. Evaluation Strategies for RLHF

LLM evaluation is an active area of research, but three approaches are most commonly used.

### 2.1. Training Curves

Training curves are plots of loss and reward against training steps, written to TensorBoard during the training process. They are the cheapest and most immediate signal of whether a model is learning at all. Two separate curves are relevant in the RLHF context: one from the reward model training phase, and one from the reinforcement learning (RL) loop itself.

### 2.2. Automated Metrics

Automated metrics are algorithmic measures of output quality that require a ground-truth reference, i.e. a human-written 'gold' completion. Familiar examples include accuracy and F1 for classification tasks, and the ROUGE family of metrics for generative tasks. ROUGE measures how similar a generated summary is to a reference summary, and is widely used in standard supervised summarisation.

However, ROUGE is a poor fit for evaluating RLHF. The reason is subtle but important: RLHF is not trying to produce text that is close to any particular reference text. It is trying to produce text that humans prefer, which is a different objective entirely. Research has shown that aggressively optimising for ROUGE can actually degrade the quality of an RLHF-tuned model, because the metric does not capture what RLHF aims to maximise.

### 2.3. Side-by-Side (SxS) Evaluation

Side-by-side evaluation presents completions from two models, the base (untuned) model and the RLHF-tuned model, for the same set of input prompts. A human (or an automated judge; see Section 5.2) then decides which completion they prefer. The result is expressed as a **win rate**, i.e. the proportion of prompts for which the tuned model produced the preferred completion. Alongside training curves, side-by-side evaluation is the most informative approach for RLHF.

---

## 3. Interpreting TensorBoard Training Curves

### 3.1. Reward Model Curves

The reward model is trained using a pairwise ranking objective, and the relevant loss is called the **rank loss**. Like any well-behaved loss, you want to see it decrease over training steps and then plateau (converge). In the notebook, the reward-model TensorBoard log shows exactly this pattern: the rank loss falls steadily and then flattens well before training ends, suggesting that fewer steps would have been sufficient and no additional data was wasted.

### 3.2. RL Loop Curves

The RL loop produces two key metrics: the **KL loss** and the **reward**.

The **KL loss** (Kullback–Leibler divergence) measures how far the tuned model's output distribution has drifted from the original base model's distribution. A small KL means the models behave similarly; a large KL means the tuned model has diverged significantly. In a healthy training run, you expect the KL loss to increase gradually and then plateau: the model is allowed to drift away from the base model in order to improve reward, but eventually the KL penalty in the objective prevents it from drifting too far.

The **reward** measures the score assigned by the reward model to the tuned model's completions. You want this to increase monotonically across training and then plateau, indicating that the model has learned to produce outputs that the reward model rates highly.

#### 3.2.1. Why Is the RL Loss Increasing?

A common point of confusion is the 'rl_loss' curve, which is also visible in TensorBoard and appears to **increase** rather than decrease. This surprises learners who expect all losses to go down. The explanation lies in what 'rl_loss' actually represents.

In the Vertex AI RLHF pipeline, the reported `rl_loss` is essentially the **negative reward** (often derived from the policy gradient objective). Because the optimiser is maximising reward, it is simultaneously minimising the negative reward. As the reward climbs upward, the negative of that reward climbs downward in absolute terms, which appears on the plot as an increasing curve when the sign convention flips or when the pipeline logs `−reward` directly. In short, the rl_loss increasing is the **mirror image** of the reward increasing, and is the expected, healthy behaviour for a model that is genuinely learning to produce higher-quality outputs. If both the reward and rl_loss were flat or erratic, that would indicate underfitting, which is exactly what the small-data (1% subset) logs showed.

### 3.3. Diagnosing Underfitting

In the notebook, the RL logs from training on only 1% of the data show neither a clear upward trend in reward nor a clear upward trend in KL loss. Both curves are essentially flat and noisy. This is the signature of **underfitting**: the model has not seen enough data to find a coherent direction in which to improve. The full-data logs, by contrast, show the reward climbing steadily and the KL loss rising and then levelling off, which is the expected healthy pattern.

---

## 4. Side-by-Side Evaluation in the Notebook

### 4.1. Loading the Evaluation Results

The pipeline accepts an evaluation dataset of prompts (no completions) and, once tuning is complete, runs a bulk inference job that generates completions for each prompt using both the base model and the tuned model. These results are written to a JSONL file in Google Cloud Storage, accessible via the 'Bulk Infer' component in the Vertex AI Pipelines console.

In the notebook, two JSONL files are loaded into Python lists, one for the untuned model and one for the tuned model. Each record in both files is a dictionary with the following structure.

- `inputs` → `inputs_pretokenized`: the full prompt string sent to the model.
- `prediction`: the completion produced by the model for that prompt.

Prompts are extracted from the tuned-model list (both files share identical prompts), and completions are extracted separately from each list. Everything is then assembled into a three-column pandas DataFrame. The full code is shown below.

```python
import json
import pandas as pd

# --- Load the JSONL files ---
eval_data_tuned = []
with open('eval_results_tuned.jsonl') as f:
    for line in f:
        eval_data_tuned.append(json.loads(line))

eval_data_untuned = []
with open('eval_results_untuned.jsonl') as f:
    for line in f:
        eval_data_untuned.append(json.loads(line))

# --- Extract prompts and completions ---
# Prompts are identical in both files; use the tuned list as the source.
prompts = [sample['inputs']['inputs_pretokenized']
           for sample in eval_data_tuned]

untuned_completions = [sample['prediction']
                       for sample in eval_data_untuned]

tuned_completions = [sample['prediction']
                     for sample in eval_data_tuned]

# --- Assemble the side-by-side DataFrame ---
results = pd.DataFrame(data={
    'prompt':      prompts,
    'base_model':  untuned_completions,
    'tuned_model': tuned_completions,
})

# Disable cell truncation so full completions are visible in the notebook.
pd.set_option('display.max_colwidth', None)

results
```

Each row of `results` corresponds to one evaluation prompt and shows the completions from both models side by side, making qualitative comparison straightforward.

### 4.2. What Does `pd.set_option('display.max_colwidth', None)` Do?

By default, pandas truncates the text in any DataFrame cell that exceeds a certain character limit (typically 50 characters), replacing the overflow with an ellipsis (`...`). This is useful for compact display but disastrous for side-by-side text comparison, where you need to read the full completion.

Calling `pd.set_option('display.max_colwidth', None)` removes this truncation limit entirely, instructing pandas to render every cell at its full width, regardless of length. The `None` argument means 'no limit'. The practical effect is that the entire prompt and both completions become visible in the notebook output, making it possible to actually read and compare them. Without this setting, the most informative parts of long completions would be invisible.

### 4.3. Qualitative Observations from the Results

One revealing difference visible in the results is the narrative **person** of the summaries. The tuned model tends to summarise Reddit posts in the first person, mirroring the voice of the original poster (e.g. 'Want to surprise my girlfriend with roses...'). The base model, by contrast, summarises in the third person (e.g. 'The author wants to surprise his girlfriend...'). This suggests the RLHF process, trained on human preference data from a Reddit summarisation task, has picked up on the stylistic convention that human raters preferred a summary that preserved the poster's own voice.

---

## 5. Advanced Evaluation Techniques

### 5.1. RLAIF: Reinforcement Learning from AI Feedback

**RLAIF** (Reinforcement Learning from AI Feedback) is a technique in which the preference dataset, traditionally labelled by human annotators, is instead generated by a capable off-the-shelf LLM acting as a labeller. The LLM is prompted to compare two completions and indicate which it prefers, producing the same (prompt, chosen, rejected) triples that a human would.

#### 5.1.1. Key Selling Points of RLAIF

The primary advantage of RLAIF is **scalability**. Human annotation is expensive, slow, and difficult to scale: recruiting, training, and quality-controlling thousands of human labellers is a significant organisational and financial undertaking. An LLM labeller can generate preference labels orders of magnitude faster and at a fraction of the cost, making it possible to produce much larger preference datasets than would be practical with humans alone.

A secondary advantage is **consistency**. Human annotators vary in their preferences and can be influenced by fatigue, cultural background, or ambiguous guidelines. An LLM labeller applies the same underlying model, resulting in more uniform labels (though this consistency can also encode the LLM's own biases, which is a known limitation).

#### 5.1.2. Is RLAIF Just Fine-Tuning to Mimic Model Y?

This is a genuinely subtle question, and the short answer is: not quite, but the concern is valid and worth taking seriously.

If model Y were used to directly generate completions and those completions were used as supervised fine-tuning (SFT) targets, then yes, you would simply be distilling model Y into the base LLM. The resulting model would try to copy model Y's outputs.

RLAIF is different because model Y is used to produce **preference labels**, not output targets. The base LLM still generates its own completions; model Y only says which of two completions it prefers. Those preference signals are then used to train a reward model, which is used in the PPO loop to adjust the base LLM's behaviour incrementally. The base LLM is therefore learning to improve within its own output space, guided by a signal derived from model Y's judgements, rather than being forced to copy model Y's outputs verbatim.

In practice, however, there is a real risk that the resulting model converges towards outputs that model Y would itself produce, since model Y's preferences naturally reflect its own style and knowledge. The hope is that if model Y is a good general-purpose judge of quality (e.g. clear, factual, helpful), the preferences it produces will capture broadly human-desirable properties rather than idiosyncratic stylistic quirks. The extent to which this hope is realised is an open empirical question, and it is one of the active research challenges in the RLAIF literature.

The practical payoff is substantial even if RLAIF is an imperfect substitute for human feedback: a model trained with AI-generated preference labels is likely to be considerably better than the untuned base model, especially when human labelling would simply be infeasible at the required scale.

### 5.2. AutoSxS: Automated Side-by-Side Evaluation

**AutoSxS** (Auto Side-by-Side) replaces the human judge in side-by-side evaluation with a third LLM, sometimes called an 'arbiter' or 'judge' model. The arbiter is given the prompt, the base model's completion, and the tuned model's completion, and is asked to decide which completion is better, along with a natural-language explanation of its reasoning. The result is a win rate that can be computed automatically over thousands of examples without any human involvement.

#### 5.2.1. Does AutoSxS Defeat the Purpose of RLHF?

This is another pointed and important question. The concern is: if we trained the model to satisfy human preferences, but then evaluate it using another LLM's preferences, have we not simply gone in a circle?

The answer requires distinguishing between the **training objective** and the **evaluation method**. The goal of RLHF remains to produce a model whose outputs better match human preferences. AutoSxS is a practical tool for measuring whether that goal has been achieved, not a substitute for it. The implicit assumption is that a capable arbiter LLM is a reasonable proxy for human judgement, in the same way that ROUGE was once assumed to be a reasonable proxy for summarisation quality (though, as discussed, ROUGE turned out to be a poor proxy for RLHF specifically).

Several considerations support the use of AutoSxS as a useful evaluation tool even if it is imperfect. First, human SxS evaluation is expensive and slow, and often cannot be run at the scale needed to get statistically reliable win rates. AutoSxS enables evaluation on thousands of examples in minutes. Second, if the arbiter model is itself a strong, general-purpose model (e.g. a frontier model prompted with a careful rubric), its preferences are likely to correlate reasonably well with average human preferences, at least for well-defined tasks like summarisation. Third, AutoSxS does not replace human evaluation entirely: it is used as a fast, scalable **screening** tool, with human evaluation reserved for final validation or for ambiguous cases where the arbiter is uncertain.

The deeper philosophical point is that 'human preference' is not monolithic. Even human evaluators disagree with each other substantially. Using a strong LLM as a judge is no more circular than using one group of humans to train a reward model and a different group to evaluate the final model: the signal is noisy and approximate in both cases, and the question is whether it is good enough to be directionally useful. Current evidence suggests that, for many tasks, it is.

---

## 6. Accessing Artefacts in Google Cloud

For completeness, the following describes how to retrieve the relevant TensorBoard logs and evaluation outputs from the Vertex AI Pipelines console.

To find TensorBoard logs, navigate to the pipeline run in the Cloud Console under Vertex AI > Pipelines > Runs, select the RLHF Train Template run, and click on either the 'reward model trainer' component (for reward model logs) or the 'reinforcer' component (for RL loop logs). Each component exposes a 'TensorBoard Metrics' output artefact whose URI points to a Google Cloud Storage path containing the log files, which can be downloaded and visualised locally.

To find bulk inference results, click on the 'Perform Inference' component and then the 'Bulk Infer' sub-component. The output parameter `output_prediction_gcs_path` provides a GCS path to the JSONL file containing the evaluation completions.

---

## 7. Summary

The table below maps each evaluation approach to its practical role in an RLHF workflow.

| Approach | What it measures | Suitable for RLHF? | Key limitation |
|---|---|---|---|
| Training curves (reward, KL) | Whether the model is learning at all. | Yes, highly useful. | Does not measure final output quality directly. |
| ROUGE / automated metrics | Similarity to a reference text. | No, poorly suited. | Does not capture human preference alignment. |
| Human SxS evaluation | Direct human preference between two models. | Yes, gold standard. | Expensive and slow at scale. |
| AutoSxS (LLM arbiter) | LLM-proxy of human preference. | Yes, scalable screening. | Arbiter may not perfectly reflect human preferences. |
| RLAIF (LLM-labelled preference data) | Enables RLHF without human annotators. | Yes, scalable training. | Risk of encoding the labeller LLM's own biases. |

---

## Appendix A: Further Discussion — If the Arbiter LLM Is That Good, Why Do We Need RLHF at All?

Sections 5.1.2 and 5.2.1 raise a natural follow-up question: if a frontier LLM is capable enough to reliably judge output quality (for RLAIF) or evaluate model performance (for AutoSxS), why not simply use that frontier model directly and skip fine-tuning the base model altogether?

This appendix unpacks why the answer is not as straightforward as it might seem, by distinguishing between three separate concepts that are easy to conflate: the ability to *judge* quality, the ability to *generate* quality, and the economics of *deploying* quality at scale.

### A.1. Judging Is Easier Than Generating

The most fundamental point is that evaluating whether an output is good is a substantially easier cognitive task than producing a good output in the first place. A food critic who has eaten at ten thousand restaurants can reliably tell you whether a dish is well-executed, even if they could not cook it themselves. A skilled editor can spot a weak paragraph in seconds even if writing an equally strong replacement would take them considerably longer.

The same asymmetry applies to LLMs. A frontier model can reliably identify which of two Llama 2 summaries is more faithful, concise, and natural, even in cases where Llama 2 itself could not have generated the better one unprompted. RLHF exploits this asymmetry deliberately: it uses the *easier* task of judging to provide a training signal for the *harder* task of generating. Crucially, the arbiter does not need to know *how* to produce a better output; it only needs to recognise *which* of two candidate outputs is better. That is a considerably lower bar, which is why a frontier model can serve as a useful judge even for a model that is significantly weaker than it.

### A.2. The Arbiter Cannot Transfer Its Weights

A second, more practical point is that knowing the right answer does not automatically make the base model capable of producing it. Even if a frontier LLM could perfectly articulate what a good summary looks like, the base model cannot absorb that knowledge through description alone. The base model's behaviour is determined by its weights, and those weights can only be updated through training. RLHF is precisely the mechanism that translates the arbiter's preference judgements into concrete weight updates via the reward model and the PPO loop. The arbiter provides the compass; RLHF is the engine that moves the model in the indicated direction.

An analogy: a GPS can tell you with perfect accuracy that you need to turn left in 200 metres. But the GPS cannot drive the car. Something still has to do the moving.

### A.3. Deployment Economics and Ownership

Even if you were willing to accept the theoretical circularity, using a frontier model directly at inference time raises serious practical obstacles. Frontier models are expensive to call, introduce latency, are subject to rate limits, and, critically, are owned by a third party. Organisations that need low-latency, low-cost, on-premises, or domain-specialised behaviour cannot rely on an external API for every user request. The goal of fine-tuning a smaller base model with RLHF is to *internalise* improved behaviour into a model you own and can deploy freely, without ongoing dependency on a third-party service.

This is also why distillation, i.e. generating completions from the frontier model and using them as supervised fine-tuning targets, is a related but distinct strategy. Distillation teaches the base model to copy the frontier model's outputs for the prompts it has seen. RLHF, by contrast, teaches the base model a more general disposition towards quality, which can generalise better to novel prompts not seen during training.

### A.4. Reconciling the Apparent Circularity

It is worth being precise about what the apparent circularity actually is. The concern is roughly: 'We train the base model using an LLM's preferences (RLAIF), then evaluate it using an LLM's preferences (AutoSxS). Are we just measuring whether the base model has learned to satisfy the LLM, rather than whether it has learned to satisfy humans?'

This concern is legitimate but not fatal, for two reasons. First, the assumption underlying both RLAIF and AutoSxS is that a capable frontier model's preferences are a *reasonable proxy* for average human preferences, at least for well-defined tasks with relatively unambiguous quality criteria (e.g. factuality, conciseness, grammaticality in summarisation). If this assumption holds, then satisfying the LLM's preferences and satisfying human preferences are approximately the same objective, and there is no real circularity. Second, even if the proxy is imperfect, the question is not whether it is perfect but whether it is *good enough to be directionally useful*. A model trained and evaluated using an imperfect proxy can still be substantially better than the untuned base model, which is the comparison that matters in practice.

The deeper insight is that 'human preference' is not a monolithic gold standard either. Human annotators disagree with each other substantially, are subject to fatigue and cultural bias, and often cannot articulate why they prefer one output over another. Using a frontier LLM as a judge is not obviously worse than using a panel of crowdsourced human annotators; it is simply a different, more scalable, and arguably more consistent approximation of the same underlying signal.

### A.5. Summary of the Argument

The three points above can be restated concisely. The arbiter LLM is good at judging but that judgement does not automatically improve the base model, so RLHF is still needed to translate judgements into weight updates. The base model after fine-tuning is cheaper and faster to deploy than calling the frontier model at inference time, and is owned outright by the organisation. And the apparent circularity of using an LLM to train and evaluate is not fatal, because the frontier model's preferences are a reasonable and scalable proxy for human preferences for many tasks. RLHF with AI feedback is therefore best understood not as a replacement for the idea of human-preference alignment, but as a practical engineering solution for achieving that alignment at a scale and cost that would be impossible with human annotators alone.
