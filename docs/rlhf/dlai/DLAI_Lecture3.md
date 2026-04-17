# 1. Overview

This note summarises Lecture 3 of the RLHF course, which covers the two datasets required before fine-tuning a large language model with Reinforcement Learning from Human Feedback. The worked examples use a Reddit post summarisation task with the OSS LLaMA 2 model.

---

# 2. Datasets Required for RLHF

RLHF requires two distinct datasets that must be prepared before any model training begins.

- A **preference dataset**, used to train the reward model.
- A **prompt dataset**, used in the reinforcement learning loop to fine-tune the base LLM.

Both datasets must come from the **same distribution**. In this course, all prompts are drawn from a corpus of Reddit posts.

---

# 3. Preference Dataset

## 3.1. Structure

Each example in the preference dataset is a dictionary with four keys.

- `input_text` — the prompt shown to the model (i.e. the Reddit post to be summarised).
- `candidate_0` — one candidate completion generated for the prompt.
- `candidate_1` — a second candidate completion for the same prompt.
- `choice` — an integer (0 or 1) recording which candidate a human labeller preferred.

## 3.2. Prompt Format Convention

All `input_text` values end with the literal string `[summary]: `. This is a deliberate formatting choice: the model must encounter the same instruction token at inference time as it did during training, so that it can recognise the pattern and generalise correctly. Mismatching training and inference formats will degrade performance.

## 3.3. Human Preference Labels

A human labeller is shown both `candidate_0` and `candidate_1` and selects the better summary. The selected candidate is the **winning** candidate; the other is the **losing** candidate. For example:

```
candidate_0: "When applying through a massive job portal, is just one HR
              person seeing ALL of them?"

candidate_1: "When applying to many jobs through a single university jobs
              portal, is just one HR person reading ALL my applications?"

choice: 1   →  candidate_1 is preferred
```

The reward model is trained on triplets of `(input_text, winning_candidate, losing_candidate)` and learns to output a scalar score indicating response quality.

## 3.4. Dataset Size Recommendations

The sample file `sample_preference.jsonl` is a small exploration subset. For production training, approximately 5,000–10,000 labelled examples are recommended.

## 3.5. Loading the Data

```python
import json

preference_data = []
with open('sample_preference.jsonl') as f:
    for line in f:
        preference_data.append(json.loads(line))

sample = preference_data[0]
print(sample.keys())
# dict_keys(['input_text', 'candidate_0', 'candidate_1', 'choice'])
```

---

# 4. Prompt Dataset

## 4.1. Structure

Each example in the prompt dataset is a dictionary with a single key.

- `input_text` — a prompt only, with no candidate completions or preference label.

Like the preference dataset, all prompts end with `[summary]: `.

## 4.2. Role in the RLHF Pipeline

Once the reward model has been trained on the preference dataset, the prompt dataset is fed into the RL loop. At each step, the base LLM generates a completion for a sampled prompt; the reward model scores that completion; and the policy is updated accordingly.

## 4.3. Loading the Data

```python
prompt_data = []
with open('sample_prompt.jsonl') as f:
    for line in f:
        prompt_data.append(json.loads(line))

print(len(prompt_data))  # 6 (exploration sample)
```

A helper function for readable inspection:

```python
def print_d(d):
    for key, val in d.items():
        print(f"key:{key}\nval:{val}\n")

print_d(prompt_data[0])
```

---

# 5. Key Takeaways

- RLHF requires two datasets: a preference dataset (for reward model training) and a prompt dataset (for RL fine-tuning).
- Both datasets must share the same prompt distribution and formatting convention.
- The `[summary]: ` suffix in `input_text` is a deliberate instruction token that must be preserved at inference time.
- Human preference labels are inherently subjective; careful labeller selection and clear annotation criteria are critical to dataset quality.
- The reward model is trained on `(prompt, winner, loser)` triplets and produces a scalar reward signal used to guide policy optimisation.
