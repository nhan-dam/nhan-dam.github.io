# Phase 2 — The Classical RLHF Pipeline

> Created on: 22 April 2026
>
> Updated on: 16 July 2026

Reinforcement learning from human feedback (RLHF), as originally described in 'Learning to Summarize from Human Feedback' (Stiennon et al., 2020) and subsequently in InstructGPT (Ouyang et al., 2022), consists of three sequential stages, each covered by one module of this phase. Supervised fine-tuning (SFT) turns a base language model into an instruction follower (Module 3). Reward model (RM) training distils human preferences into a scalar scoring function (Module 4). Proximal policy optimisation (PPO) then optimises the SFT model against the reward model (Module 5), reusing the PPO machinery developed in Module 2 ([Algorithm 4](phase01.md#alg-ppo)). Each stage consumes the artefact produced by the previous one, so the modules must be completed in order.

## 3. Module 3: Supervised Fine-Tuning

### 3.1. Theory

A pre-trained language model is a next-token predictor trained on web-scale text. It has no notion of instruction-following: prompted with a question, it is as likely to continue with another question as to answer it, because both continuations are plausible web text. The SFT step addresses this by fine-tuning the base model on a curated dataset of (prompt, response) pairs, where the responses are high-quality demonstrations written by human annotators or distilled from a larger model.

SFT is standard supervised learning with cross-entropy loss. Given a prompt $x$ and a target response $y = (y_1, \ldots, y_T)$, the SFT loss is

<span id="eq-sft-loss"></span>

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid x, y_{\lt t}), \tag{21}$$

where $y_{\lt t} = (y_1, \ldots, y_{t-1})$ denotes the response tokens preceding position $t$, and $p_\theta$ is the model's next-token distribution. Note that the sum in [(21)](#eq-sft-loss) runs over response tokens only. The prompt tokens condition the prediction but contribute no loss terms of their own, because the model is never asked to generate a prompt. Computing loss on the prompt as well would dilute the gradient signal with ordinary language-modelling terms unrelated to the behaviour being taught. In implementations the prompt and response are concatenated into a single sequence $[x, y]$ and fed through the model in one forward pass, which produces a next-token prediction at every position. Restricting the loss to response positions therefore requires masking out the prompt positions from the cross-entropy. This masking is what is known as the 'completion-only loss'.

The SFT model is denoted $\pi_{\text{ref}}$ and called the **reference policy**, because the later PPO stage treats it as a fixed anchor: a Kullback–Leibler (KL) divergence penalty (introduced in [Section 5.1](#51-theory)) punishes the policy for drifting away from it.

**LoRA for efficient fine-tuning.** Full fine-tuning of a multi-billion-parameter model is impractical on a single workstation: every weight matrix needs gradients, optimiser state, and storage for a full model copy per experiment. Low-Rank Adaptation (LoRA) (Hu et al., 2021) exploits the empirical observation that the *change* a fine-tune makes to a weight matrix has low intrinsic rank, even when the matrix itself does not. LoRA therefore freezes each pre-trained weight matrix $W \in \mathbb{R}^{d \times k}$ and learns an additive update factorised as $\Delta W = BA$, with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$, typically injected into the attention projection layers. The adapted forward pass is

$$h = Wx + \frac{\alpha}{r} BAx,$$

where $\alpha$ is a constant scaling factor (conventionally $\alpha = 2r$) that decouples the update's effective magnitude from the choice of rank. $B$ is initialised to zero, so $\Delta W = 0$ at the start of training and the adapted model behaves exactly like the base model, with no disruptive random perturbation. The number of trainable parameters scales as $O(r(d + k))$ rather than $O(dk)$, and a rank of $r = 16$ to $64$ is typically sufficient for instruction following. Quantised LoRA (QLoRA) further quantises the frozen base weights to 4-bit precision, reducing memory by a factor of approximately 4 with minimal quality degradation.

In practice $r$ is rarely the first hyperparameter to tune. Performance is fairly flat across the usual $r = 8$ to $64$ band, because once $r$ exceeds the update's intrinsic rank the extra capacity mostly adds parameters and overfitting risk. Most pipelines therefore fix a default of 16 or 32. The learning rate, the choice of which modules to adapt (e.g. attention-only versus all linear layers), and the effective scale $\alpha / r$ are higher-leverage knobs and are swept first. When $r$ is tuned at all, it is usually a coarse sweep jointly with $\alpha$. The Adaptive LoRA (AdaLoRA) variant allocates rank adaptively per layer, automating the choice.

### 3.2. Hands-on Project: Supervised Fine-Tuning with Low-Rank Adaptation

**Objective.** Fine-tune Qwen2.5-0.5B (or a comparable small model) on a dialogue dataset to produce $\pi_{\text{ref}}$. A small base model is used so that the four-model PPO stage of [Section 5.4](#54-hands-on-project-full-reinforcement-learning-from-human-feedback-training-loop) fits in unified memory on a single workstation.

**Setup.**

```bash
mkdir /Volumes/ML_Workspace/projects/rlhf-course
cd /Volumes/ML_Workspace/projects/rlhf-course
uv init --python 3.12
uv add torch transformers datasets tokenizers accelerate peft trl
uv add huggingface_hub tensorboard rich
```

**Dataset.** Use `databricks/databricks-dolly-15k`, a permissively licensed instruction dataset with 15,000 examples. Each example holds an `instruction`, an optional `context` passage, and a `response`. Mapping the examples to explicit `prompt`/`completion` columns lets `trl.SFTTrainer` apply the completion-only loss of [(21)](#eq-sft-loss): the column boundary tells the trainer where the prompt ends.

**Implementation outline.**

```python
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Map Dolly to prompt/completion columns so the trainer can mask the prompt
def to_prompt_completion(example):
    context = f"### Context:\n{example['context']}\n\n" if example["context"] else ""
    return {
        "prompt": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"{context}### Response:\n"
        ),
        "completion": example["response"],
    }

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
dataset = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.05, seed=42)  # held-out split for overfitting detection

# warmup_ratio is deprecated in transformers >= 5.2; derive absolute warmup steps
num_epochs      = 3
effective_batch = 8 * 2   # per-device batch size x gradient accumulation
total_steps     = math.ceil(len(dataset["train"]) / effective_batch) * num_epochs

training_args = SFTConfig(
    output_dir="/Volumes/ML_Workspace/projects/rlhf-course/sft-output",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,   # effective batch size = 8 x 2 = 16
    learning_rate=2e-4,
    warmup_steps=round(0.03 * total_steps),
    gradient_checkpointing=True,     # trades recompute for activation memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    # Keep the lowest eval-loss checkpoint rather than the final one
    # (save_steps must be a multiple of eval_steps for this to work)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,           # supported on M4 Pro
    max_length=512,
    completion_only_loss=True,   # loss on response tokens only
    report_to="tensorboard",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    processing_class=tokenizer,
)
trainer.train()
# save_model writes the LoRA adapter and the tokenizer together, so the reward
# modelling stage can load both from this one path
trainer.save_model(
    "/Volumes/ML_Workspace/projects/rlhf-course/sft-model"
)
```

**Implementation tip.** There are two equivalent routes to a LoRA-wrapped model. The snippet above wraps manually with `get_peft_model(model, lora_config)` and hands the resulting parameter-efficient fine-tuning (PEFT) model to the trainer, which makes the 'freeze the base, inject trainable adapters' step explicit and lets you call `model.print_trainable_parameters()` as a sanity check before training. Alternatively, pass the plain model together with `peft_config=lora_config` to the trainer, which then calls `get_peft_model` internally and handles related preparation for you (e.g. enabling input gradients under gradient checkpointing). The manual route is used here, where LoRA is the lesson. The later projects treat LoRA as plumbing and use the trainer-managed route instead (see the snippet in [Section 4.3](#43-hands-on-project-reward-model-training)), which is also the only sensible route for the PPO stage, where the trainer manages the PEFT structure itself to recover the frozen reference policy. Do not mix the two routes: passing an already-wrapped PEFT model together with a `peft_config` is an error, and TRL's trainers raise a `ValueError` at construction. Give the trainer either a PEFT model and no `peft_config` (as here), or a plain model plus the config (as in [Section 4.3](#43-hands-on-project-reward-model-training)).

**What to observe.** Monitor training and validation loss, and inspect qualitative outputs on held-out prompts. A well-tuned SFT model should consistently produce coherent, on-format responses before proceeding to reward modelling. Falling training loss with flat or rising validation loss indicates overfitting. In that case, reduce the number of epochs or increase dropout.

---

## 4. Module 4: Reward Model Training

### 4.1. Theory

The reward model $r_\phi(x, y)$ is a neural network that maps a (prompt, response) pair to a scalar, the estimated human preference score. It is trained on a dataset of pairwise comparisons: for each prompt $x$, a human annotator has indicated that response $y_w$ ('chosen') is preferred over $y_l$ ('rejected'). Pairwise comparison is used rather than absolute scoring because humans are far more consistent at ranking two responses than at assigning calibrated numeric scores.

**The Bradley–Terry model.** The training objective derives from the **Bradley–Terry model** of pairwise preferences, which assumes each response has a latent quality score and that the probability of preferring one response over another depends only on the *difference* of their scores. Identifying the latent score with the reward model's output gives

<span id="eq-bt-pref"></span>

$$p(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right), \tag{22}$$

where $y_w \succ y_l$ denotes that $y_w$ is preferred over $y_l$ ($\succ$ is the preference-ordering symbol, a ranking analogue of $>$), and $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid function: a large positive score gap makes the preference near-certain, a zero gap makes it a coin flip. Maximising the log-likelihood of the observed preferences under [(22)](#eq-bt-pref) gives the loss

<span id="eq-rm-loss"></span>

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]. \tag{23}$$

One consequence of modelling only differences: [(22)](#eq-bt-pref) is unchanged if a constant is added to every reward, so the absolute reward scale is unanchored and can drift during training. This is harmless for ranking but inconvenient for the PPO stage, which consumes raw scores. A small auxiliary penalty that pins the mean reward near zero (the `center_rewards_coefficient` in the project below) removes this degeneracy.

**Architecture.** The reward model is typically initialised from $\pi_{\text{ref}}$ (the SFT model), with the language modelling head replaced by a linear layer projecting to a single scalar. The scalar is read at the position of the *last* token of the sequence: under causal attention, only the final position attends to the entire prompt and response, so it is the only position whose hidden state summarises the whole pair. Initialising from SFT rather than from the base model is important: the reward model must 'understand' the kinds of responses being scored, and starting from a model already in the instruction-following distribution accelerates convergence.

**Reward model limitations.** The RM is a proxy for human preference, not the preference itself. It is trained on a finite, potentially biased dataset and will generalise imperfectly. Specifically, it will assign high scores to responses that superficially resemble preferred responses (e.g. long, confident-sounding text) even when the content is poor. This is the root cause of reward hacking in the PPO stage.

### 4.2. Intuitions for Reward Model Quality {: #rm-quality-checks}

The following checks should be performed before using the RM in the PPO stage.

- **Pairwise accuracy on a held-out set.** A well-trained RM should achieve at least 65–70% accuracy on held-out preference pairs. Below 60% suggests the data is too noisy or the model is undertrained.
- **Reward distribution.** Plot the distribution of $r_\phi(x, y_w)$ and $r_\phi(x, y_l)$. The distributions should be separated, with the chosen distribution shifted higher. If they overlap substantially, the RM has low discriminative power.
- **Out-of-distribution behaviour.** Manually probe the RM with adversarial inputs: very long repetitive responses, confident but factually wrong responses, and responses in unexpected formats. The RM's scores on these inputs preview what the PPO stage will optimise towards.

### 4.3. Hands-on Project: Reward Model Training

**Objective.** Train a reward model on the `Anthropic/hh-rlhf` dataset, a corpus of human preference pairs in exactly the `chosen`/`rejected` format that [(23)](#eq-rm-loss) requires.

**Depends on [Section 3.2](#32-hands-on-project-supervised-fine-tuning-with-low-rank-adaptation).** This project consumes the LoRA adapter trained there. It first merges that adapter into the base model to produce a standalone `sft-model-merged` checkpoint, then initialises the reward model from the merged weights, swapping the language-model head for a scalar reward head. The merged checkpoint is also reused by [Section 5.4](#54-hands-on-project-full-reinforcement-learning-from-human-feedback-training-loop), so this project must be run before the PPO stage.

**Implementation outline.**

```python
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(
    "/Volumes/ML_Workspace/projects/rlhf-course/sft-model"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# The SFT step saved a LoRA adapter; merge it into the base model first. The
# tokenizer is saved alongside so the PPO stage can load both from one path.
sft_model = AutoPeftModelForCausalLM.from_pretrained(
    "/Volumes/ML_Workspace/projects/rlhf-course/sft-model", dtype=torch.bfloat16
).merge_and_unload()
sft_model.save_pretrained(
    "/Volumes/ML_Workspace/projects/rlhf-course/sft-model-merged"
)
tokenizer.save_pretrained(
    "/Volumes/ML_Workspace/projects/rlhf-course/sft-model-merged"
)

# Initialise from the merged SFT model, swapping the language model (LM) head
# for a scalar head
model = AutoModelForSequenceClassification.from_pretrained(
    "/Volumes/ML_Workspace/projects/rlhf-course/sft-model-merged",
    num_labels=1,
    dtype=torch.bfloat16,
)
# Decoder-only sequence classification pools the logit of the last non-padding
# token, so the model config must know the pad token id
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA on the backbone; the scalar head is newly initialised, so it is trained
# in full via modules_to_save rather than adapted
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_CLS",
    modules_to_save=["score"],
)

train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
test_dataset = load_dataset("Anthropic/hh-rlhf", split="test")  # the authors' held-out split

# max_length below FILTERS out over-long pairs rather than truncating them,
# and RewardTrainer silently applies the same filter to any eval set it is
# given. Filtering the test split once, up front, exactly as training
# tokenisation sees the pairs (EOS appended) guarantees the in-training
# subsample keeps all 1,000 pairs and gives the post-training acceptance
# gate the same length-admissible distribution the model is trained on.
def within_cap(batch, cap=512):        # cap matches max_length below
    eos = tokenizer.eos_token
    chosen = tokenizer([t if t.endswith(eos) else t + eos
                        for t in batch["chosen"]])["input_ids"]
    rejected = tokenizer([t if t.endswith(eos) else t + eos
                          for t in batch["rejected"]])["input_ids"]
    return [len(c) <= cap and len(r) <= cap for c, r in zip(chosen, rejected)]

gate_dataset = test_dataset.filter(within_cap, batched=True)
eval_dataset = gate_dataset.shuffle(seed=42).select(range(1_000))

config = RewardConfig(
    output_dir="/Volumes/ML_Workspace/projects/rlhf-course/rm-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,      # small to cap the eval-time memory peak
    gradient_accumulation_steps=4,     # effective batch size = 4 x 4 = 16
    learning_rate=1e-4,                # adapters are freshly initialised and need
                                       # larger steps than full fine-tuning (1e-5)
    bf16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    # Keep the lowest eval-loss checkpoint: the Bradley-Terry loss is strictly
    # monotone in the reward margin, so it is a sound selection metric
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_length=512,                    # pairs with a longer side are filtered out
    center_rewards_coefficient=0.01,   # pins rewards near zero (see Section 4.1)
    report_to="tensorboard",
    seed=42,
)

trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)
trainer.train()
# Saves the LoRA adapter, the fully trained scalar head, and the tokenizer
trainer.save_model(
    "/Volumes/ML_Workspace/projects/rlhf-course/rm-model"
)
```

**Implementation tips.** The module names passed to `LoraConfig` are simply the attribute names in the model's `nn.Module` tree, defined by the model's implementation in `transformers` (e.g. `Qwen2Attention` declares `self.q_proj = nn.Linear(...)` and `Qwen2ForSequenceClassification` declares `self.score = nn.Linear(...)`). To discover them, print the model with `print(model)` or list fully qualified names with `[n for n, _ in model.named_modules()]`, which yields entries such as `model.layers.0.self_attn.q_proj`. PEFT matches each `target_modules` entry against the *suffix* of those qualified names, so `"q_proj"` catches that projection in every layer, and the modules actually matched can be confirmed afterwards via `peft_model.targeted_module_names`. Two caveats: the names are architecture-specific (GPT-2 uses a fused `c_attn` rather than separate query/key/value projections, and BERT-style models name the classification head `classifier` rather than `score`), and a name that matches nothing raises an error rather than silently training nothing. The shortcut `target_modules="all-linear"` adapts every linear layer except the output head, useful when inspecting the module tree is inconvenient.

**What to observe.** Monitor the evaluation loss: `RewardTrainer` reports only `eval_loss` during evaluation, and since the Bradley–Terry loss of [(23)](#eq-rm-loss) is strictly monotone in the reward margin, falling evaluation loss tracks rising pairwise accuracy. After training, compute held-out pairwise accuracy directly, i.e. the fraction of pairs where the score difference inside the sigmoid of [(23)](#eq-rm-loss) has the correct sign, on the whole filtered test split (`gate_dataset` above) rather than on the 1,000-pair subsample. At 1,000 pairs the binomial standard error of the accuracy (about 0.015) exceeds the typical margin over an acceptance floor, while the full filtered split (7,952 of 8,552 pairs at cap 512) brings it down to about 0.005. Judge the floor to within one standard error (accept when accuracy is at least the floor minus one standard error, with a floor of 0.65 tied to human agreement on HH-RLHF), and run the diagnostic checks described in [Section 4.2](#rm-quality-checks) before proceeding to the PPO stage. Ties count as failures under the strict inequality, which is a further reason the gate must score filtered rather than truncated pairs: a pair whose two sides diverge beyond the cap truncates to identical texts and can only tie.

---

## 5. Module 5: The Full Reinforcement Learning from Human Feedback Training Loop {: #ppo-rlhf-loop}

### 5.1. Theory

The PPO stage ties together all previous components. The policy (initialised from $\pi_{\text{ref}}$, the SFT model) generates responses, the reward model scores them, and PPO updates the policy to maximise the scores.

**Text generation as a reinforcement learning problem.** To apply the agent–environment formalism of [Module 1](phase01.md#rl-problem), generation is cast as a Markov decision process (MDP). The state $s_t = (x, a_1, \ldots, a_{t-1})$ is the prompt plus the tokens generated so far. The action $a_t$ is the next token, chosen from the vocabulary. The transition is deterministic: the chosen token is appended to the state. An episode is one complete generation, ending when the policy emits an end-of-sequence (EOS) token or hits the length limit. The environment reward is sparse: zero at every intermediate step, with the reward model's score $r_\phi(x, y)$ delivered once at the terminal step, since the RM can only judge a complete response.

**The KL penalty.** The objective is not pure reward maximisation. An unconstrained policy would quickly learn to produce degenerate outputs that achieve high reward model scores through exploitation rather than genuine quality improvement, which is the reward hacking failure anticipated in [Section 4.1](#41-theory). It is mitigated by adding a **Kullback–Leibler (KL) divergence penalty** to the reward signal:

<span id="eq-kl-reward"></span>

$$r(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}\left[\pi_\theta(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)\right]. \tag{24}$$

The penalty term punishes the policy for moving probability mass away from the reference. The intuition: the RM is only trustworthy on the distribution of responses it was trained on, which is (approximately) the SFT distribution. The further $\pi_\theta$ drifts from $\pi_{\text{ref}}$, the less the RM's scores can be trusted, so [(24)](#eq-kl-reward) makes distance from the reference a cost that high RM scores must outweigh.

**Per-token KL estimation.** The KL term in [(24)](#eq-kl-reward) is an expectation over all possible responses, which is intractable to compute exactly. It is instead estimated from the single response actually sampled. For each generated token, define

<span id="eq-token-kl"></span>

$$\text{KL}_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t), \tag{25}$$

the log-probability ratio of the sampled action under the two policies. Because the response is sampled from $\pi_\theta$, the sum of [(25)](#eq-token-kl) over the response tokens is an unbiased single-sample estimate of the sequence-level KL divergence. Rather than subtracting this sum from the terminal reward in one lump, each $-\beta \cdot \text{KL}_t$ is attributed to its own timestep as a per-token shaped reward. The total penalty is identical either way, but the per-token form gives the credit-assignment machinery (advantage estimation) a dense signal locating *where* in the response the policy drifted, instead of a single end-of-episode correction.

**The four-model setup.** Classical PPO RLHF requires four models in memory simultaneously:

- The **actor** $\pi_\theta$, the policy being trained.
- The **reference** $\pi_{\text{ref}}$, a frozen copy of the SFT model used for KL computation.
- The **reward model** $r_\phi$, frozen.
- The **critic** $V_\psi$, used for advantage estimation, typically initialised from $r_\phi$, whose response-quality estimates are a better starting point for value prediction than a fresh head.

On the M4 Pro with 64 GB of unified memory, the four models fit comfortably for a 0.5B-parameter base, which is why this phase uses one. Larger backbones are tight: a 3B base requires LoRA and 4-bit quantisation to run four models concurrently, and 7B+ models additionally need the reference and reward models kept in 4-bit.

[Algorithm 5](#alg-rlhf-ppo) summarises the full loop. The inner optimisation step is exactly the clipped PPO update of Phase 1 ([Algorithm 4](phase01.md#alg-ppo)). What changes is where the reward comes from.

<figure id="alg-rlhf-ppo" style="text-align: center;" markdown="1">
<div style="border: 1px solid #ccc; display: inline-block; text-align: left; padding: 1em; font-family: monospace;" markdown="1">

**Input:**<br>
$\mathcal{D}$: prompt dataset<br>
$\pi_\theta$: policy, initialised from $\pi_{\text{ref}}$ (the SFT model)<br>
$\pi_{\text{ref}}$: frozen reference policy<br>
$r_\phi$: frozen reward model<br>
$V_\psi$: value network, initialised from $r_\phi$<br>
$\beta$: KL penalty coefficient<br>
$N$: prompts per rollout batch

**Output:**<br>
$\theta$: trained policy parameters

**repeat until** converged:<br>
$\quad$ *// 1. Rollout: generate with the frozen current policy*<br>
$\quad$ $\theta_\text{old} \leftarrow \theta$<br>
$\quad$ Sample $N$ prompts $x \sim \mathcal{D}$<br>
$\quad$ **for each** prompt $x$:<br>
$\quad\quad$ Generate response $y = (a_1, \ldots, a_T) \sim \pi_{\theta_\text{old}}(\cdot \mid x)$<br>
$\quad\quad$ Record $\log \pi_{\theta_\text{old}}(a_t \mid s_t)$, $\log \pi_{\text{ref}}(a_t \mid s_t)$, and $V_\psi(s_t)$ for $t = 1, \ldots, T$<br>
$\quad\quad$ *// 2. Score: KL-shaped per-token rewards, RM score at the terminal step*<br>
$\quad\quad$ **for** $t = 1, \ldots, T$:<br>
$\quad\quad\quad$ $R_t \leftarrow -\beta \cdot \text{KL}_t$ *// [(25)](#eq-token-kl) under $\pi_{\theta_\text{old}}$, from the recorded log-probabilities*<br>
$\quad\quad$ $R_T \leftarrow R_T + r_\phi(x, y)$ *// reward model scores the complete response*<br>
$\quad$ *// 3. Update: GAE advantages, then the clipped PPO update*<br>
$\quad$ Compute $\hat{A}_t$ from the recorded $\{(s_t, a_t, R_t)\}$ of all $N$ responses and $V_\psi$ ([Section 2.1.3](phase01.md#gae))<br>
$\quad$ Update $\theta, \psi$ for $K$ epochs with the clipped surrogate objective ([Algorithm 4](phase01.md#alg-ppo), step 3); the rewards $R_t$ stay fixed throughout

**return** $\theta$

</div>
<figcaption>Algorithm 5: PPO-based RLHF training loop with KL-shaped reward.</figcaption>
</figure>

**The $\beta$ hyperparameter.** The KL coefficient $\beta$ in [(24)](#eq-kl-reward) controls the trade-off between reward maximisation and staying close to $\pi_{\text{ref}}$. A high $\beta$ produces a conservative policy that closely tracks $\pi_{\text{ref}}$. A low $\beta$ allows aggressive optimisation of the reward model, increasing the risk of hacking. Values of $\beta = 0.1$ to $0.3$ are typical. In practice, $\beta$ is the first hyperparameter to tune if the policy drifts too far from the reference or if reward hacking is observed.

### 5.2. Debugging the Reinforcement Learning from Human Feedback Training Loop

Training instability in the PPO RLHF loop is common. The following signals and remedies cover the most frequent failure modes.

**KL divergence grows unboundedly.** The policy is drifting from the reference. Increase $\beta$ in [(24)](#eq-kl-reward), reduce the learning rate, or reduce the number of PPO mini-batch epochs.

**Reward increases sharply then plateaus or collapses.** This is classic reward hacking. Inspect the actual generated text: it will exhibit repetition, format gaming (e.g. very long lists), or other superficially reward-signal-shaped artefacts. The remedy is to either increase $\beta$, retrain the reward model with adversarial examples, or switch to direct preference optimisation (DPO) (see Phase 3).

**Critic value loss diverges.** The value network cannot track the rapidly changing policy. Use a smaller learning rate for the critic than for the actor, or detach the critic's gradient from the actor's forward pass.

**Entropy collapses.** The actor is converging to a near-deterministic policy. Increase the entropy bonus coefficient $c_2$ in the PPO loss.

**Memory exhaustion.** Reduce batch size, reduce the sequence length, or apply more aggressive quantisation to the frozen reference and reward models.

### 5.3. Suggested Reading

- Stiennon et al. (2020), *Learning to Summarize from Human Feedback.* The first large-scale RLHF paper. Read before the project to see how the three-step pipeline was originally applied.
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback.* The canonical reference for the full SFT → RM → PPO pipeline. Read before the project alongside Stiennon et al. (2020).

#### 5.3.1. Stiennon et al. (2020): *Learning to Summarize from Human Feedback*

**Overview**

Stiennon et al. (2020) trained summarisation policies directly from human preference comparisons, establishing the SFT → RM → PPO recipe that this phase implements. Working on the TL;DR dataset of Reddit posts with GPT-style models of 1.3B and 6.7B parameters, the paper demonstrated that optimising a learned preference model with RL produces summaries that humans prefer to those of much larger supervised models.

**Problem**

Summarisation models were trained with maximum-likelihood on reference summaries and evaluated with automatic metrics such as ROUGE, but both are proxies: a model can match references and score well on ROUGE while producing summaries humans judge as mediocre. Scaling data and parameters improves the proxy without necessarily improving the quality that actually matters, because the training objective never sees a human judgement.

**Key Contributions**

The paper contributed the complete preference-learning pipeline in its modern form. A large dataset of human comparisons between summary pairs was collected under unusually tight quality control, with researcher–labeller agreement monitored throughout, and the authors attribute much of the result to this data quality. The reward model was trained on these comparisons with the Bradley–Terry loss of [(23)](#eq-rm-loss), and the policy was optimised with PPO against the RM under a per-token KL penalty towards the SFT policy, exactly the shaped reward of [(24)](#eq-kl-reward). The paper also documented RM **over-optimisation**: pushing the policy to maximise a frozen RM (by lowering the KL penalty or using best-of-$N$ sampling with large $N$) first improves and then *degrades* true human preference, even as the RM score keeps rising. This curve is the canonical empirical demonstration of reward hacking ([Section 4.1](#41-theory)) and the justification for the KL anchor.

**Results**

The 6.7B human-feedback model's summaries were preferred by humans to the human-written reference summaries on TL;DR, and the 1.3B human-feedback model outperformed a supervised baseline ten times its size. The models also transferred to CNN/DailyMail news summarisation without any news-specific fine-tuning, producing summaries nearly matching the quality of the references there.

**Significance**

This is the template that InstructGPT and subsequent assistant training followed: every component of Module 5's loop ([Algorithm 5](#alg-rlhf-ppo)) appears here in its mature form. Its two lasting lessons are that labelling quality dominates labelling quantity, and that a learned reward model must never be optimised without a constraint, the lesson encoded in $\beta$.

#### 5.3.2. Ouyang et al. (2022): *Training Language Models to Follow Instructions with Human Feedback*

**Overview**

Ouyang et al. (2022) applied the Stiennon et al. (2020) recipe to general instruction following, producing the InstructGPT models from GPT-3 backbones of 1.3B, 6B, and 175B parameters. It is the canonical reference for the full SFT → RM → PPO pipeline and the direct precursor of ChatGPT.

**Problem**

Making language models bigger does not make them better at following a user's intent. Large models trained on next-token prediction produce outputs that are untruthful, toxic, or simply unhelpful, because the pre-training objective (predict web text) is misaligned with the deployment objective (follow the user's instruction helpfully and safely).

**Key Contributions**

The paper executed the three-stage pipeline at production scale and contributed several refinements. SFT used labeller-written demonstrations under the loss of [(21)](#eq-sft-loss). The reward model was trained on labeller *rankings* of between 4 and 9 sampled responses per prompt, with each ranking decomposed into pairwise comparisons for the loss of [(23)](#eq-rm-loss), an efficient design since one ranking of $K$ responses yields $\binom{K}{2}$ comparisons. The PPO stage used the per-token KL penalty of [(25)](#eq-token-kl). The paper also identified the **alignment tax**, performance regressions on public NLP benchmarks caused by RLHF, and mitigated it with the PPO-ptx variant, which mixes pre-training gradients into the PPO update.

**Results**

Outputs from the 1.3B InstructGPT model were preferred by labellers to outputs from the 175B GPT-3, a model over one hundred times larger. InstructGPT showed improved truthfulness on TruthfulQA and reduced toxic output when instructed to be respectful, though little improvement on bias benchmarks, and PPO-ptx largely eliminated the alignment tax. The models generalised to instruction styles rare in the fine-tuning data, such as non-English prompts and code questions, while still making simple mistakes on others.

**Significance**

InstructGPT demonstrated that the preference-learning pipeline scales from a single task (summarisation) to open-ended instruction following, and its publication preceded ChatGPT, which was trained with the same method. The hyperparameter shapes used in this module's projects (RM from the SFT checkpoint, small PPO learning rates, KL anchoring) descend directly from this paper, as does the practice of evaluating alignment by labeller preference rates rather than benchmark scores alone.

### 5.4. Hands-on Project: Full Reinforcement Learning from Human Feedback Training Loop

**Objective.** Run the full PPO loop using `trl.experimental.ppo.PPOTrainer`. In TRL v1, `PPOTrainer` lives in `trl.experimental.ppo` and handles generation, reward scoring, and the PPO update internally: the trainer is constructed with the models and run with a single `train()` call. When the policy is trained with LoRA (as here), only three of the four models of [Section 5.1](#51-theory) are loaded: passing `ref_model=None` makes the trainer recover the frozen reference $\pi_{\text{ref}}$ exactly by disabling the policy's adapters, since the base weights underneath them are the SFT model.

**Depends on [Section 3.2](#32-hands-on-project-supervised-fine-tuning-with-low-rank-adaptation) and [Section 4.3](#43-hands-on-project-reward-model-training).** The PPO models are all initialised from those two projects. The policy loads the merged SFT model `sft-model-merged` (produced by the merge step in [Section 4.3](#43-hands-on-project-reward-model-training)) and receives fresh LoRA adapters; the frozen reference is the same model with those adapters disabled. The reward model and the value model (critic) load the reward model trained in [Section 4.3](#43-hands-on-project-reward-model-training) — itself a LoRA adapter, so it is merged into its backbone first, mirroring the SFT merge step. Both earlier projects must therefore be completed first.

**Implementation outline.**

```python
import torch
from datasets import load_dataset
from peft import AutoPeftModelForSequenceClassification, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer

sft_path = "/Volumes/ML_Workspace/projects/rlhf-course/sft-model-merged"
rm_path  = "/Volumes/ML_Workspace/projects/rlhf-course/rm-model"

# The RM step saved a LoRA adapter; merge it into its backbone first
rm_merged_path = rm_path + "-merged"
AutoPeftModelForSequenceClassification.from_pretrained(
    rm_path, num_labels=1, dtype=torch.bfloat16
).merge_and_unload().save_pretrained(rm_merged_path)

# Left padding for batched generation
tokenizer = AutoTokenizer.from_pretrained(sft_path, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Three of the four models of Section 5.1: with a LoRA policy, the frozen
# reference is not loaded — disabling the adapters recovers pi_ref exactly
policy       = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.bfloat16)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_merged_path, num_labels=1, dtype=torch.bfloat16
)
value_model  = AutoModelForSequenceClassification.from_pretrained(
    rm_merged_path, num_labels=1, dtype=torch.bfloat16  # critic initialised from the RM
)
for model in (policy, reward_model, value_model):
    model.config.pad_token_id = tokenizer.pad_token_id

# Fresh LoRA adapters on the policy
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# PPOTrainer expects pre-tokenised prompts in an 'input_ids' column
def extract_prompt(text: str) -> str:
    marker = "\n\nAssistant:"
    return text[: text.rindex(marker) + len(marker)]

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset = dataset.map(
    lambda ex: tokenizer(extract_prompt(ex["chosen"])),
    remove_columns=dataset.column_names,
)
# Filter out (not truncate) over-long prompts: a truncated dialogue can lose
# the actual question, leaving the policy to optimise reward on nonsense
dataset = dataset.filter(lambda ex: len(ex["input_ids"]) <= 256)
dataset = dataset.train_test_split(test_size=100, seed=42)  # held-out eval prompts

config = PPOConfig(
    output_dir="/Volumes/ML_Workspace/projects/rlhf-course/ppo-output",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    # Rollout batch = train batch x grad accum x mini-batches = 4 x 4 x 1 = 16
    # episodes generated per update; each pass steps once per mini-batch
    num_mini_batches=1,
    # Memory-only knob: sequences per forward pass while generating and
    # scoring rollouts; no effect on the training result
    local_rollout_forward_batch_size=4,
    total_episodes=10_000,  # generation budget: 10_000 / 16 per rollout = 625 updates
    num_ppo_epochs=4,       # passes over each rollout batch (episodes reused 4x)
    response_length=128,    # generation budget (max new tokens)
    temperature=0.7,
    stop_token="eos",       # truncate responses at EOS before scoring
    missing_eos_penalty=1.0,   # subtracted from the RM score if no EOS emitted
    kl_coef=0.2,            # beta (fixed; the adaptive KL controller was removed)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # The PPO loop prints its metrics every update regardless of logging_steps;
    # the value still gates other consumers, e.g. callbacks
    logging_steps=10,
    # train() takes no arguments, so PPO runs cannot resume; checkpoints exist
    # only for manual adapter recovery after a crash
    save_steps=100,
    report_to="tensorboard",
    seed=42,
)

ppo_trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy,
    ref_model=None,   # pi_ref = the policy with its adapters disabled
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
)
ppo_trainer.train()
# Saves the policy's LoRA adapter
ppo_trainer.save_model("/Volumes/ML_Workspace/projects/rlhf-course/ppo-model")
```

**Implementation tips: the critic is trained in full.** In the snippet above, LoRA applies to the policy alone. `PPOTrainer` applies `peft_config` only to `model`, then wraps the policy and the critic in one module and builds the optimiser over both, so every parameter of `value_model`, backbone and scalar head alike, is updated. Of the four models of [Section 5.1](#51-theory), the reward model and the reference are frozen, the policy trains only its adapters, and the critic is the sole fully trained network. It therefore carries the largest optimiser state (full Adam moments for the whole backbone), which is worth remembering when budgeting memory.

TRL implements the critic this way because TRL v1's trainer reproduces the canonical OpenAI recipe: a separate value function, initialised from the reward model and trained in full, as in the papers of [Section 5.3](#53-suggested-reading). Note that those works predate LoRA and fully fine-tuned every component, so the full critic was the default of its era rather than a verdict against a low-rank critic. TRL grafts PEFT on only where it buys something structural, namely the policy, where the adapter also yields the free reference model via adapter disabling. A pragmatic argument for keeping the default is that value accuracy directly controls advantage noise and hence PPO stability, so maximal critic plasticity is the conservative choice. The critic is also discarded after training, so no artefact-size argument favours an adapter.

A LoRA critic is nonetheless possible. The trainer exposes no `value_peft_config`, but it accepts a critic that is already PEFT-wrapped, because both of its access paths (`getattr(value_model, value_model.base_model_prefix)` for the backbone and `value_model.score` for the head) resolve correctly through PEFT's attribute forwarding. Wrap the critic before constructing the trainer and change nothing else:

```python
from peft import get_peft_model

value_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_CLS",
    # Value targets are non-stationary (the policy keeps moving), so the
    # scalar head must remain trainable
    modules_to_save=["score"],
)
value_model = get_peft_model(value_model, value_lora_config)
```

Three caveats. This route relies on PEFT attribute forwarding, an implementation detail rather than a documented contract, so re-verify it after any TRL or PEFT upgrade. The saving is in gradients and optimiser state only, since the critic's weights and activations stay resident either way. Finally, `save_model` stores only the policy adapter, so the critic adapter is discarded unless `value_model` is saved manually, which is usually fine because the critic is scaffolding.

**What to observe.** Track `objective/kl`, `objective/scores`, `objective/rlhf_reward`, `policy/entropy_avg`, and `loss/value_avg` in TensorBoard. For entropy, prefer `policy/entropy_avg`, the closed-form per-token entropy, over the more prominently named `objective/entropy`. The latter sums sampled-token surprisal across the response window after filling every position past the end-of-sequence token with a sentinel log-probability of 1.0, so it subtracts a nat per padded position and measures response length as much as entropy, falling steeply and even turning negative as responses learn to terminate earlier. Since TRL v1 exposes no adaptive KL controller, $\beta$ is fixed via `kl_coef` and must be tuned manually: if `objective/kl` grows unboundedly, increase `kl_coef` per the remedies in [Section 5.2](#52-debugging-the-reinforcement-learning-from-human-feedback-training-loop). If `objective/scores` increases while `objective/kl` remains bounded, training is proceeding correctly.
