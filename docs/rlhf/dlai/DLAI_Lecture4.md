# Tuning an LLM with RLHF on Vertex AI

> Created on: 20 April 2026
>
> Updated on: 20 April 2026

## 1. Overview

This chapter covers the practical steps needed to execute a Reinforcement Learning from Human Feedback (RLHF) tuning job on Google Cloud's Vertex AI platform. The process is encapsulated in a pre-built ML pipeline, which handles the multi-stage workflow — reward model training, reinforcement learning fine-tuning, and evaluation — as a single reproducible unit. The learner does not need to author or modify the pipeline itself; the task is to compile it and configure the job parameters correctly.

---

## 2. ML Pipelines and Why RLHF Uses Them

A machine learning pipeline is a portable, scalable workflow composed of sequential **components** (steps where code executes) and **artefacts** (outputs produced by those steps, e.g. datasets, trained models, metrics). Pipelines are a natural fit for RLHF because the process involves multiple interdependent stages:

1. A preference dataset is used to train a reward model.
2. The reward model is used alongside a prompt dataset in a reinforcement learning loop to fine-tune the base LLM.
3. An optional evaluation dataset triggers a batch inference job after tuning completes.

Vertex AI represents all of these stages visually, with components shown as blue-cube boxes and artefacts as yellow-triangle boxes. Notably, the pipeline contains a **Reward Model Trainer** component and a **Reinforcer** component, the latter being the RL loop that directly fine-tunes the base model.

---

## 3. Compiling the Pipeline

The RLHF pipeline is distributed as part of the `google-cloud-pipeline-components` library (currently in preview). Before it can be executed on Vertex AI, it must be **compiled** into a YAML file — a self-contained, human-readable description of every pipeline step and its inputs. The compiler comes from the `kfp` (Kubeflow Pipelines) library.

```python
# Install dependencies (only needed outside the course environment)
# !pip3 install google-cloud-pipeline-components kfp

# Import the pre-built RLHF pipeline (in preview as of writing)
from google_cloud_pipeline_components.preview.llm import rlhf_pipeline

# Import the KFP compiler
from kfp import compiler

# Define an output path for the compiled YAML
RLHF_PIPELINE_PKG_PATH = "rlhf_pipeline.yaml"

# Compile: this writes all pipeline metadata and step definitions to the YAML file
compiler.Compiler().compile(
    pipeline_func=rlhf_pipeline,
    package_path=RLHF_PIPELINE_PKG_PATH
)

# Inspect the first few lines of the generated file
# !head rlhf_pipeline.yaml
# To see the full file: !cat rlhf_pipeline.yaml
```

The resulting YAML file is auto-generated and should not be edited manually. It is the artefact that Vertex AI reads to understand what the pipeline should do.

---

## 4. Configuring the Pipeline Parameters

All job-specific configuration is passed as a Python dictionary called `parameter_values`. The parameters fall into four logical groups.

### 4.1. Dataset Paths

Three datasets are required, all stored in Google Cloud Storage (paths prefixed with `gs://`). They must all be in JSON Lines (`.jsonl`) format and reside in the same GCS bucket.

```python
parameter_values = {
    # Human preference comparisons, used to train the reward model
    "preference_dataset": "gs://vertex-ai/generative-ai/rlhf/text_small/"
                          "summarize_from_feedback_tfds/comparisons/train/*.jsonl",

    # Reddit posts used as prompts during the RL fine-tuning loop
    "prompt_dataset": "gs://vertex-ai/generative-ai/rlhf/text_small/"
                      "reddit_tfds/train/*.jsonl",

    # Validation prompts; triggers a batch inference job post-tuning
    "eval_dataset": "gs://vertex-ai/generative-ai/rlhf/text_small/"
                    "reddit_tfds/val/*.jsonl",
    ...
}
```

### 4.2. Base Model Selection

The `large_model_reference` parameter specifies which foundational model to fine-tune. Supported options include `"llama-2-7b"`, `"text-bison@001"`, and T5x family models.

```python
"large_model_reference": "llama-2-7b",
```

### 4.3. Training Steps

The pipeline accepts the number of **training steps** (not epochs) for each of the two training phases. The conversion from epochs to steps uses the formula:

$$stepsPerEpoch = \left\lceil \frac{datasetSize}{batchSize} \right\rceil$$

$$trainSteps = stepsPerEpoch \times numEpochs$$

The batch size is fixed at 64 for this pipeline. The recommended epoch ranges are 20–30 for the reward model and 10–20 for the RL loop.

**Reward model training steps:**

```python
import math

PREF_DATASET_SIZE = 3000  # Size of the preference dataset
BATCH_SIZE = 64           # Fixed by the pipeline; cannot be changed

REWARD_STEPS_PER_EPOCH = math.ceil(PREF_DATASET_SIZE / BATCH_SIZE)  # = 47
REWARD_NUM_EPOCHS = 30

reward_model_train_steps = REWARD_STEPS_PER_EPOCH * REWARD_NUM_EPOCHS  # = 1410
```

**RL fine-tuning steps:**

```python
PROMPT_DATASET_SIZE = 2000  # Size of the prompt dataset
BATCH_SIZE = 64

RL_STEPS_PER_EPOCH = math.ceil(PROMPT_DATASET_SIZE / BATCH_SIZE)  # = 32
RL_NUM_EPOCHS = 10

reinforcement_learning_train_steps = RL_STEPS_PER_EPOCH * RL_NUM_EPOCHS  # = 320
```

> **Tip:** It is good practice to run the pipeline on a small subset of the data first (e.g. 2,000–3,000 examples) to verify that the job executes without errors before committing to a full run, which can take over a day.

### 4.4. Learning Rate Multipliers and KL Coefficient

These are more advanced parameters that can be left at their defaults initially and tuned later.

**Learning rate multipliers** (`reward_model_learning_rate_multiplier`, `reinforcement_learning_rate_multiplier`): The absolute learning rate is fixed by the pipeline to match the rate used during the base model's original training (which may not be publicly known). These multipliers scale that fixed rate up or down — values greater than 1 increase the magnitude of gradient updates, values less than 1 decrease it. Both default to `1.0`.

**KL coefficient** (`kl_coeff`): This is a regularisation term that guards against **reward hacking** — the tendency for the policy model to exploit the reward signal in unintended ways. For example, the reward model might give high scores to completions filled with positive-sounding words like 'excellent' or 'fantastic', even when the response is otherwise nonsensical. The KL coefficient penalises the tuned model for diverging too far from the original model's output distribution. Setting it to `0` removes all penalty; larger values impose stronger regularisation. The default is `0.1`.

### 4.5. Task Instruction

The `instruction` parameter prepends a task description to every prompt in both the preference and prompt datasets. It should only be set if the instruction is **not** already embedded in the dataset prompts themselves.

```python
"instruction": "Summarise in less than 50 words"
```

---

## 5. Complete Parameter Dictionary

Below is the full `parameter_values` dictionary used for the small-dataset demonstration run:

```python
parameter_values = {
    "preference_dataset": "gs://vertex-ai/generative-ai/rlhf/text_small/"
                          "summarize_from_feedback_tfds/comparisons/train/*.jsonl",
    "prompt_dataset":     "gs://vertex-ai/generative-ai/rlhf/text_small/"
                          "reddit_tfds/train/*.jsonl",
    "eval_dataset":       "gs://vertex-ai/generative-ai/rlhf/text_small/"
                          "reddit_tfds/val/*.jsonl",
    "large_model_reference":                    "llama-2-7b",
    "reward_model_train_steps":                 1410,
    "reinforcement_learning_train_steps":       320,
    "reward_model_learning_rate_multiplier":    1.0,
    "reinforcement_learning_rate_multiplier":   1.0,
    "kl_coeff":                                 0.1,
    "instruction":                              "Summarise in less than 50 words",
}
```

For a full-dataset run (the configuration used by the course team to produce the evaluation results in the next lesson), the step counts are substantially higher and the RL learning rate multiplier is reduced:

```python
# Full dataset configuration (takes ~24 hours on TPUs/GPUs)
parameter_values = {
    "preference_dataset": "gs://vertex-ai/generative-ai/rlhf/text/"
                          "summarize_from_feedback_tfds/comparisons/train/*.jsonl",
    "prompt_dataset":     "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/train/*.jsonl",
    "eval_dataset":       "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/val/*.jsonl",
    "large_model_reference":                    "llama-2-7b",
    "reward_model_train_steps":                 10000,
    "reinforcement_learning_train_steps":       10000,
    "reward_model_learning_rate_multiplier":    1.0,
    "reinforcement_learning_rate_multiplier":   0.2,  # Reduced to prevent reward hacking
    "kl_coeff":                                 0.1,
    "instruction":                              "Summarise in less than 50 words",
}
```

---

## 6. Creating and Submitting the Pipeline Job

Once the parameters are defined and the YAML file has been compiled, the job is submitted to Vertex AI using the `google-cloud-aiplatform` SDK.

```python
# Authenticate and retrieve project credentials
from utils import authenticate
credentials, PROJECT_ID, STAGING_BUCKET = authenticate()

# RLHF pipeline is available in this specific region only (as of writing)
REGION = "europe-west4"

import google.cloud.aiplatform as aiplatform

# Initialise the SDK with project and regional settings
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    credentials=credentials
)

# Define the pipeline job
job = aiplatform.PipelineJob(
    display_name="tutorial-rlhf-tuning",
    pipeline_root=STAGING_BUCKET,        # GCS location for intermediate artefacts
    template_path=RLHF_PIPELINE_PKG_PATH, # Path to the compiled YAML file
    parameter_values=parameter_values    # Job-specific configuration
)

# Submit the job (runs remotely on Vertex AI, not locally)
job.run()
```

The job executes on Google Cloud infrastructure — not locally — and requires significant compute resources (i.e. TPUs or GPUs). A full-scale run takes well over 24 hours. For initial experimentation, always begin with a small subset of data.

---

## 7. Key Concepts Summary

| Concept | Description |
|---|---|
| **Pipeline** | A portable, multi-step ML workflow compiled to YAML and executed on Vertex AI. |
| **Reward model** | Trained on preference data to score completions; trained for 20–30 epochs. |
| **Reinforcer** | RL loop component that fine-tunes the base LLM; trained for 10–20 epochs. |
| **Batch size** | Fixed at 64; used to convert epochs to training steps. |
| **KL coefficient** | Regularisation term preventing the policy from diverging too far from the original model. |
| **Reward hacking** | When the policy exploits the reward signal in unintended ways (e.g. generating hollow but highly-scored text). |
| **GCS** | Google Cloud Storage; all datasets and artefacts must reside here in `.jsonl` format. |
| **`large_model_reference`** | Specifies which foundational model to tune (e.g. `llama-2-7b`, `text-bison@001`). |
