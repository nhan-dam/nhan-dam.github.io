# 1. Projects

Hands-on implementations accompanying the RLHF curriculum. Each write-up explains the design decisions and links directly to the source code.

---

| Project | Description | Status |
|---------|-------------|--------|
| [Tabular Q-Learning on Blackjack](report_qlearning_blackjack.md) | Tabular Q-learning on `Blackjack-v1`, with an $\varepsilon$-decay sweep across learning rates and an analysis of the learnt policy against basic strategy. | Completed |
| [Tabular Q-Learning on CartPole](report_qlearning_cartpole.md) | Tabular Q-learning on `CartPole-v1`, discretising the continuous observation space into bins and sweeping learning rate and decay strategy. | Completed |
| [Deep Q-Network on CartPole](report_dqn_cartpole.md) | From-scratch DQN on `CartPole-v1` with experience replay and a target network, a 30-config sweep, and appendices on Double DQN and parallelising the sweep. | Completed |
| [PPO on LunarLander](report_ppo_lunarlander.md) | From-scratch PPO on `LunarLanderContinuous-v3`, with a 12-config sweep and a TensorBoard vs Weights and Biases observability comparison. | Completed |
| [Supervised Fine-Tuning with LoRA on Dolly-15k](report_sft_lora_dolly.md) | LoRA fine-tuning of Qwen2.5-0.5B on `databricks/databricks-dolly-15k` to produce the reference policy, with a root-caused unified-memory growth pattern on Apple Silicon. | Completed |
| [Reward Model Training on Anthropic HH-RLHF](report_reward_model_hh.md) | A scalar-output reward model trained on `Anthropic/hh-rlhf` pairwise preferences, with adversarial probes exposing its blind spots and a higher-capacity run showing the ceiling is data, not adapter size. | Completed |
| [Full RLHF Loop with PPO](report_ppo_rlhf_loop.md) | PPO of the SFT policy against the trained reward model, with a paired held-out evaluation and an audit of how post-EOS padding corrupts several logged metrics. | Completed |
