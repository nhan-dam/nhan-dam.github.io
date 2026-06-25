# 1. Projects

Hands-on implementations accompanying the RLHF curriculum. Each write-up explains the design decisions and links directly to the source code.

---

| Project | Description | Status |
|---------|-------------|--------|
| [Tabular Q-Learning on Blackjack](report_qlearning_blackjack.md) | Tabular Q-learning on `Blackjack-v1`, with an $\varepsilon$-decay sweep across learning rates and an analysis of the learnt policy against basic strategy. | Completed |
| [Tabular Q-Learning on CartPole](report_qlearning_cartpole.md) | Tabular Q-learning on `CartPole-v1`, discretising the continuous observation space into bins and sweeping learning rate and decay strategy. | Completed |
| [Deep Q-Network on CartPole](report_dqn_cartpole.md) | From-scratch DQN on `CartPole-v1` with experience replay and a target network, a 30-config sweep, and appendices on Double DQN and parallelising the sweep. | Completed |
| [PPO on LunarLander](report_ppo_lunarlander.md) | From-scratch PPO on `LunarLanderContinuous-v3`, with a 12-config sweep and a TensorBoard vs Weights and Biases observability comparison. | Completed |
