# 🤖 Reinforcement Learning Implementations

A comprehensive collection of reinforcement learning implementations - from fundamental algorithms to modern deep RL approaches. This repository provides detailed implementations of various RL methods with clear documentation and experimental results.

## 🎯 Implementations

### Classical Methods

#### Multi-Armed Bandits & Tabular Methods
- Comparison of exploration vs exploitation strategies
  - UCB (Upper Confidence Bound)
  - ε-greedy with optimistic initialization
  - Cheater and Random Agents
- Policy evaluation and iteration
- Dynamic programming methods
- Comprehensive analysis of convergence rates

#### SARSA & Q-Learning
- On-policy vs off-policy TD control
- Experience replay implementation
- Custom environments with configurable dynamics
- Comparative performance analysis

### Deep Reinforcement Learning 

#### DQN (Deep Q-Network)
- Experience replay buffer
- Target network for stability
- Prioritized experience replay
- DQN implementation
- Integration with gym environments

#### PPO (Proximal Policy Optimization)  
- Actor-critic architecture
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Parallel environment processing
- Support for discrete/continuous actions

#### RLHF (Reinforcement Learning from Human Feedback)
- Simple Function-Based Reward Model
- Policy iteration with preference learning
- Integration with language models
- KL-divergence penalty implementation

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run experiments:
```python 
# Multi-armed bandit example
python multi_armed_bandit_tabular/multiarmed_bandit_and_tabular_rl.py

# DQN example 
python SARSA_Q-Learning_DQN/SARSA_Q-Learning_DQN.py

# PPO example
python proximal_policy_optimization_ppo/ppo.py

# RLHF example
python rlhf/rlhf.py
```

## 📊 Project Structure

```
.
├── multi_armed_bandit_tabular/     # Classic RL algorithms
├── SARSA_Q-Learning_DQN/           # Value-based methods
├── proximal_policy_optimization_ppo/# Policy optimization 
├── rlhf/                           # Human feedback learning
├── utils.py                        # Shared utilities
├── plot_utils.py                   # Visualization helpers
└── requirements.txt                # Dependencies
```

## 🛠️ Tech Stack

- **PyTorch** - Deep learning & autograd
- **Gymnasium (OpenAI Gym)** - RL environments
- **NumPy/Einops** - Efficient computation
- **Weights & Biases** - Experiment tracking
- **Plotly** - Interactive visualizations

## 📑 Documentation

Each implementation includes:
- Theoretical background
- Implementation details
- Experimental results & analysis
- Hyperparameter tuning studies
- Training visualizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## 🔗 References

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Sutton & Barto
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
