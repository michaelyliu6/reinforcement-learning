# 🤖 Reinforcement Learning Implementations

Playing around with reinforcement learning algorithms - from the classics to modern deep RL. This repo walks through my implementations of various RL approaches, with detailed documentation and experimental results.

## 🎯 Implementations

### 🎲 Classical Methods
- **Multi-Armed Bandits**
  - Exploration vs exploitation trade-off analysis
  - Implementation of UCB, ε-greedy, and optimistic initialization
  - Custom environments for both stationary and non-stationary reward distributions
  - Empirical analysis of convergence rates

- **Tabular Methods**
  - Q-Learning and SARSA with convergence guarantees
  - Dynamic programming for policy evaluation and improvement
  - Implementation of various exploration strategies
  - Custom grid world environments with configurable dynamics

### 🧠 Deep Reinforcement Learning
- **Deep Q-Network (DQN)**
  - Experience replay for sample efficiency
  - Target networks for training stability
  - Prioritized replay buffer implementation
  - Comprehensive hyperparameter tuning analysis
  - Integration with classic control environments

- **Proximal Policy Optimization (PPO)**
  - Parallel environment processing for efficient training
  - Clipped surrogate objective implementation
  - Generalized Advantage Estimation (GAE)
  - Adaptive learning rate scheduling
  - Support for both discrete and continuous action spaces

## 🛠️ Technical Implementation

### 🏗️ Architecture Design
- Modular implementation separating agents, environments, and training loops
- Type-safe codebase with comprehensive static typing
- Vectorized operations for computational efficiency
- Custom environment wrappers extending OpenAI Gymnasium

### ⚡ Performance Optimization
- Parallel environment execution for PPO
- Efficient replay buffer implementation using NumPy
- Vectorized advantage calculation
- Optimized policy network architectures

### 📊 Experiment Tracking
- Integration with Weights & Biases for:
  - Hyperparameter optimization
  - Training metrics visualization
  - Model performance analysis
  - Learning curves and reward tracking

## 📁 Project Structure

```
.
├── multi_armed_bandit_tabular/              # Foundational RL concepts
│   └── multiarmed_bandit_and_tabular_rl.ipynb
│       - Multi-armed bandit implementations
│       - Tabular Q-learning and SARSA
│       - Dynamic programming methods
│
├── sarsa_q-learning_dqn/                    # Deep Q-Learning implementation
│   └── SARSA_Q-Learning_DQN.ipynb
│       - DQN with experience replay
│       - Target network implementation
│       - Classic control environment integration
│
├── proximal_policy_optimization_ppo/        # Advanced policy optimization
│   └── ppo.ipynb
│       - PPO implementation with parallel training
│       - GAE computation
│       - Policy and value network architectures
│
├── plot_utils.py                           # Plotting utilities
│   - Visualization functions
│   - Training curve plotting
│
└── utils.py                                # Shared utilities and environments
    - Custom environment implementations
    - Training helper functions
    - Common utility functions
```

## 🚀 Tech Stack

Built with modern ML tools:
- **🔥 PyTorch**: Neural networks and autograd
- **🎮 OpenAI Gymnasium**: Environment simulation
- **🔢 NumPy/Einops**: Efficient computation
- **📈 Weights & Biases**: Experiment tracking
- **✨ Type Hints**: Code reliability
- **📊 Plotly**: Interactive visualizations

## 🏃‍♂️ Getting Started

```bash
pip install -r requirements.txt
```

Each notebook comes packed with:
- Theoretical background and math
- Implementation details with docs
- Experimental results and analysis
- Hyperparameter tuning studies
- Learning visualizations

## 💡 Implementation Notes

The code is built with:
- Clean, readable implementations following theory
- Detailed documentation explaining concepts
- Efficient vectorized operations
- Reproducible results (fixed seeds)
- Flexible, extensible architectures

## 🔮 What's Next?

### 🚧 In Progress
- **RLHF (Reinforcement Learning from Human Feedback)**
  - Reward model training pipeline
  - Human feedback collection interface
  - Integration with existing PPO implementation

### 📝 On the Roadmap
- **GPTO (Generalized Policy Trust Optimization)**
  - Enhanced trust region optimization
  - Multi-task policy adaptation
  - Improved sample efficiency mechanisms

### 💭 Future Ideas
- Distributed training for large-scale environments
- MuJoCo integration for robotics
- Offline RL algorithms
- Multi-agent RL extensions