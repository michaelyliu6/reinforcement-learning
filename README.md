# Reinforcement Learning Implementations

A comprehensive implementation of fundamental and advanced reinforcement learning algorithms, following the theoretical foundations from Sutton & Barto's "Reinforcement Learning: An Introduction" and incorporating modern deep learning practices.

## 🚀 Key Implementations

### 1. Multi-Armed Bandit and Tabular RL
- Custom OpenAI Gymnasium environment implementation
- Multiple agent strategies:
  - Epsilon-Greedy with reward averaging
  - Optimistic initialization
  - Upper Confidence Bound (UCB)
  - Random agent baseline
- Visualization of reward distributions and agent performance
- Implementation of policy improvement methods

### 2. SARSA, Q-Learning, and Deep Q-Network (DQN)
- Implementation of fundamental value-based RL algorithms:
  - SARSA (State-Action-Reward-State-Action)
  - Q-Learning with various exploration strategies
  - Deep Q-Network with experience replay and target networks
- Custom environment implementations:
  - Discrete environment wrapper for OpenAI Gymnasium
  - Norvig's Grid World
  - CartPole environment integration
- Advanced features:
  - Experience replay buffer
  - Target network for stability
  - Epsilon decay for exploration
  - Customizable hyperparameters

### 3. Proximal Policy Optimization (PPO)
- State-of-the-art policy gradient implementation
- Support for multiple environment types:
  - Classic control (CartPole)
  - Atari games
  - MuJoCo continuous control
- Advanced features:
  - Parallel environment processing
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Value function clipping
  - Entropy bonus for exploration
- Integration with Weights & Biases for experiment tracking

## 🛠️ Technologies & Skills Demonstrated

- **Python** - Advanced object-oriented programming and software design patterns
- **PyTorch** - Deep learning model implementation with modern best practices
- **OpenAI Gymnasium** 
  - Custom environment development
  - Environment wrappers and preprocessing
  - Support for discrete and continuous action spaces
- **NumPy** - Efficient numerical computations and array operations
- **Weights & Biases** - Professional experiment tracking and hyperparameter optimization
- **Data Visualization**
  - Plotly for interactive visualizations
  - Custom plotting utilities for RL metrics
- **Software Engineering Best Practices**
  - Modular code design
  - Type hints and documentation
  - Configurable hyperparameters
  - Version control with Git

## 📚 Theoretical Foundations

The implementations are based on foundational RL literature and papers:
- Sutton and Barto's "Reinforcement Learning: An Introduction"
- Original Q-Learning paper
- PPO paper and implementation details
- Multi-Armed Bandit theory

## 🔧 Key Features

- **Environment Development**
  - Custom environment implementations
  - Environment wrappers for preprocessing
  - Support for various observation and action spaces
- **Algorithm Implementation**
  - Progression from basic to advanced RL concepts
  - Multiple agent architectures and strategies
  - Comprehensive hyperparameter configuration
- **Experiment Management**
  - Integration with Weights & Biases
  - Custom logging and visualization
  - Video recording of agent behavior
- **Code Quality**
  - Type hints and comprehensive documentation
  - Modular and extensible design
  - Clear separation of concerns

## 📊 Project Structure

```
.
├── multiarmed_bandit_and_tabular_rl.ipynb  # Basic RL concepts and implementations
├── SARSA_Q-Learning_DQN.ipynb              # Value-based methods and deep RL
├── ppo.ipynb                               # Advanced policy gradient methods
├── utils.py                                # Shared utilities and environment wrappers
├── plotly_utils.py                         # Visualization utilities
├── wandb/                                  # Experiment tracking data
└── videos/                                 # Recorded agent demonstrations
```

## 🔗 Dependencies

- Python 3.x
- PyTorch
- OpenAI Gymnasium
- NumPy
- Weights & Biases
- Plotly
- tqdm
- einops 