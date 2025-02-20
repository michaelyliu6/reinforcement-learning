#!/usr/bin/env python
# coding: utf-8

# # References
# 
# In this notebook, we will be implementing the Q Learning described in Chapter 6 of [Reinforcement Learning: An Introduction by Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) and the original [Q-learning paper](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf).

# # Environment Setup

# In[14]:


import os
import sys
sys.path.append("..")
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
import utils


import gymnasium as gym
import numpy as np
import torch as t
import wandb
from gymnasium.spaces import Box, Discrete
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from tqdm import tqdm, trange
from plot_utils import cliffwalk_imshow, line, plot_cartpole_obs_and_dones
from utils import Environment, Toy, Norvig, find_optimal_policy, make_env, set_global_seeds

warnings.filterwarnings("ignore")

Arr = np.ndarray

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# ## Setting up the environment
# 
# In this section, we'll set up our environment using OpenAI Gym's interface. The Gym framework provides a standardized way to interact with reinforcement learning environments.
# 
# The key components of a Gym environment are:
# - The `step` method: Takes an action and returns (observation, reward, done, info)
#   - observation: The agent's new state/observation after taking the action
#   - reward: Numerical reward received for the action
#   - done: Boolean indicating if episode has terminated
#   - info: Additional information dictionary for debugging
# - The `reset` method: Resets environment to initial state
# - Observation space: Defines valid observations (can be discrete or continuous)
# - Action space: Defines valid actions (can be discrete or continuous) 
# - The `render` method: Optional visualization of the environment
# 
# Below we'll implement a discrete environment by wrapping it in the Gym interface.
# This standardized interface allows us to:
# 1. Have a consistent API for environment interactions
# 2. Make the code more modular and reusable
# 3. Easily swap between different environment implementations
# 
# 

# In[15]:


ObsType: TypeAlias = int | np.ndarray
ActType: TypeAlias = int


class DiscreteEnviroGym(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    """
    A discrete environment class for reinforcement learning, compatible with OpenAI Gym.

    This class represents a discrete environment where actions and observations are discrete.
    It is designed to interface with a provided `Environment` object which defines the
    underlying dynamics, states, and actions.

    Attributes:
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Discrete): The space of possible observations (states).
        env (Environment): The underlying environment with its own dynamics and properties.
    """

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        """
        Execute an action and return the new state, reward, done flag, and additional info.
        The behaviour of this function depends primarily on the dynamics of the underlying
        environment.
        """
        states, rewards, probs = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        new_state, reward = states[idx], rewards[idx]
        self.pos = new_state
        terminated = self.pos in self.env.terminal
        truncated = False
        info = {"env": self.env}
        return new_state, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options=None) -> tuple[ObsType, dict]:
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.pos = self.env.start
        return self.pos, {}

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"


# In[16]:


gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
    id="ToyGym-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=3,
    nondeterministic=False,
    kwargs={"env": Toy()},
)


# ## Agent Class
# 
# The Agent class below provides a foundation for implementing reinforcement learning agents. It uses two key dataclasses:
# - Experience: Stores individual experiences from episodes, containing the state $(s_t)$, action $(a_t)$, reward $(r_{t+1})$, 
#   next state $(s_{t+1})$, and next action $(a_{t+1})$. This structured way of storing experience is essential for learning.
# - AgentConfig: A convenient way to package hyperparameters like epsilon (exploration rate), learning rate, and optimism 
#   parameter that control the agent's learning behavior.
# 
# 
# The base Agent class provides core functionality like environment interaction and random number generation, while leaving the specific learning algorithms to be implemented by subclasses. This modular design allows us to easily experiment with different learning approaches while reusing common infrastructure.
# 

# In[17]:


@dataclass
class Experience:
    """
    A class for storing one piece of experience during an episode run.
    """

    obs: ObsType
    act: ActType
    reward: float
    new_obs: ObsType
    new_act: ActType | None = None


@dataclass
class AgentConfig:
    """Hyperparameters for agents"""

    epsilon: float = 0.1
    lr: float = 0.05
    optimism: float = 0


defaultConfig = AgentConfig()


class Agent:
    """Base class for agents interacting with an environment (you do not need to add any implementation here)"""

    rng: np.random.Generator

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        self.env = env
        self.reset(seed)
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__

    def get_action(self, obs: ObsType) -> ActType:
        raise NotImplementedError()

    def observe(self, exp: Experience) -> None:
        """
        Agent observes experience, and updates model as appropriate.
        Implementation depends on type of agent.
        """
        pass

    def reset(self, seed: int) -> tuple[ObsType, dict]:
        self.rng = np.random.default_rng(seed)
        return None, {}

    def run_episode(self, seed) -> list[int]:
        """
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed : Seed for the random number generator
        Outputs:
            The rewards obtained during the episode
        """
        rewards = []
        obs, info = self.env.reset(seed=seed)
        self.reset(seed=seed)
        done = False
        while not done:
            act = self.get_action(obs)
            new_obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated
            exp = Experience(obs, act, reward, new_obs)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
        return rewards

    def train(self, n_runs=500):
        """
        Run a batch of episodes, and return the total reward obtained per episode
        Inputs:
            n_runs : The number of episodes to simulate
        Outputs:
            The discounted sum of rewards obtained for each episode
        """
        all_rewards = []
        for seed in trange(n_runs):
            rewards = self.run_episode(seed)
            all_rewards.append(utils.sum_rewards(rewards, self.gamma))
        return all_rewards


# ## Random Agent

# In[18]:


class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)


# # Cheater Agent
# 
# 
# The Cheater agent is a special type of agent that has access to the optimal policy of the environment beforehand. It uses the `find_optimal_policy()` function to calculate the best action for each state when initialized. This agent serves as an upper baseline - it shows us the best possible performance we could achieve, since it already knows the perfect actions to take.
# 
# In the code below, we'll create a Cheater agent and compare it against a Random agent on the ToyGym environment. The Cheater agent should significantly outperform the Random agent since it has perfect knowledge of what to do.
# 

# In[19]:


class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        self.pi_opt = find_optimal_policy(self.env.unwrapped.env, self.gamma)

    def get_action(self, obs):
        return self.pi_opt[obs]


env_toy = gym.make("ToyGym-v0")
agents_toy: list[Agent] = [Cheater(env_toy), Random(env_toy)]
returns_dict = {}
for agent in agents_toy:
    returns = agent.train(n_runs=100)
    returns_dict[agent.name] = utils.cummean(returns)

line(
    list(returns_dict.values()),
    names=list(returns_dict.keys()),
    title=f"Avg. reward on {env_toy.spec.name}",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    template="simple_white",
    width=700,
    height=400,
)


# # Tabular Methods
# 
# In the next sections, we'll implement two key tabular reinforcement learning methods: SARSA and Q-Learning. Tabular methods work by creating lookup tables that map state-action pairs to Q-values, essentially memorizing the value of each action in each state.
# 
# While these methods can be limited by their inability to generalize across similar states/actions, they serve as important building blocks for understanding more sophisticated approaches. They work well for environments with small, discrete state and action spaces like our toy examples.
# 
# Both SARSA and Q-Learning will inherit from the EpsilonGreedy class, which implements the epsilon-greedy exploration strategy. This strategy balances exploration and exploitation by:
# - Taking a random action with probability epsilon (exploration)
# - Taking the action with highest Q-value with probability 1-epsilon (exploitation)
# 
# The EpsilonGreedy class maintains a Q-table initialized with optimistic values to encourage early exploration. Both SARSA and Q-Learning will use this table to store and update their action-value estimates, but they differ in how they update these values:
# - SARSA (State-Action-Reward-State-Action) is an on-policy method that learns from the actual actions taken by the agent
# - Q-Learning is an off-policy method that learns about the optimal policy regardless of the actions actually taken during training
# 

# In[20]:


class EpsilonGreedy(Agent):
    """
    A class for SARSA and Q-Learning to inherit from.
    """

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        super().__init__(env, config, gamma, seed)
        self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism

    def get_action(self, obs: ObsType) -> ActType:
        """
        Selects an action using epsilon-greedy with respect to Q-value estimates
        """
        if self.rng.random() < self.config.epsilon:
            return self.rng.integers(0, self.num_actions)
        else:
            return self.Q[obs].argmax()


# ## SARSA: On-Policy TD Control
# 
# SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference control algorithm that learns directly from experience. Unlike the previous agents, SARSA doesn't need access to the environment's dynamics - it learns by interacting with the environment and updating its Q-value estimates based on the rewards it receives.
# 
# The key idea is that SARSA tries to estimate the optimal Q-value function $Q*(s,a)$ which represents the expected future rewards when taking action $a$ in state $s$ and following the optimal policy thereafter. It does this by using the temporal difference (TD) error between its current estimate and the actual reward received plus discounted future value.
# 
# For each experience tuple $(s_t, a_t, r_t+1, s_t+1, a_t+1)$, SARSA updates its Q-value estimates using:
# $Q(s_t,a_t) ← Q(s_t,a_t) + η[r_t+1 + γQ(s_t+1,a_t+1) - Q(s_t,a_t)]$
# 
# Where:
# - η is the learning rate
# - γ is the discount factor
# - The term in brackets is the TD error
# 
# SARSA is "on-policy" because it learns about the policy it's currently following, including exploration. This means it takes into account that the agent will sometimes take exploratory actions rather than just the optimal ones. 
# 
# 

# In[21]:


class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        s_t, a_t, r_t_1, s_t_1, a_t_1 = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act
        self.Q[s_t, a_t] += self.config.lr * (r_t_1 + self.gamma * self.Q[s_t_1, a_t_1] - self.Q[s_t, a_t])

    def run_episode(self, seed) -> list[int]:
        rewards = []

        self.reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        act = self.get_action(obs)
        self.reset(seed=seed)
        done = False
        while not done:
            new_obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated
            new_act = self.get_action(new_obs)
            exp = Experience(obs, act, reward, new_obs, new_act)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
            act = new_act
        return rewards


# ## Q-Learning: Off-Policy TD Control
# 
# Q-Learning is a key modification to SARSA that allows us to learn the optimal Q-function Q* directly, regardless of the policy being followed. The key difference is in how the TD error is calculated:
# 
# While SARSA uses the actual next action taken $a_{t+1}$ to compute the TD target:
# $$
# Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \eta \left( r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t) \right)
# $$
# 
# 
# Q-Learning instead uses the maximum Q-value over all possible next actions:
# $$
# Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \eta \left( r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t) \right)
# $$
# 
# This means Q-Learning is **"off-policy"** - it learns about the optimal policy even while following an exploratory policy. The update tries to make $Q(s,a)$ more consistent with the best possible action in the next state, rather than the action that was actually taken.
# 
# This is powerful because:
# 1. We can learn optimal behavior while following a suboptimal exploratory policy
# 2. We directly estimate Q* rather than Q^π for some policy π
# 3. We can reuse old experience data since the learning doesn't depend on the policy that generated it
# 
# However, this off-policy approach has some significant drawbacks:
# 1. It can be less stable than SARSA because it learns about actions that weren't actually taken, leading to overoptimistic value estimates
# 2. In stochastic environments, Q-Learning can be overly aggressive in pursuing high rewards while ignoring risks, since it assumes optimal future actions
# 3. The mismatch between the behavior policy (exploratory) and target policy (greedy) can lead to slower convergence in practice
# 

# In[22]:


class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        s_t, a_t, r_t_1, s_t_1 = exp.obs, exp.act, exp.reward, exp.new_obs
        self.Q[s_t, a_t] += self.config.lr * (r_t_1 + self.gamma * np.max(self.Q[s_t_1]) - self.Q[s_t, a_t])


# ## NorvigGrid
# 
# The NorvigGrid environment is a grid-world problem that tests how agents handle risk-reward tradeoffs:
# - The agent starts in the bottom-left corner and needs to reach the goal in the bottom-right corner
# - The bottom edge of the grid contains a "cliff" - stepping into it results in a large negative reward (-100) and sends the agent back to start
# - Each non-cliff step gives a small negative reward (-1) to encourage finding the shortest path
# - The agent can move in 4 directions: up, right, down, left
# - The optimal path is to move along the bottom edge next to the cliff, but this is risky due to the exploration policy
# - A safer but longer path is to move up and around the cliff area
# 
# ![NorvigGrid](ref/1_oo2jt_HMxVsweRRqT1efcQ.webp)
# 

# In[23]:


n_runs = 1000
gamma = 0.99
seed = 1
env_norvig = gym.make("NorvigGrid-v0")
config_norvig = AgentConfig(epsilon=0.1, lr=0.1, optimism=0.5)
args_norvig = (env_norvig, config_norvig, gamma, seed)
agents_norvig: list[Agent] = [
    Cheater(*args_norvig),
    QLearning(*args_norvig),
    SARSA(*args_norvig),
    Random(*args_norvig),
]
returns_dict = {}
for agent in agents_norvig:
    returns = agent.train(n_runs)
    returns_dict[agent.name] = utils.cummean(returns)

line(
    list(returns_dict.values()),
    names=list(returns_dict.keys()),
    title=f"Avg. reward on {env_norvig.spec.name}",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    template="simple_white",
    width=700,
    height=400,
)


# Which seems to work better, SARSA or Q-Learning? 
# - From the results collected in the NorvigGrid environment, SARSA and Q-Learning perform quite similarly, with both algorithms converging to comparable average returns. The plot shows both algorithms learning effective policies, though Q-Learning appears to have a slight edge in terms of final performance. This is because Q-Learning tends to learn the optimal path along the bottom edge since it considers the maximum possible future reward regardless of the exploration policy. In contrast, SARSA takes into account that the exploration policy might cause occasional falls into hazardous areas, so it learns a more conservative path that stays further from dangerous regions. While this makes SARSA's path slightly longer, it is safer given the possibility of random exploratory actions.
# 

# ## CliffWalking
# 
# The CliffWalking environment is a classic grid-world problem that tests how agents handle risk-reward tradeoffs:
# 
# - The agent starts in the bottom-left corner and needs to reach the goal in the bottom-right corner
# - The bottom edge of the grid contains a "cliff" - stepping into it results in a large negative reward (-100) and sends the agent back to start
# - Each non-cliff step gives a small negative reward (-1) to encourage finding the shortest path
# - The agent can move in 4 directions: up, right, down, left
# - The optimal path is to move along the bottom edge next to the cliff, but this is risky due to the exploration policy
# - A safer but longer path is to move up and around the cliff area
# 
# This environment is particularly useful for comparing SARSA and Q-Learning since they handle the exploration-exploitation tradeoff differently.
# 

# In[24]:


gamma = .99
seed = 0

config_cliff = AgentConfig(epsilon=0.2, lr=0.1, optimism=0)
env = gym.make("CliffWalking-v0")
n_runs = 2500
args_cliff = (env, config_cliff, gamma, seed)

returns_list = []
name_list = []
agents = [QLearning(*args_cliff), SARSA(*args_cliff)]

for agent in agents:
    returns = agent.train(n_runs)[1:]
    returns_list.append(utils.cummean(returns))
    name_list.append(agent.name)
    V = agent.Q.max(axis=-1).reshape(4, 12)
    pi = agent.Q.argmax(axis=-1).reshape(4, 12)
    cliffwalk_imshow(V, pi, title=f"CliffWalking: {agent.name} Agent", width=800, height=400)

line(
    returns_list,
    names=name_list,
    template="simple_white",
    title="Q-Learning vs SARSA on CliffWalking-v0",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    width=700,
    height=400,
)


# The results above compare Q-Learning and SARSA algorithms on the CliffWalking environment, highlighting their key differences:
# 
# 1. Algorithm Behavior Differences:
#    - Q-Learning is **"off-policy"** and learns the optimal policy regardless of the exploration strategy
#    - SARSA is **"on-policy"** and learns a policy that accounts for the exploration strategy
#    - This fundamental difference shows in their performance: SARSA learns a safer path further from the cliff
#      because it accounts for the possibility of random exploratory moves that could lead to falling
#    - Q-Learning learns a riskier path closer to the cliff because it assumes optimal future actions
# 
# 2. Performance Comparison:
#    - The graph shows SARSA achieving higher average rewards than Q-Learning
#    - This is because SARSA's safer policy leads to fewer cliff falls during both training and evaluation
#    - Q-Learning's theoretically optimal but riskier path results in more frequent falls when combined
#      with the ε-greedy exploration strategy
# 
# 3. Value Function & Policy Visualization:
#    - In the heatmaps, Q-Learning shows higher value estimates near the cliff (brighter colors)
#    - SARSA's value estimates are more conservative near dangerous areas
#    - The policy arrows (pi) show Q-Learning choosing a more direct but dangerous path
#    - SARSA's policy takes a longer but safer route to avoid catastrophic failures
# 
# 
# 
# 

# # Deep Q-Learning (DQN)
# 
# Deep Q-Learning (DQN) is a groundbreaking algorithm that combines Q-Learning with deep neural networks to handle complex state spaces. Unlike traditional Q-Learning which uses a table to store Q-values, DQN uses a neural network to approximate the Q-function, allowing it to:
# 
# 1. Handle continuous state spaces and high-dimensional inputs
# 2. Generalize across similar states through the network's learned features
# 3. Scale to problems where a Q-table would be impractical
# 
# Key Components of DQN:
# 
# - **Q-Network**: A neural network that takes a state as input and outputs Q-values for each possible action
# - **Experience Replay Buffer**: Stores transitions (state, action, reward, next_state) to break correlations in sequential data
# - **Target Network**: A separate network used for computing target Q-values, updated periodically to improve training stability
# - **Epsilon-Greedy Exploration**: Balances exploration and exploitation during training
# 
# The algorithm alternates between collecting experiences using the current policy and updating the Q-network using randomly sampled batches from the replay buffer. This approach has proven highly effective on many challenging tasks, from Atari games to robotics.
# 
# ## The CartPole Environment
# 
# [![CartPole](https://img.youtube.com/vi/46wjA6dqxOM/0.jpg)](https://www.youtube.com/watch?v=46wjA6dqxOM "CartPole")
# 
# The CartPole environment is a classic control problem where a pole is attached to a cart that moves along a frictionless track.
# The goal is to prevent the pole from falling over by moving the cart left or right. The environment has:
# 
# State Space (4 dimensions):
# - Cart Position: Position of cart on track (-4.8 to 4.8)
# - Cart Velocity: Speed of cart (-∞ to ∞)
# - Pole Angle: Angle of pole from vertical (~ -0.418 to 0.418 radians)
# - Pole Angular Velocity: Rate of change of angle (-∞ to ∞)
# 
# Action Space (2 actions):
# - Push cart left (0)
# - Push cart right (1)
# 
# Rewards:
# - +1 for every timestep the pole remains upright
# - Episode ends when:
#   1. Pole angle exceeds ±12 degrees
#   2. Cart position exceeds ±2.4 units
#   3. Episode length reaches 500 timesteps
# 
# 
# 
# 

# In[25]:


env = gym.make("CartPole-v1", render_mode="rgb_array")

print(env.action_space)  # 2 actions: left and right
print(env.observation_space)  # Box(4): each action can take a continuous range of values


# ## Q-Network
# 
# The Q-Network is a neural network that takes a **state observation** as input and outputs **Q-values for each possible action**. 
# For CartPole, it takes a 4-dimensional state vector (cart position, cart velocity, pole angle, pole angular velocity) 
# and outputs 2 Q-values - one for moving left and one for moving right.
# 
# The architecture we'll use is a simple Multi-Layer Perceptron (MLP) with:
# - Input layer: 4 units (matching observation space)
# - Hidden layer 1: 120 units with ReLU activation 
# - Hidden layer 2: 84 units with ReLU activation
# - Output layer: 2 units (matching action space)
# 
# We don't use ReLU on the output layer since Q-values can be negative. The network learns to approximate
# the expected sum of future discounted rewards (Q-value) for each action given the current state.
# 
# This replaces the Q-table from traditional Q-learning, allowing us to handle continuous state spaces
# and generalize across similar states through the network's learned features.
# 
# The Q-network learns to predict the total future rewards (Q-values) for each state-action pair.
# For example:
# - A state where the pole is tilting left + action to move cart left = High Q-value
#   Because this helps balance the pole, leading to a long episode with many +1 rewards
# - A state where cart is near edge + action toward edge = Low Q-value  
#   Because this quickly ends the episode by going out of bounds, earning few rewards
# So the Q-values reflect the long-term value of actions, not just immediate rewards.

# In[26]:


class QNetwork(nn.Module):
    """For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`."""

    layers: nn.Sequential

    def __init__(self, obs_shape: tuple[int], num_actions: int, hidden_sizes: list[int] = [120, 84]):
        super().__init__()
        assert len(obs_shape) == 1, "Expecting a single vector of observations"
        in_features_list = [obs_shape[0]] + hidden_sizes
        out_features_list = hidden_sizes + [num_actions]
        layers = []
        for i, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(in_features_list) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)


net = QNetwork(obs_shape=(4,), num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
assert isinstance(getattr(net, "layers", None), nn.Sequential)
print(net)
print(f"Total number of parameters: {n_params}")


# ## Replay Buffer
# 
# The Replay Buffer is a crucial component in DQN that helps stabilize training by breaking the correlation between consecutive experiences.
# It stores past experiences (state, action, reward, next state, termination) and allows random sampling during training.
# 
# Key benefits of replay buffers:
# 1. Better sample efficiency by reusing past experiences multiple times for learning
# 2. Reduces correlation in training data by randomly sampling experiences instead of using consecutive timesteps
# 3. Helps break the non-i.i.d nature of RL data collection, making training more like supervised learning
# 4. Allows learning from rare but important experiences that may not be frequently encountered
# 
# The buffer has a fixed size and operates like a queue - when full, old experiences are removed to make room for new ones.
# During training, mini-batches are randomly sampled from the buffer to update the Q-network.
# 
# For CartPole, we use a relatively small buffer since the environment is simple. More complex environments like Dota 2
# require much larger buffers (millions of experiences) to capture the diversity of possible scenarios.
# 
# Note: We store termination (out of bounds) separately from truncation (timeout) since we don't want the agent to learn
# that timing out is a form of failure - it should try to balance indefinitely, not just until timeout.
# 

# In[27]:


@dataclass
class ReplayBufferSamples:
    """
    Samples from the replay buffer, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}). Note - here, d_{t+1} is actually **terminated** rather
    than **done** (i.e. it records the times when we went out of bounds, not when the environment timed out).
    """

    obs: Float[Tensor, "sample_size *obs_shape"]
    actions: Float[Tensor, "sample_size *action_shape"]
    rewards: Float[Tensor, "sample_size"]
    terminated: Bool[Tensor, "sample_size"]
    next_obs: Float[Tensor, "sample_size *obs_shape"]


class ReplayBuffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    """

    rng: np.random.Generator
    obs: Float[Arr, "buffer_size *obs_shape"]
    actions: Float[Arr, "buffer_size *action_shape"]
    rewards: Float[Arr, "buffer_size"]
    terminated: Bool[Arr, "buffer_size"]
    next_obs: Float[Arr, "buffer_size *obs_shape"]

    def __init__(self, num_envs: int, obs_shape: tuple[int], action_shape: tuple[int], buffer_size: int, seed: int):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.obs = np.empty((0, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, *self.action_shape), dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.terminated = np.empty(0, dtype=bool)
        self.next_obs = np.empty((0, *self.obs_shape), dtype=np.float32)

    def add(
        self,
        obs: Float[Arr, "num_envs *obs_shape"],
        actions: Int[Arr, "num_envs *action_shape"],
        rewards: Float[Arr, "num_envs"],
        terminated: Bool[Arr, "num_envs"],
        next_obs: Float[Arr, "num_envs *obs_shape"],
    ) -> None:
        """
        Add a batch of transitions to the replay buffer.
        """
        # Check shapes & datatypes
        for data, expected_shape in zip(
            [obs, actions, rewards, terminated, next_obs], [self.obs_shape, self.action_shape, (), (), self.obs_shape]
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_envs, *expected_shape)

        # Add data to buffer, slicing off the old elements
        self.obs = np.concatenate((self.obs, obs))[-self.buffer_size :]
        self.actions = np.concatenate((self.actions, actions))[-self.buffer_size :]
        self.rewards = np.concatenate((self.rewards, rewards))[-self.buffer_size :]
        self.terminated = np.concatenate((self.terminated, terminated))[-self.buffer_size :]
        self.next_obs = np.concatenate((self.next_obs, next_obs))[-self.buffer_size :]

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        """
        Sample a batch of transitions from the buffer, with replacement.
        """
        indices = self.rng.integers(0, self.buffer_size, sample_size)

        return ReplayBufferSamples(
            *[
                t.tensor(x[indices], device=device)
                for x in [self.obs, self.actions, self.rewards, self.terminated, self.next_obs]
            ]
        )


# ## Exploration 
# 
# DQN uses a very simple exploration strategy called ε-greedy (epsilon-greedy). Here's how it works:
# 
# With probability ε, the agent takes a completely random action (exploration)
# With probability 1-ε, the agent takes what it currently thinks is the best action (exploitation)
# The value of ε starts high (lots of exploration) and gradually decreases over time (more exploitation)
# 
# This simple approach has some major limitations:
# The biggest issue is that it explores completely randomly, with no intelligence or direction. Imagine trying to solve a maze by randomly walking around - you might eventually find the exit, but it's extremely inefficient.
# This becomes particularly problematic in environments with sparse rewards - situations where the agent needs to complete several specific steps to get any reward. The Montezuma's Revenge game is a perfect example:
# 
# The agent needs to find a key
# - It needs to understand that the key is important
# - It needs to remember where the door is
# - It needs to use the key on the correct door
# 
# Random exploration has an extremely low probability of completing this sequence by chance. It's like trying to solve a complex puzzle by randomly moving pieces around.
# To address these limitations, researchers have tried several approaches:
# 
# - **Reward Shaping** - Adding extra rewards to guide the agent. For example, giving a small reward for picking up keys. However, this can backfire spectacularly:
#     - In CoastRunners, an agent learned to drive in circles collecting power-ups instead of racing
#     - In Montezuma's Revenge, an agent learned to repeatedly approach a key without actually collecting it
# 
# 
# - **Curiosity-Based Exploration** - Instead of random exploration, the agent is rewarded for visiting "surprising" or "novel" states. Methods like Random Network Distillation measure novelty by how well the agent can predict certain properties of a state. However, this too can have issues - like the "noisy TV problem" where agents become fixated on random noise sources because they're always novel.
# 
# The search for better exploration strategies remains an active area of research. The key challenge is finding ways to explore intelligently without requiring human-engineered rewards or falling into unintended behavior patterns.

# ### Linear Schedule
# 
# - The linear schedule is used to gradually decrease the exploration rate (epsilon, $\epsilon$) over time.
# - At the start of training, we want a high epsilon to encourage lots of random exploration.
# - As training progresses, we want to reduce epsilon so the agent relies more on its learned Q-values.
# - This schedule provides a smooth transition from exploration to exploitation.
# 

# In[28]:


def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    """Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    In other words, we are in "explore mode" with start_e >= epsilon >= end_e for the first `exploration_fraction` fraction
    of total timesteps, and then stay at end_e for the rest of the episode.
    """
    return start_e + (end_e - start_e) * min(current_step / (exploration_fraction * total_timesteps), 1)


epsilons = [
    linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
    for step in range(500)
]
line(epsilons, labels={"x": "steps", "y": "epsilon"}, title="Probability of random action", height=400, width=600)


# ### Epsilon Greedy Policy
# 
# The epsilon-greedy policy is a fundamental exploration strategy in reinforcement learning that balances exploration and exploitation:
# 
# - With probability epsilon (ε), the agent takes a random action (exploration)
# - With probability 1-epsilon, the agent takes the greedy action - the one with highest predicted Q-value (exploitation)
# 
# This helps address the maximization bias mentioned earlier, where the agent can become overly optimistic about certain actions.
# By occasionally taking random actions, the agent:
# 1. Explores the state space more thoroughly
# 2. Can discover better strategies it wouldn't find through pure exploitation
# 3. Avoids getting stuck in local optima
# 
# The epsilon parameter is typically annealed over time using a schedule (like the linear schedule above),
# starting high to encourage exploration and decreasing to favor exploitation of learned knowledge.
# 

# In[29]:


def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv,
    q_network: QNetwork,
    rng: np.random.Generator,
    obs: Float[Arr, "num_envs *obs_shape"],
    epsilon: float,
) -> Int[Arr, "num_envs *action_shape"]:
    """With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs:       The family of environments to run against
        q_network:  The QNetwork used to approximate the Q-value function
        obs:        The current observation for each environment
        epsilon:    The probability of taking a random action
    Outputs:
        actions:    The sampled action for each environment.
    """
    # Convert `obs` into a tensor so we can feed it into our model
    obs = t.from_numpy(obs).to(device)

    num_actions = envs.single_action_space.n
    if rng.random() < epsilon:
        return rng.integers(0, num_actions, size=(envs.num_envs,))
    else:
        q_scores = q_network(obs)
        return q_scores.argmax(-1).detach().cpu().numpy()


# ## DQN Model Training
# 
# Deep Q-Network (DQN) Algorithm
# 
# DQN combines Q-learning with deep neural networks to learn value functions for complex environments.
# The key components are:
# 
# 1. Q-Network: Neural network $Q(s,a;θ)$ that approximates the optimal Q-value function
#    - Takes state $s$ as input and outputs Q-values for each possible action
#    - Parameters $θ$ are optimized to minimize TD error
# 
# 2. Experience Replay Buffer:
#    - Stores transitions $(s_t, a_t, r_t+1, d_t+1, s_t+1)$ from agent's interactions
#    - Enables random sampling of past experiences for training
#    - Breaks correlations in sequential data and improves learning stability
# 
# 3. Target Network:
#    - Separate network $Q(s,a;θ_{target})$ with parameters $θ_{target}$
#    - Used to compute TD targets, making training more stable
#    - Parameters periodically updated to match the main Q-network

# 
# The DQN dataclass represents the hyperparameters and configuration for training a Deep Q-Network agent:
# 
# Training Process:
# - Agent takes `total_timesteps` steps in environment during training
# - First `buffer_size` steps fill replay buffer (no gradient updates yet)
# - After buffer fills, optimizer step every `steps_per_train` agent steps
# - Target network updated every `trains_per_target_update` Q-network steps
# 
# Key Parameters:
# - Basic: `seed, env_id, num parallel envs`
# - Logging: `wandb config for experiment tracking`
# - Training phases: `total_timesteps, buffer_size, update frequencies`
# - Optimization: `batch_size, learning_rate`
# - RL-specific: `gamma (discount), exploration parameters`
# 
# The `total_training_steps` is computed as:
# `(total_timesteps - buffer_size) // steps_per_train`
# This represents actual gradient update steps after buffer is filled

# In[30]:


from pathlib import Path

@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    num_envs: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "DQNCartPole"
    wandb_entity: str | None = None
    video_log_freq: int | None = 50

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 500_000
    steps_per_train: int = 10
    trains_per_target_update: int = 100
    buffer_size: int = 10_000

    # Optimization hparams
    batch_size: int = 128
    learning_rate: float = 2.5e-4

    # RL-specific
    gamma: float = 0.99
    exploration_fraction: float = 0.2
    start_e: float = 1.0
    end_e: float = 0.1

    def __post_init__(self):
        assert self.total_timesteps - self.buffer_size >= self.steps_per_train
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.steps_per_train
        self.video_save_path = Path("videos")


args = DQNArgs(total_timesteps=400_000)  # changing total_timesteps will also change ???
utils.arg_help(args)


# ## DQN Agent
# 
# The DQNAgent class handles interactions between the agent and environment. It:
# 1. Maintains the agent's state (observations, epsilon value for exploration)
# 2. Gets actions using epsilon-greedy policy via `get_actions()`
# 3. Steps through environment using `play_step()`:
#    - Gets actions using current policy
#    - Steps environment with these actions 
#    - Adds experiences `(obs, action, reward, next_obs)` to replay buffer
#    - Updates observation for next step
# 4. Follows separation of concerns - only handles env interaction, not Q-network/buffer creation

# In[31]:


class DQNAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        buffer: ReplayBuffer,
        q_network: QNetwork,
        start_e: float,
        end_e: float,
        exploration_fraction: float,
        total_timesteps: int,
        rng: np.random.Generator,
    ):
        self.envs = envs
        self.buffer = buffer
        self.q_network = q_network
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.rng = rng

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.obs, _ = self.envs.reset()  # Need a starting observation
        self.epsilon = start_e  # Starting value (will be updated in `get_actions`)

    def play_step(self) -> dict:
        """
        Carries out a single interaction step between agent & environment, and adds results to the replay buffer.

        Returns `infos` (list of dictionaries containing info we will log).
        """
        self.obs = np.array(self.obs, dtype=np.float32)
        actions = self.get_actions(self.obs)
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        # Get `real_next_obs` by finding all environments where we terminated & replacing `next_obs` with the actual terminal states
        true_next_obs = next_obs.copy()
        for n in range(self.envs.num_envs):
            if (terminated | truncated)[n]:
                true_next_obs[n] = infos["final_observation"][n]

        self.buffer.add(self.obs, actions, rewards, terminated, true_next_obs)
        self.obs = next_obs

        self.step += self.envs.num_envs
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        """
        self.epsilon = linear_schedule(
            self.step, self.start_e, self.end_e, self.exploration_fraction, self.total_timesteps
        )
        actions = epsilon_greedy_policy(self.envs, self.q_network, self.rng, obs, self.epsilon)
        assert actions.shape == (len(self.envs.envs),)
        return actions


# ## DQN Trainer
# 
# Training Loop:
# 1. Agent interacts with environment using ε-greedy policy based on Q-network
# 2. Store transitions in replay buffer
# 3. Sample random batch of transitions
# 4. Compute TD targets using target network:
#    $y_i = r_i + (1-d_i)\gamma \max_{a'} Q(s'_i,a';\theta_\text{target})$
# 5. Update Q-network parameters to minimize squared TD error:
#    $L(\theta) = \frac{1}{N}\sum_i(y_i - Q(s_i,a_i;\theta))^2$
# 6. Periodically update target network parameters:
#    $\theta_\text{target} \leftarrow \theta$
# 
# ![DQN Algorithm](../ref/ch2-dqn-algo.png)

# In[32]:


def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Helper function: returns dict of data from the first terminated environment, if at least one terminated.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }


class DQNTrainer:
    def __init__(self, args: DQNArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
        )

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Create our replay buffer
        self.buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        self.q_network = QNetwork(obs_shape, num_actions).to(device)
        self.target_network = QNetwork(obs_shape, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = t.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

        # Create our agent
        self.agent = DQNAgent(
            self.envs,
            self.buffer,
            self.q_network,
            args.start_e,
            args.end_e,
            args.exploration_fraction,
            args.total_timesteps,
            self.rng,
        )

    def add_to_replay_buffer(self, n: int, verbose: bool = False):
        """
        Takes n steps with the agent, adding to the replay buffer (and logging any results). Should return a dict of
        data from the last terminated episode, if any.

        Optional argument `verbose`: if True, we can use a progress bar (useful to check how long the initial buffer
        filling is taking).
        """
        data = None
        t0 = time.time()

        for step in tqdm(range(n), disable=not verbose, desc="Adding to replay buffer"):
            infos = self.agent.play_step()

            # Get data from environments, and log it if some environment did actually terminate
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data  # makes sure we return a non-empty dict at the end, if some episode terminates
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.agent.step)

        # Log Steps Per Second
        if self.args.use_wandb:
            wandb.log({"Steps Per Second": (n * self.envs.num_envs) / (time.time() - t0)}, step=self.agent.step)

        return data

    def prepopulate_replay_buffer(self):
        """
        Called to fill the replay buffer before training starts.
        """
        n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
        self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)

    def training_step(self, step: int) -> Float[Tensor, ""]:
        """
        Samples once from the replay buffer, and takes a single training step. The `step` argument is used to track the
        number of training steps taken.
        """
        data = self.buffer.sample(self.args.batch_size, device)  # s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}

        with t.inference_mode():
            target_max = self.target_network(data.next_obs).max(-1).values
        predicted_q_vals = self.q_network(data.obs)[range(len(data.actions)), data.actions]

        td_error = data.rewards + self.args.gamma * target_max * (1 - data.terminated.float()) - predicted_q_vals
        loss = td_error.pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.args.use_wandb:
            wandb.log(
                {"td_loss": loss, "q_values": predicted_q_vals.mean().item(), "epsilon": self.agent.epsilon},
                step=self.agent.step,
            )

    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch(self.q_network, log="all", log_freq=50)

        self.prepopulate_replay_buffer()

        pbar = tqdm(range(self.args.total_training_steps))
        last_logged_time = time.time()  # so we don't update the progress bar too much

        for step in pbar:
            data = self.add_to_replay_buffer(self.args.steps_per_train)
            if data is not None and time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(**data)

            self.training_step(step)

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()


# In[33]:


args = DQNArgs(use_wandb=True)
trainer = DQNTrainer(args)
trainer.train()


# ## Results
# 
# For metrics, logs, and videos, see: https://wandb.ai/michaelyliu6-none/DQNCartPole/runs/64y7t99t
# 
# ![catastrophic-forgetting.png](../ref/catastrophic-forgetting.png)
# 
# As shown above, the agent's performance randomly drops after a certain number of steps. This is a common issue in DQN where the agent suddenly "forgets" previously learned behaviors and experiences a dramatic drop in performance. This happens because DQN uses a neural network that is constantly being updated with new experiences, and these updates can overwrite or interfere with previously learned knowledge.
# 
# This issue occurs because:
# 1. The neural network is trained on recent experiences from the replay buffer
# 2. As new experiences replace old ones in the buffer, the network may lose access to important past experiences
# 3. The network weights are updated to optimize for recent experiences, potentially degrading performance on older scenarios
#  
#  Common solutions include:
#  - Larger replay buffers to retain more past experiences
#  - Prioritized experience replay to keep important experiences longer
#  - Multiple networks (like in Rainbow DQN) to preserve different aspects of learning
# 
# 
# 
#  
# 

# 
# <video controls src="ref/rl-video-episode-2800.mp4" />
