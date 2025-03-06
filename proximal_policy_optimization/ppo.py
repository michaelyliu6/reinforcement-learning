#!/usr/bin/env python
# coding: utf-8

# In[18]:


import itertools
import os
import sys
sys.path.append("..")
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import einops
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from matplotlib.animation import FuncAnimation
from numpy.random import Generator
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

Arr = np.ndarray

from utils import ppo_arg_help, set_global_seeds, make_env, get_episode_data_from_infos, prepare_atari_env

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# ## Policy Gradient vs Q-Learning Methods
# 
# Let's compare Policy Gradient methods (like PPO) with DQN implemented in `SARSA-Q-Learning-DQN.ipynb` to understand their key differences:
# 
# Policy gradient methods take a more direct approach to reinforcement learning compared to DQN. While DQN learns a **Q-function** $Q(s,a)$ that estimates expected future rewards and derives a policy by choosing actions that maximize Q-values, policy gradient methods directly learn and optimize a **policy function** $π(a|s)$ that maps states to action probabilities.
# 
# The key insight is that policy gradient methods perform gradient ascent directly on the expected future reward $J(π)$ with respect to the policy parameters. This avoids having to learn Q-values as an intermediate step and allows for more direct optimization of what we actually care about - getting high rewards.
# 
# Here's a comparison of the key aspects:
# 
# |                     | DQN                                                                                               | Policy Gradient Methods                                                                                                                                                                                                                                                                                                                                                                                          |
# |---------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | **Core Approach** | Learns Q-function to estimate action values, derives policy by choosing highest value actions | Directly learns and optimizes a policy function that outputs action probabilities |
# | **Networks** | Single network that outputs Q-values for each action | Policy network (actor) that outputs action probabilities, often with a separate value network (critic) |
# | **Action Spaces** | Limited to discrete action spaces, struggles with large action spaces | Works well with both discrete and continuous action spaces |
# | **Exploration** | Requires explicit exploration strategies like epsilon-greedy | Natural exploration through stochastic policy |
# | **Optimization** | Minimizes TD error based on Bellman equation | Directly maximizes expected rewards through policy gradient theorem |
# | **Stability** | Uses target networks to stabilize learning | Various techniques like trust regions (PPO) to ensure stable updates |
# 
# The direct optimization of the policy in policy gradient methods often leads to more stable learning compared to DQN's indirect approach through Q-values. However, estimating the policy gradient can be challenging and typically requires careful implementation of variance reduction techniques.
# 

# # Policy Gradient Objective Function
# 
# 
# The policy gradient objective function is central to understanding PPO and other policy gradient methods. Let's break it down:
# 
# For a policy $π_θ$ parameterized by $θ$, we want to maximize the expected return $J(π_θ)$:
# 
# $$
# J(\pi_\theta) = \underset{\tau \sim \pi_\theta}{\mathbb{E}} \left[ \sum_{t=0}^T r_{t+1}(s_t, a_t, s_{t+1}) \right]
# $$
# 
# where the expectation is over trajectories τ sampled from πθ. The policy gradient theorem gives us:
# 
# $$
# \nabla_\theta J\left(\pi_\theta\right)=\underset{\tau \sim \pi_\theta}{\mathbb{E}}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) A_\theta(s_t, a_t)\right] \quad 
# $$
# 
# where $A_θ(s_t, a_t)$ is the advantage function: $Q(s_t, a_t) - V(s_t)$. This tells us how much better action $a_t$ is compared
# to the average action in state $s_t$.
# 
# The intuition is:
# - Positive advantage → increase probability of that action
# - Negative advantage → decrease probability of that action
# 
# Note: While we could use total trajectory reward $R(τ)$ instead of advantages (known as REINFORCE), using advantages gives much lower variance since it
# isolates the contribution of each action from the overall trajectory outcome.
# 
# To optimize this in practice, we:
# 1. Sample trajectories using current policy
# 2. Estimate advantages using a frozen target network
# 3. Maximize the objective:
# $$
# L(\theta) = \frac{1}{|B|} \sum_{t \in B} \log \pi_\theta(a_t \mid s_t) \hat{A}_{\theta_\text{target}}(s_t, a_t) 
# $$
# 
# This gives us $∇_θL(θ) ≈ ∇_θJ(π_θ)$, allowing us to perform gradient ascent to improve the policy. This approximation is crucial because it transforms our sparse, discrete rewards into a continuous, differentiable learning signal - we can estimate the quality of every action through advantages and action probabilities, rather than only learning from occasional reward signals.
# 
# Note: For simplicity of explanation, we present the objective function in the finite-horizon setting without a discount factor $γ$. 
# In our implementation below, we use infinite-horizon with discounting to handle continuing tasks and to prioritize near-term rewards. The discounted version simply adds $γ^t$ to each reward term.
# 
# 
# 

# # Missing Components of PPO
# 
# The policy gradient objective function described above is a good starting point, but it's missing a few key components that make PPO work in practice. These are:
# 
# 1. **Entropy Bonus**: We add an entropy bonus to the objective function to encourage exploration. This is a measure of the uncertainty of the policy, and is defined as:
# 
#     $$
#     H(\pi_\theta(s_t)) = -\sum_{a} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)
#     $$
# 
#     We add this to the objective function with a coefficient $c_1$:
# 
#     $$
#     L(\theta) = \frac{1}{|B|} \sum_{t \in B} \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_{\theta_\text{target}}(s_t, a_t) + c_1 H(\pi_\theta(s_t)) \right]
#     $$
# 
#     The entropy bonus encourages the policy to explore more, and prevents it from converging to a suboptimal policy too early. The coefficient $c_1$ is usually decayed over time, as we move from exploration to exploitation.
# 
# 2. **Clipped Objective Function**: We clip the objective function to prevent the policy from changing too much too fast. This is done by clipping the probability ratio between the new and old policy:
# 
#     $$
#     r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{target}}(a_t \mid s_t)}
#     $$
# 
#     We then clip this ratio to be within a range $[1-\epsilon, 1+\epsilon]$:
# 
#     $$
#     L^\text{CLIP}(\theta) = \frac{1}{|B|} \sum_{t \in B} \min \left( r_t(\theta) \hat{A}_{\theta_\text{target}}(s_t, a_t), \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{\theta_\text{target}}(s_t, a_t) \right)
#     $$
# 
#     This prevents the policy from changing too much in a single update, which can lead to instability and poor performance. The clipping parameter $\epsilon$ is usually set to 0.2.
# 
# 3. **Using the Probability Ratio**: Instead of using the log probability $\log \pi_\theta(a_t \mid s_t)$ in the objective function, we use the probability ratio $r_t(\theta)$. This is valid because we're using clipping, which means the ratio is usually close to 1. We can then use the approximation $\log(x) \approx x - 1$ for $x \approx 1$, which means the two loss functions are equal up to a constant that doesn't depend on $\theta$.
# 
# These three components are crucial to the success of PPO. The entropy bonus encourages exploration, the clipped objective function prevents instability, and using the probability ratio simplifies the objective function.

# # PPO Overview
# 
# Training PPO has two main phases:
# 1. Rollouts: Collect data from the environment
# 2. Learning: Update the policy and value function
# 
# In the rollout phase, we sample actions from the current policy and execute them in the environment. We store the observations, actions, rewards, and other relevant information in a buffer.
# 
# In the learning phase, we use the data in the buffer to update the policy and value function. We compute the advantages and returns, and use these to update the policy and value function.
# 
# # # <img src="../ref/ppo-algo.png" width="50%">
# 
# Our implementation is split into 3 main components:
# 1. `PPOAgent`- contains the actor and critic networks, and the play_step method carries out a single interaction step between the agent and the environment and has a `get_minibatches` method to get the current state of the replay memory.
# 2. `ReplayMemory` - stores the experiences generated in the rollout phase.
# 3. `PPOTrainer` -  contains the PPO algorithm (`rollout_phase` and `learn_phase`).
# 

# # The CartPole Environment
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

# # PPO Arguments

# In[19]:


@dataclass
class PPOArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    mode: Literal["classic-control", "atari", "mujoco"] = "classic-control"

    # Wandb / logging
    use_wandb: bool = False
    video_log_freq: int | None = None
    wandb_project_name: str = "PPOCartPole"
    wandb_entity: str = None

    # Duration of different phases
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps_per_rollout: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99

    # PPO-specific hyperparameters
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches

        self.video_save_path = Path("ppo-videos")


args = PPOArgs(num_minibatches=2)  # changing this also changes minibatch_size and total_training_steps
ppo_arg_help(args)


# # Actor-Critic Networks
# 
# Unlike value-based methods like DQN, policy gradient methods like PPO use two distinct networks:
# 
# 1. Actor Network (Policy Network):
#    - Takes state s_t as input
#    - Outputs probability distribution over actions $π_θ(a_t|s_t)$
#    - For discrete actions: outputs logits for each possible action
#    - For continuous actions: outputs means and variances of action distribution
#    - Optimized using PPO objective to maximize expected rewards
# 
# 2. Critic Network (Value Network): 
#    - Takes state $s_t$ as input
#    - Outputs scalar value estimate $V_θ(s_t)$
#    - Used to compute advantage estimates $Â_θ(s_t,a_t)$
#    - Trained by minimizing TD residual loss: $(V_θ(s_t) - V_target)²$
#    - Where $V_{target}$ = $V_{θ_{target}}(s_t) + Â_{θ_{target}}(s_t,a_t)$
# 
# Key Differences and Complementary Roles:
# |                | Actor                     | Critic                    |
# |----------------|---------------------------|---------------------------|
# | Purpose        | Learn optimal policy      | Estimate state values    |
# | Output         | Action probabilities      | Single value scalar      |
# | Training       | PPO objective function    | TD residual loss         |
# | Role           | Select actions           | Guide policy updates     |
# 
# Why Both Networks Are Essential:
# - The critic enables stable advantage estimation vs using raw returns
# - Without the critic, we'd rely on accumulated rewards R(τ) which has:
#   * Very high variance
#   * Poor credit assignment to specific actions
# - The actor learns the actual policy we want to deploy
# - Without the actor, we'd have no policy to learn values for
# 
# Key Benefits of This Architecture:
# - Can handle both discrete and continuous action spaces
# - More stable training through advantage estimation
# - Better credit assignment through TD learning
# - Separates policy learning from value estimation
# 
# 

# In[20]:


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_actor_and_critic(
    envs: gym.vector.SyncVectorEnv,
    mode: Literal["classic-control", "atari", "mujoco"] = "classic-control",
) -> tuple[nn.Module, nn.Module]:
    """
    Returns (actor, critic), the networks used for PPO, in one of 3 different modes.
    """
    assert mode in ["classic-control", "atari", "mujoco"]

    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else np.array(envs.single_action_space.shape).prod()
    )

    if mode == "classic-control":
        actor, critic = get_actor_and_critic_classic(num_obs, num_actions)
    if mode == "atari":
        actor, critic = get_actor_and_critic_atari(obs_shape, num_actions) # TODO: Implement this
    if mode == "mujoco":
        actor, critic = get_actor_and_critic_mujoco(num_obs, num_actions) # TODO: Implement this

    return actor.to(device), critic.to(device)


def get_actor_and_critic_classic(num_obs: int, num_actions: int):
    """
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    """
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )

    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01),
    )
    return actor, critic


# # Generalized Advantage Estimation
# 
# Generalized Advantage Estimation (GAE) is a crucial technique for estimating how good an action was compared to the average action in a given state. It helps solve a fundamental tradeoff in reinforcement learning between immediate feedback and long-term consequences.
# 
# The advantage function $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ measures how much better an action is compared to the average. We can estimate it in different ways:
# 
# | Method | Formula | Pros | Cons |
# |--------|---------|------|------|
# | 1-step TD | $δ_t = r_t + γV(s_{t+1}) - V(s_t)$ | Low variance, clear credit | Myopic - misses long-term effects |
# | Full trajectory | $Σ_k δ_{t+k}$ | Captures all future effects | High variance, hard to credit actions |
# | GAE | $Σ_k (γλ)^k δ_{t+k}$ | Balanced approach | Requires tuning λ parameter |
# 
# Here, $δ_t$ is the TD error - the difference between the actual reward plus discounted next state value, and our current value estimate. It represents our 1-step prediction error and serves as a fundamental building block for advantage estimation.
# 
# 
# The full GAE formula is:
# $$A^{GAE(λ)}_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ... + (γλ)^{T-t+1}δ_{T-1}$$
# 
# We can compute this efficiently using the recursive form:
# $$A^{GAE(λ)}_t = δ_t + (1-d_{t+1})(γλ)A^{GAE(λ)}_{t+1}$$
# 
# Where:
# - $γ$ is the discount factor
# - $λ$ controls the tradeoff between bias and variance
# - $d_{t+1}$ is 1 if the episode terminated at $t+1$, 0 otherwise
# 
# Key benefits of GAE:
# 1. Balances between immediate and future rewards
# 2. Allows proper credit assignment to individual actions
# 3. Reduces variance while maintaining information about long-term effects
# 4. Improves training stability through controlled weighting of future rewards
# 
# The λ parameter gives us precise control:
# - $λ=0$: Only immediate effects (like 1-step TD)
# - $λ=1$: Full trajectory consideration
# - $0 < λ < 1$: Balanced weighting that works best in practice
# 
# 
# 

# In[21]:


@t.inference_mode()
def compute_advantages(
    next_value: Float[Tensor, "num_envs"],
    next_terminated: Bool[Tensor, "num_envs"],
    rewards: Float[Tensor, "buffer_size num_envs"],
    values: Float[Tensor, "buffer_size num_envs"],
    terminated: Bool[Tensor, "buffer_size num_envs"],
    gamma: float,
    gae_lambda: float,
) -> Float[Tensor, "buffer_size num_envs"]:
    """
    Compute advantages using Generalized Advantage Estimation.
    """
    T = values.shape[0]
    terminated = terminated.float() # casting bool to float to use in the recursive formula
    next_terminated = next_terminated.float() 

    # Get tensors of V(s_{t+1}) and d_{t+1} for all t = 0, 1, ..., T-1
    next_values = t.concat([values[1:], next_value[None, :]])
    next_terminated = t.concat([terminated[1:], next_terminated[None, :]])

    # Compute deltas: \delta_t = r_t + (1 - d_{t+1}) \gamma V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * (1.0 - next_terminated) - values

    # Compute advantages using the recursive formula, starting with advantages[T-1] = deltas[T-1] and working backwards
    advantages = t.zeros_like(deltas)
    advantages[-1] = deltas[-1]
    for s in reversed(range(T - 1)):
        advantages[s] = deltas[s] + (1.0 - terminated[s + 1]) * (gamma * gae_lambda) * advantages[s + 1]

    return advantages


# # Replay Memory
# 
# The `ReplayMemory` class is designed to store and manage experiences collected during the rollout phase of the PPO algorithm. It serves as a buffer to hold data from multiple environments over a series of time steps. This data is then used to update the agent's policy and value function during the learning phase.
# 
# Key Features:
# 
# 1. **Storage**: The class stores observations (`obs`), actions (`actions`), log probabilities of actions (`logprobs`), values (`values`), rewards (`rewards`), and termination flags (`terminated`) for each time step in each environment.
# 
# 2. **Sampling**: It provides a method to sample minibatches of experiences from the stored data. These minibatches are used to train the neural networks, ensuring that updates are made based on a diverse set of experiences.
# 
# 3. **Minibatch Creation**: The `get_minibatches` method divides the collected data into smaller, randomly shuffled minibatches. This helps in stabilizing training by reducing the variance in updates and allows for parallel processing.
# 
# 4. **Data Management**: The class handles the organization and retrieval of data, ensuring that experiences are used efficiently during the learning phase. It also manages the random number generator (`rng`) for reproducibility.
# 
# Usage:
# 
# -   During the rollout phase, experiences are added to the `ReplayMemory` using the `add` method.
# -   During the learning phase, `get_minibatches` is called to retrieve a set of minibatches for training.
# -   The `batches_per_learning_phase` parameter determines how many times the entire dataset is sampled and used for updates in each learning phase.
# 
# Variables:
# 
# -   `obs`: Stores the observations from the environment.
# -   `actions`: Stores the actions taken by the agent.
# -   `logprobs`: Stores the log probabilities of the actions taken, calculated from the policy's output.
# -   `values`: Stores the estimated values of the states, as predicted by the value function.
# -   `rewards`: Stores the rewards received from the environment.
# -   `terminated`: Stores boolean flags indicating whether an episode has terminated at each step.
# 
# Purpose:
# 
# The primary purpose of `ReplayMemory` is to facilitate efficient and stable training of the PPO agent by:
# 
# -   Storing a large set of experiences.
# -   Providing a mechanism to sample diverse minibatches for training.
# -   Ensuring that all collected experiences are utilized multiple times during the learning phase, enhancing data efficiency.
# 

# In[22]:


@dataclass
class ReplayMinibatch:
    """
    Samples from the replay memory, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_{t+1})
    """

    obs: Float[Tensor, "minibatch_size *obs_shape"]
    actions: Int[Tensor, "minibatch_size *action_shape"]
    logprobs: Float[Tensor, "minibatch_size"]
    advantages: Float[Tensor, "minibatch_size"]
    returns: Float[Tensor, "minibatch_size"]
    terminated: Bool[Tensor, "minibatch_size"]


class ReplayMemory:
    """
    Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
    """

    rng: Generator
    obs: Float[Arr, "buffer_size num_envs *obs_shape"]
    actions: Int[Arr, "buffer_size num_envs *action_shape"]
    logprobs: Float[Arr, "buffer_size num_envs"]
    values: Float[Arr, "buffer_size num_envs"]
    rewards: Float[Arr, "buffer_size num_envs"]
    terminated: Bool[Arr, "buffer_size num_envs"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.batches_per_learning_phase = batches_per_learning_phase
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Resets all stored experiences, ready for new ones to be added to memory."""
        self.obs = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_envs, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
        self.values = np.empty((0, self.num_envs), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
        self.terminated = np.empty((0, self.num_envs), dtype=bool)

    def add(
        self,
        obs: Float[Arr, "num_envs *obs_shape"],
        actions: Int[Arr, "num_envs *action_shape"],
        logprobs: Float[Arr, "num_envs"],
        values: Float[Arr, "num_envs"],
        rewards: Float[Arr, "num_envs"],
        terminated: Bool[Arr, "num_envs"],
    ) -> None:
        """Add a batch of transitions to the replay memory."""
        # Check shapes & datatypes
        for data, expected_shape in zip(
            [obs, actions, logprobs, values, rewards, terminated], [self.obs_shape, self.action_shape, (), (), (), ()]
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_envs, *expected_shape)

        # Add data to buffer (not slicing off old elements)
        self.obs = np.concatenate((self.obs, obs[None, :]))
        self.actions = np.concatenate((self.actions, actions[None, :]))
        self.logprobs = np.concatenate((self.logprobs, logprobs[None, :]))
        self.values = np.concatenate((self.values, values[None, :]))
        self.rewards = np.concatenate((self.rewards, rewards[None, :]))
        self.terminated = np.concatenate((self.terminated, terminated[None, :]))

    def get_minibatches(
        self, next_value: Tensor, next_terminated: Tensor, gamma: float, gae_lambda: float
    ) -> list[ReplayMinibatch]:
        """
        Returns a list of minibatches. Each minibatch has size `minibatch_size`, and the union over all minibatches is
        `batches_per_learning_phase` copies of the entire replay memory.
        """
        # Convert everything to tensors on the correct device
        obs, actions, logprobs, values, rewards, terminated = (
            t.tensor(x, device=device)
            for x in [self.obs, self.actions, self.logprobs, self.values, self.rewards, self.terminated]
        )

        # Compute advantages & returns
        advantages = compute_advantages(next_value, next_terminated, rewards, values, terminated, gamma, gae_lambda)
        returns = advantages + values

        # Return a list of minibatches
        minibatches = []
        for _ in range(self.batches_per_learning_phase):
            for indices in get_minibatch_indices(self.rng, self.batch_size, self.minibatch_size):
                minibatches.append(
                    ReplayMinibatch(
                        *[
                            data.flatten(0, 1)[indices]
                            for data in [obs, actions, logprobs, advantages, returns, terminated]
                        ]
                    )
                )

        # Reset memory (since we only need to call this method once per learning phase)
        self.reset()

        return minibatches
    
def get_minibatch_indices(rng: Generator, batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """
    Return a list of length `num_minibatches`, where each element is an array of `minibatch_size` and the union of all
    the arrays is the set of indices [0, 1, ..., batch_size - 1] where `batch_size = num_steps_per_rollout * num_envs`.
    """
    assert batch_size % minibatch_size == 0
    num_minibatches = batch_size // minibatch_size
    indices = rng.permutation(batch_size).reshape(num_minibatches, minibatch_size)
    return list(indices)


# In[23]:


class PPOAgent:
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, actor: nn.Module, critic: nn.Module, memory: ReplayMemory):
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.next_obs = t.tensor(envs.reset()[0], device=device, dtype=t.float)  # need starting obs (in tensor form)
        self.next_terminated = t.zeros(envs.num_envs, device=device, dtype=t.bool)  # need starting termination=False

    def play_step(self) -> list[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        """
        # Get newest observations (i.e. where we're starting from)
        obs = self.next_obs
        terminated = self.next_terminated

        # Compute logits based on newest observation, and use it to get an action distribution we sample from
        with t.inference_mode():
            logits = self.actor(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        # Step environment based on the sampled action
        next_obs, rewards, next_terminated, next_truncated, infos = self.envs.step(actions.cpu().numpy())

        # Calculate logprobs and values, and add this all to replay memory
        logprobs = dist.log_prob(actions).cpu().numpy()
        with t.inference_mode():
            values = self.critic(obs).flatten().cpu().numpy()
        self.memory.add(obs.cpu().numpy(), actions.cpu().numpy(), logprobs, values, rewards, terminated.cpu().numpy())

        # Set next observation & termination state
        self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float)
        self.next_terminated = t.from_numpy(next_terminated).to(device, dtype=t.float)

        self.step += self.envs.num_envs
        return infos

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Gets minibatches from the replay memory, and resets the memory
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches
    



# # Objective Function
# 
# 
# 

# ## Clipped Surrogate Objective
# 
# The clipped surrogate objective is a key component of PPO that helps ensure stable policy updates.
# 
# The objective function is:
# $$L^{CLIP}(θ) = \frac{1}{|B|} \sum_t[ \min(r_t(θ)A_t, \text{clip}(r_t(θ), 1-ε, 1+ε)A_t) ]$$
# 
# Where:
# - $r_t(θ) = \frac{π_θ(a_t|s_t)}{π_{θ_{old}}(a_t|s_t)}$ is the probability ratio between new and old policies
# - $A_t$ is the advantage estimate
# - $ε$ is the clip coefficient (typically 0.2)
# 
# The clipping works as follows:
# - For positive advantages ($A_t > 0$):
#   - If $r_t > 1+ε$: The objective is clipped, discouraging too large policy changes
#   - If $r_t < 1+ε$: The objective remains unclipped, allowing policy improvements
# 
# - For negative advantages ($A_t < 0$):
#   - If $r_t < 1-ε$: The objective is clipped, preventing too large policy changes
#   - If $r_t > 1-ε$: The objective remains unclipped
# 
# This clipping mechanism is what makes the policy updates "proximal" - it prevents too large changes
# to the policy in a single update, improving training stability.
# 
# ![Clipped Surrogate Objective](../ref/clipping.png)
# 

# In[24]:


def calc_clipped_surrogate_objective(
    probs: Categorical,
    minibatch_action: Int[Tensor, "minibatch_size"],
    minibatch_advantages: Float[Tensor, "minibatch_size"],
    minibatch_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    minibatch_action:
        what actions actions were taken in the sampled minibatch
    minibatch_advantages:
        advantages calculated from the sampled minibatch
    minibatch_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    """
    assert minibatch_action.shape == minibatch_advantages.shape == minibatch_logprobs.shape
    logits_diff = probs.log_prob(minibatch_action) - minibatch_logprobs

    prob_ratio = t.exp(logits_diff)

    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + eps)

    non_clipped = prob_ratio * minibatch_advantages
    clipped = t.clip(prob_ratio, 1 - clip_coef, 1 + clip_coef) * minibatch_advantages

    return t.minimum(non_clipped, clipped).mean()


# ## Value Function Loss
# 
# The value function loss measures how well the critic network predicts the expected returns. It is computed as:
# 
# $L^{VF} = c_1 \cdot \frac{1}{|MB|} \sum_{s \in MB} (V_\theta(s) - R_t)^2$
# 
# Where:
# - $V_\theta(s)$ is the value prediction from the critic network for state s
# - $R_t$ is the target return (computed as advantages + old values)
# - $c_1$ is the value function coefficient that weights this loss term
# - $|MB|$ is the minibatch size
# 
# The loss is a simple MSE between predicted values and target returns, scaled by `vf_coef`. Minimizing this loss helps the critic make better value predictions, which in turn leads to better advantage estimates for policy updates.
# 

# In[25]:


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"], mb_returns: Float[Tensor, "minibatch_size"], vf_coef: float
) -> Float[Tensor, ""]:
    """Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    """
    assert values.shape == mb_returns.shape

    return vf_coef * (values - mb_returns).pow(2).mean()


# ## Entropy Bonus
# 
# The entropy bonus term is intended to incentivize exploration by increasing the entropy of the action distribution.
# For a discrete probability distribution p, the entropy H is defined as:
# 
# $$
# H(p) = \sum_x p(x) \ln \frac{1}{p(x)}
# $$
# 
# If $p(x) = 0$, then we define $0 * ln(1/0) := 0$ (by taking the limit as $p(x) -> 0$).
# 
# The entropy bonus is computed as:
# $$L^{ENT} = c_2 * H(π_θ)$$
# 
# Where:
# - $H(π_θ)$ is the entropy of the current policy's action distribution
# - $c_2$ is the entropy coefficient that weights this term
# 
# The entropy bonus is added to the objective function to encourage exploration. When entropy is high,
# the policy is more random and exploratory. When entropy is low, the policy is more deterministic.
# Early in training we want high entropy for exploration, while later we want it to decrease as the
# policy converges to optimal behavior. Monitoring entropy during training provides a useful diagnostic:
# if entropy remains too high, the policy may be failing to learn; if it drops too quickly, the policy
# may be prematurely converging to a suboptimal solution.
# 
# 

# In[26]:


def calc_entropy_bonus(dist: Categorical, ent_coef: float):
    """Return the entropy bonus term, suitable for gradient ascent.

    dist:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    """
    return ent_coef * dist.entropy().mean()


# # Learning Rate Scheduler
# 
# The learning rate scheduler implements linear learning rate decay over the course of training.
# The optimizer is configured as AdamW with epsilon=1e-5 (as per PPO implementation details).
# 
# The scheduler linearly decreases the learning rate from initial_lr to end_lr over total_phases steps.
# This is done by directly modifying the learning rate in the optimizer's parameter groups.
# 
# The optimizer combines both actor and critic parameters into a single parameter group using itertools.chain.
# We set maximize=True since we're doing gradient ascent rather than descent.
# 
# This learning rate annealing helps stabilize training - starting with a higher learning rate allows
# for faster initial progress, while gradually reducing it helps fine-tune the policy more precisely.

# In[27]:


class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_phases: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_phases = total_phases
        self.n_step_calls = 0

    def step(self):
        """Implement linear learning rate decay so that after `total_phases` calls to step, the learning rate is end_lr.

        Do this by directly editing the learning rates inside each param group (i.e. `param_group["lr"] = ...`), for each param
        group in `self.optimizer.param_groups`.
        """
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_phases
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)


def make_optimizer(
    actor: nn.Module, critic: nn.Module, total_phases: int, initial_lr: float, end_lr: float = 0.0
) -> tuple[optim.Adam, PPOScheduler]:
    """
    Return an appropriately configured Adam with its attached scheduler.
    """
    optimizer = optim.AdamW(
        itertools.chain(actor.parameters(), critic.parameters()), lr=initial_lr, eps=1e-5, maximize=True
    )
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_phases)
    return optimizer, scheduler


# # Trainer
# 
# 
# The PPOTrainer class handles the core training loop for Proximal Policy Optimization (PPO).
# It manages:
# - Environment setup and vectorization for parallel environments
# - Network initialization (actor and critic networks)
# - Replay memory for storing experiences
# - Optimization setup with learning rate scheduling
# - Rollout collection by stepping through environments
# - Policy updates via PPO objective optimization
# - Logging of training metrics and episode statistics
# 
# The trainer coordinates the interaction between the agent and environments during rollout phases
# to collect experiences, then uses these experiences to update the policy during learning phases.
# 
# 

# In[30]:


class PPOTrainer:
    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
        )

        # Define some basic variables from our environment
        self.num_envs = self.envs.num_envs
        self.action_shape = self.envs.single_action_space.shape
        self.obs_shape = self.envs.single_observation_space.shape

        # Create our replay memory
        self.memory = ReplayMemory(
            self.num_envs,
            self.obs_shape,
            self.action_shape,
            args.batch_size,
            args.minibatch_size,
            args.batches_per_learning_phase,
            args.seed,
        )

        # Create our networks & optimizer
        self.actor, self.critic = get_actor_and_critic(self.envs, mode=args.mode)
        self.optimizer, self.scheduler = make_optimizer(self.actor, self.critic, args.total_training_steps, args.lr)

        # Create our agent
        self.agent = PPOAgent(self.envs, self.actor, self.critic, self.memory)

    def rollout_phase(self) -> dict | None:
        """
        This function populates the memory with a new set of experiences, using `self.agent.play_step` to step through
        the environment. It also returns a dict of data which you can include in your progress bar postfix.
        """
        data = None
        t0 = time.time()

        for step in range(self.args.num_steps_per_rollout):
            # Play a step, returning the infos dict (containing information for each environment)
            infos = self.agent.play_step()

            # Process episode completion data if available from any environment
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.agent.step)
                

        if self.args.use_wandb:
            wandb.log(
                {"SPS": (self.args.num_steps_per_rollout * self.num_envs) / (time.time() - t0)}, step=self.agent.step
            )

        return data

    def learning_phase(self) -> None:
        """
        This function does the following:
            - Generates minibatches from memory
            - Calculates the objective function, and takes an optimization step based on it
            - Clips the gradients 
            - Steps the learning rate scheduler
        """
        minibatches = self.agent.get_minibatches(self.args.gamma, self.args.gae_lambda)
        for minibatch in minibatches:
            objective_fn = self.compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.args.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        """
        Handles learning phase for a single minibatch. Returns objective function to be maximized.
        """
        # Calculate actor and critic outputs
        logits = self.actor(minibatch.obs) # policy model
        dist = Categorical(logits=logits) # action distribution
        values = self.critic(minibatch.obs).squeeze() # value model

        # Clipped surrogate objective (MAXIMIZE) Make more good actions more likely
        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            dist, # action distribution
            minibatch.actions, # actions taken in minibatch
            minibatch.advantages, # GAE (Generalized Advantage Estimation) 
            minibatch.logprobs, # Reference policy logprobs of actions from before update
            self.args.clip_coef # clip coefficient
        )

        # Value loss function (MINIMIZE) Improve Critic/Value function better at predicting returns
        value_loss = calc_value_function_loss(values, # Value model output
                                              minibatch.returns, # Reward function
                                              self.args.vf_coef)
        
        # Entropy bonus (MAXIMIZE) Encourage exploration
        entropy_bonus = calc_entropy_bonus(dist, self.args.ent_coef)

        # Calculate total objective function to be maximized
        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        # Log to wandb
        if self.args.use_wandb:
            with t.inference_mode():
                newlogprob = dist.log_prob(minibatch.actions) 
                logratio = newlogprob - minibatch.logprobs 
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        
                wandb.log(
                    dict(
                        total_steps=self.agent.step,
                        values=values.mean().item(),
                        lr=self.scheduler.optimizer.param_groups[0]["lr"],
                        value_loss=value_loss.item(),
                        clipped_surrogate_objective=clipped_surrogate_objective.item(),
                        entropy=entropy_bonus.item(),
                        approx_kl=approx_kl,
                        clipfrac=np.mean(clipfracs),
                    ),
                    step=self.agent.step,
                )

        return total_objective_function

    def train(self) -> None:
        if args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch([self.actor, self.critic], log="all", log_freq=50)

        pbar = tqdm(range(self.args.total_phases))
        last_logged_time = time.time()  # so we don't update the progress bar too much

        for phase in pbar:
            data = self.rollout_phase()
            if data is not None and time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(phase=phase, **data)
            self.learning_phase()

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()


# In[31]:


args = PPOArgs(use_wandb=True, video_log_freq=50)
trainer = PPOTrainer(args)
trainer.train() # uncomment to run training loop


# See wandb for training metrics and videos: https://wandb.ai/michaelyliu6-none/PPOCartPole?nw=nwusermichaelyliu6

# # Atari (TODO: Get this working)

# In[14]:


gym.envs.registration.registry.keys()


# In[15]:


env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

print(env.action_space)  # Discrete(4): 4 actions to choose from
print(env.observation_space)  # Box(0, 255, (210, 160, 3), uint8): an RGB image of the game screen


# In[16]:


print(env.get_action_meanings())


# In[17]:


def display_frames(frames: Int[Arr, "timesteps height width channels"], figsize=(4, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(frames[0])
    plt.close()

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    display(HTML(ani.to_jshtml()))


nsteps = 150

frames = []
obs, info = env.reset()
for _ in tqdm(range(nsteps)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(obs)

display_frames(np.stack(frames))


# In[18]:


env_wrapped = prepare_atari_env(env)

frames = []
obs, info = env_wrapped.reset()
for _ in tqdm(range(nsteps)):
    action = env_wrapped.action_space.sample()
    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    obs = einops.repeat(np.array(obs), "frames h w -> h (frames w) 3")  # stack frames across the row
    frames.append(obs)

display_frames(np.stack(frames), figsize=(12, 3))


# In[19]:


def get_actor_and_critic_atari(obs_shape: tuple[int,], num_actions: int) -> tuple[nn.Sequential, nn.Sequential]:
    """
    Returns (actor, critic) in the "atari" case, according to diagram above.
    """
    assert obs_shape[-1] % 8 == 4

    L_after_convolutions = (obs_shape[-1] // 8) - 3
    in_features = 64 * L_after_convolutions * L_after_convolutions

    hidden = nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4, padding=0)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(in_features, 512)),
        nn.ReLU(),
    )

    actor = nn.Sequential(hidden, layer_init(nn.Linear(512, num_actions), std=0.01))
    critic = nn.Sequential(hidden, layer_init(nn.Linear(512, 1), std=1))

    return actor, critic


# In[20]:


args = PPOArgs(
    env_id="ALE/Breakout-v5",
    wandb_project_name="PPOAtari",
    use_wandb=True,
    mode="atari",
    clip_coef=0.1,
    num_envs=8,
    video_log_freq=25,
)
trainer = PPOTrainer(args)


# In[22]:


# trainer.train()

