#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import sys
sys.path.append("..")
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
from tqdm import tqdm
import einops
from pathlib import Path
import matplotlib.pyplot as plt
import gym
import gym.envs.registration
import gym.spaces
from typing import Optional, Union, List, Tuple
from plot_utils import imshow
import utils


# # Reference
# 
# In this notebook, we will be implementing the k-armed Bandit problem described in Chapter 2 and Tabular RL/Policy Improvement described in Chapter 3 and Chapter 4 of [Reinforcement Learning: An Introduction by Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf). 

# # Multi-Armed Bandit
# ## Setting up OpenAI Gymnasium
# 
# The main methods to override in gym.env are step, reset, and render. The step method is called when the agent takes an action, the reset method is called when the environment is reset, and the render method is called when the environment is rendered. 
# - The step method should return the observation, reward, whether the episode has terminated, whether the episode has timed out, and any additional information. 
# - The reset method should return the initial observation and any additional information. 
# - The render method should render the environment in a human-readable format.

# In[12]:


ObsType = int
ActType = int

class MultiArmedBandit(gym.Env):
    """
    A class representing a multi-armed bandit environment, based on OpenAI Gym's Env class.

    Attributes:
        action_space (gym.spaces.Discrete): The space of possible actions, representing the arms of the bandit.
        observation_space (gym.spaces.Discrete): The space of possible observations.
        num_arms (int): The number of arms in the bandit.
        stationary (bool): Indicates whether the reward distribution (i.e. the arm_reward_means) is stationary or not.
        arm_reward_means (np.ndarray): The mean rewards for each arm.
    """

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray

    def __init__(self, num_arms=10, stationary=True):
        """
        Initializes the MultiArmedBandit environment.

        Args:
            num_arms (int): The number of arms for the bandit. Defaults to 10.
            stationary (bool): Whether the bandit has a stationary reward distribution. Defaults to True.
        """
        super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(num_arms)
        self.reset()

    def step(self, arm: ActType) -> tuple[ObsType, float, bool, dict]:
        """
        Takes an action by choosing an arm and returns the result of the action.

        Args:
            arm (ActType): The selected arm to pull in the bandit.

        Returns:
            obs (ObsType): The observation.
            reward (float): The reward.
            terminated (bool): Whether the episode has terminated, i.e. for non-timeout related reasons.
            truncated (bool): Whether the episode has timed out.
            info (dict): Additional information.
        """
        assert self.action_space.contains(arm)
        if not self.stationary:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.num_arms)
            self.arm_reward_means += q_drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        reward = self.np_random.normal(loc=self.arm_reward_means[arm], scale=1.0)
        obs = 0
        terminated = False
        truncated = False
        info = dict(best_arm=self.best_arm)
        return (obs, reward, terminated, truncated, info)

    def reset(self, seed: int | None = None, options=None) -> tuple[ObsType, dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int | None): The seed for random number generation. Defaults to None.
            return_info (bool): If True, return additional info. Defaults to False.
            options (dict): Additional options for environment reset. Defaults to None.

        Returns:
            obs (ObsType): The initial observation.
            info (dict): Additional information.
        """
        super().reset(seed=seed)
        if self.stationary:
            self.arm_reward_means = self.np_random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        else:
            self.arm_reward_means = np.zeros(shape=[self.num_arms])
        self.best_arm = int(np.argmax(self.arm_reward_means))

        obs = 0
        info = {}
        return obs, info

    def render(self, mode="human"):
        """
        Renders the state of the environment, in the form of a violin plot.
        """
        assert mode == "human", f"Mode {mode} not supported!"
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [np.random.normal(loc=self.arm_reward_means[arm], scale=1.0, size=1000)]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.show()


# Our environment is wrapped in a `TimeLimit` wrapper, which limits the number of steps in an episode and a `OrderEnforcing` wrapper, which enforces the order of method calls.
# 
# `register` allows to store environments in a registry so we can alter call `gym.make` to create a new instance of our environment with the necessary wrappers.

# In[13]:


max_episode_steps = 1000

gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": 10, "stationary": True},
)

env = gym.make("ArmedBanditTestbed-v0")
print(f"Our env inside its wrappers looks like: {env}")


# ## Agent Class
# 
# Here, we define the base class for agents in a multi-armed bandit environment. 
# - `get_action()` is the main method that agents need to implement. It should return the action to take in the environment. This does not take any arguments, as the agent should be able to decide on an action based on its internal state.
# - `observe()` is an optional method that agents can implement to update their internal state based on the action taken and the reward received. 
# - `reset()` is a method that agents can implement to reset their internal state at the beginning of each episode. It is run at the beginning of each episode.
# 
# Additionally, we define a helper function `run_episode()` that runs a single episode of interaction between an agent and an environment, and a function `run_agent()` that runs multiple episodes of interaction between an agent and an environment.
# 

# In[14]:


class Agent:
    """
    Base class for agents in a multi-armed bandit environment

    (you do not need to add any implementation here)
    """

    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)


def run_episode(env: gym.Env, agent: Agent, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs a single episode of interaction between an agent and an environment.

    Args:
        env (gym.Env): The environment in which the agent operates.
        agent (Agent): The agent that takes actions in the environment.
        seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
        A tuple containing arrays of rewards received in each step and a flag indicating if the chosen arm was best.
    """
    (rewards, was_best) = ([], [])

    env.reset(seed=seed)
    agent.reset(seed=seed)

    done = False
    while not done:
        arm = agent.get_action()
        obs, reward, terminated, truncated, info = env.step(arm)
        done = terminated or truncated
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_best.append(1 if arm == info["best_arm"] else 0)

    rewards = np.array(rewards, dtype=float)
    was_best = np.array(was_best, dtype=int)
    return (rewards, was_best)


def run_agent(env: gym.Env, agent: Agent, n_runs=200, base_seed=1) -> tuple[np.ndarray, np.ndarray]:
    all_rewards = []
    all_was_bests = []
    base_rng = np.random.default_rng(base_seed)
    for n in tqdm(range(n_runs)):
        seed = base_rng.integers(low=0, high=10_000, size=1).item()
        (rewards, corrects) = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(corrects)
    return np.array(all_rewards), np.array(all_was_bests)


# ## Random Agent
# 
# Next, we define a RandomAgent class that selects actions uniformly at random from the set of arms. This agent is used as a baseline to compare the performance of other agents. If later agents perform worse than the RandomAgent, then they are not learning anything useful.

# In[15]:


class RandomAgent(Agent):
    def get_action(self) -> ActType:
        return self.rng.integers(low=0, high=self.num_arms)

    def __repr__(self):
        return "RandomAgent"


# In[16]:


num_arms = 10
stationary = True
env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
agent = RandomAgent(num_arms, 0)
all_rewards, all_corrects = run_agent(env, agent)

print(f"Expected correct freq: {1/10}, actual: {all_corrects.mean():.6f}")
assert np.isclose(all_corrects.mean(), 1 / 10, atol=0.05), "Random agent is not random enough!"

print(f"Expected average reward: 0.0, actual: {all_rewards.mean():.6f}")
assert np.isclose(
    all_rewards.mean(), 0, atol=0.05
), "Random agent should be getting mean arm reward, which is zero."

print("All tests passed!")


# ## EpsilonGreedy Agent
# 
# In the EpsilonGreedy class, the `get_action` method should return the index of the arm to pull, based on the **epsilon-greedy policy**, meaning that with probability epsilon, a random arm should be chosen, and with probability 1-epsilon, the arm with the highest estimated reward should be chosen. The optimistic initialization should be used to initialize the Q values (estimated rewards) of each arm. The N array keeps track of the number of times each arm has been pulled and is initialized to zeros.

# In[17]:


class EpsilonGreedy(Agent):
    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        self.epsilon = epsilon
        self.optimism = optimism
        super().__init__(num_arms, seed)

    def get_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(low=0, high=self.num_arms).item()
        else:
            return np.argmax(self.Q)

    def observe(self, action, reward, info):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self, seed: int):
        super().reset(seed)
        self.N = np.zeros(self.num_arms)
        self.Q = np.full(self.num_arms, self.optimism, dtype=float)

    def __repr__(self):
        # For the legend, when plotting
        return f"EpsilonGreedy(eps={self.epsilon}, optimism={self.optimism})"


# In[18]:


N_RUNS = 200
num_arms = 10
stationary = True
names = []
all_rewards = []
env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)

for optimism in [0, 5]:
    agent = EpsilonGreedy(num_arms, 0, epsilon=0.01, optimism=optimism)
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    all_rewards.append(rewards)
    names.append(str(agent))
    print(agent)
    print(f" -> Frequency of correct arm: {num_correct.mean():.4f}")
    print(f" -> Average reward: {rewards.mean():.4f}")

utils.plot_rewards(all_rewards, names, moving_avg_window=15)


# ## Cheater Agent
# 
# The Cheater agent is a simple agent that always chooses the best arm by cheating and looking at the environment's best arm. This agent is useful for sanity checks and for testing the performance of other agents.

# In[19]:


class Cheater(Agent):
    def __init__(self, num_arms: int, seed: int):
        super().__init__(num_arms, seed)
        self.best_arm = 0

    def get_action(self):
        return self.best_arm

    def observe(self, action: int, reward: float, info: dict):
        self.best_arm = info["best_arm"]

    def __repr__(self):
        return "Cheater"


# In[20]:


cheater = Cheater(num_arms, 0)
reward_averaging = EpsilonGreedy(num_arms, 0, epsilon=0.1, optimism=0)
random = RandomAgent(num_arms, 0)

names = []
all_rewards = []

for agent in [cheater, reward_averaging, random]:
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    names.append(str(agent))
    all_rewards.append(rewards)

utils.plot_rewards(all_rewards, names, moving_avg_window=15)

assert (all_rewards[0] < all_rewards[1]).mean() < 0.001, "Cheater should be better than reward averaging"
print("Tests passed!")


# ## Upper Confidence Bound (UCB) Action Selection
# 
# The UCB action selection rule is defined as:
# $A_t = argmax_a Q_t(a) + c * sqrt(log(t) / N_t(a))$
# where:
# - $A_t$ is the action to take at time step $t$
# - $Q_t(a)$ is the estimated value of action a at time step $t$
# - $c$ is a hyperparameter that controls the degree of exploration
# - $N_t(a)$ is the number of times action a has been selected up to time step $t$
# 
# This is very similar to the epsilon-greedy action selection rule, but instead of choosing a random action with probability epsilon, we choose the action that maximizes the upper confidence bound.
# 
# This gives us a small improvement over the EpsilonGreedy Agent.

# In[21]:


class UCBActionSelection(Agent):
    def __init__(self, num_arms: int, seed: int, c: float, eps: float = 1e-6):
        super().__init__(num_arms, seed)
        self.c = c
        self.eps = eps

    def get_action(self):
        # This method avoids division by zero errors, and makes sure N=0 entries are chosen by argmax
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + self.eps))
        return np.argmax(ucb)

    def observe(self, action, reward, info):
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self, seed: int):
        super().reset(seed)
        self.t = 1
        self.N = np.zeros(self.num_arms)
        self.Q = np.zeros(self.num_arms)

    def __repr__(self):
        return f"UCB(c={self.c})"


# In[22]:


cheater = Cheater(num_arms, 0)
reward_averaging = EpsilonGreedy(num_arms, 0, epsilon=0.1, optimism=0)
reward_averaging_optimism = EpsilonGreedy(num_arms, 0, epsilon=0.1, optimism=5)
ucb = UCBActionSelection(num_arms, 0, c=2.0)
random = RandomAgent(num_arms, 0)

names = []
all_rewards = []

for agent in [cheater, reward_averaging, reward_averaging_optimism, ucb, random]:
    (rewards, num_correct) = run_agent(env, agent, n_runs=N_RUNS, base_seed=1)
    names.append(str(agent))
    all_rewards.append(rewards)

utils.plot_rewards(all_rewards, names, moving_avg_window=15)


# # Tabular RL and Policy Improvement

# This class is a base class for environments that can be used with reinforcement learning agents.
# It provides a common interface for interacting with the environment, and for rendering the environment state.
# The Environment class is an abstract class, and must be subclassed to provide implementations for the dynamics and render methods.
# 
# The Environment class has the following methods:
# - `__init__(num_states: int, num_actions: int, start: int = 0, terminal: Optional[np.ndarray] = None)`
# - `build() -> tuple[np.ndarray, np.ndarray]`: Constructs the T and R tensors from the dynamics of the environment.
# - `dynamics(state: int, action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: Computes the distribution over possible outcomes for a given state and action.
# - `render(pi: np.ndarray)`: Takes a policy pi, and draws an image of the behavior of that policy, if applicable.
# - `out_pad(states: np.ndarray, rewards: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: Pads the states, rewards, and probabilities to match the size of the state space.

# In[23]:


class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = np.array([], dtype=int) if terminal is None else terminal
        (self.T, self.R) = self.build()

    def build(self):
        """
        Constructs the T and R tensors from the dynamics of the environment.

        Returns:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        """
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the distribution over possible outcomes for a given state
        and action.

        Args:
            state  : int (index of state)
            action : int (index of action)

        Returns:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        """
        raise NotImplementedError()

    def render(pi: np.ndarray):
        """
        Takes a policy pi, and draws an image of the behavior of that policy, if applicable.

        Args:
            pi : (num_actions,) a policy

        Returns:
            None
        """
        raise NotImplementedError()

    def out_pad(self, states: np.ndarray, rewards: np.ndarray, probs: np.ndarray):
        """
        Args:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair

        Returns:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including zero-prob outcomes.)
        """
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return out_s, out_r, out_p


# Simple Toy environment to build intuition about creating optimal policies. 
# 
# <img src="../ref/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f63616c6c756d6d63646f7567616c6c2f636f6d7075746174696f6e616c2d7468726561642d6172742f6d61737465722f6578616d706c655f696d616765732f6d6973632f6d61726b6f762d6469616772616d2e706e67.png" width="50%">

# In[24]:


class Toy(Environment):
    def dynamics(self, state: int, action: int):
        """
        Sets up dynamics for the toy environment:
            - In state s_L, we move right & get +0 reward regardless of action
            - In state s_R, we move left & get +2 reward regardless of action
            - In state s_0, we can move left & get +1, or right & get +0
        """
        (SL, S0, SR) = (0, 1, 2)
        LEFT = 0
        assert 0 <= state < self.num_states and 0 <= action < self.num_actions
        if state == S0:
            (next_state, reward) = (SL, 1) if action == LEFT else (SR, 0)
        elif state == SL:
            (next_state, reward) = (S0, 0)
        elif state == SR:
            (next_state, reward) = (S0, 2)
        return (np.array([next_state]), np.array([reward]), np.array([1]))

    def __init__(self):
        super().__init__(num_states=3, num_actions=2)


# In[25]:


toy = Toy()

actions = ["a_L", "a_R"]
states = ["s_L", "s_0", "s_R"]

imshow(
    toy.T,  # dimensions (s, a, s_next)
    title="Transition probabilities T(s_next | s, a) for toy environment",
    facet_col=0,
    facet_labels=[f"Current state is s = {s}" for s in states],
    y=actions,
    x=states,
    labels={"x": "Next state, s_next", "y": "Action taken, a", "color": "TransitionProbability"},
    text_auto=".0f",
    border=True,
    width=850,
    height=350,
)

imshow(
    toy.R,  # dimensions (s, a, s_next)
    title="Rewards R(s, a, s_next) for toy environment",
    facet_col=0,
    facet_labels=[f"Current state is s = {s}" for s in states],
    y=actions,
    x=states,
    labels={"x": "Next state, s_next", "y": "Action taken, a", "color": "Reward"},
    text_auto=".0f",
    border=True,
    width=850,
    height=350,
)


# Norvig creates a 3x4 grid world with a start state at (1, 2) and a goal state at (1, 3).
# 
# <img src="../ref/1_oo2jt_HMxVsweRRqT1efcQ.webp" width="50%">
# 
# The agent can move up, down, left, or right, but not through walls (black squares).
# 
# The agent can learn the optimal policy using Bellman equations and value iteration.

# In[26]:


class Norvig(Environment):
    def dynamics(self, state: int, action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for i, s_new in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def render(self, pi: np.ndarray):
        assert len(pi) == self.num_states
        emoji = ["⬆️", "->", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[3] = "🟩"
        grid[7] = "🟥"
        grid[5] = "⬛"
        print("".join(grid[0:4]) + "\n" + "".join(grid[4:8]) + "\n" + "".join(grid[8:]))

    def __init__(self, penalty=-0.04):
        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.dim = (self.height, self.width)
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(num_states, num_actions, start=8, terminal=terminal)


# Example use of `render`: print out a random policy
norvig = Norvig()
pi_random = np.random.randint(0, 4, (12,))
norvig.render(pi_random)


# ## Bellman Equation
# 
# The Bellman equation states that the optimal value of a decision problem at any point equals the reward from the best immediate action plus the discounted future rewards that will result from that action.
# 
# The Bellman equation for the value function is given by:
# $$V(s) = \sum_a \pi(a|s) \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V(s')]$$
# where:
# - $V(s)$ is the value of state $s$
# - $pi(a|s)$ is the probability of taking action $a$ in state $s$
# - $T(s, a, s')$ is the probability of transitioning from state $s$ to state $s'$ under action $a$
# - $R(s, a, s')$ is the reward of transitioning from state $s$ to state $s$' under action $a$
# - $gamma$ is the discount factor
# - $V(s')$ is the value of state $s$
# 
# ### Iterative

# In[27]:


def policy_eval_numerical(env: Environment, pi: np.ndarray, gamma=0.99, eps=1e-8, max_iterations=10_000) -> np.ndarray:
    """
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Args:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
        max_iterations: int - Maximum number of iterations to run
    Outputs:
        value : float (num_states,) - The value function for policy pi
    """
    num_states = env.num_states

    value = np.zeros((num_states,))
    transition_matrix = env.T[range(num_states), pi, :]  # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]  # shape [s, s_next]

    for i in range(max_iterations):
        new_value = einops.reduce(transition_matrix * (reward_matrix + gamma * value), "s s_next -> s", "sum")

        delta = np.abs(new_value - value).max()
        if delta < eps:
            break
        value = np.copy(new_value)
    else:
        print(f"Failed to converge after {max_iterations} steps.")

    return value


# ### Exact

# In[28]:


def policy_eval_exact(env: Environment, pi: np.ndarray, gamma=0.99) -> np.ndarray:
    """
    Finds the exact solution to the Bellman equation.
    """
    num_states = env.num_states

    transition_matrix = env.T[range(num_states), pi, :]  # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]  # shape [s, s_next]

    r = einops.einsum(transition_matrix, reward_matrix, "s s_next, s s_next -> s")

    mat = np.eye(num_states) - gamma * transition_matrix

    return np.linalg.solve(mat, r)


# ## Policy Improvement
# 
# The policy_improvement function returns a new policy that is greedy with respect to the value function V from the previous iteration.

# In[29]:


def policy_improvement(env: Environment, V: np.ndarray, gamma=0.99) -> np.ndarray:
    """
    Args:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    """
    q_values_for_every_state_action_pair = einops.einsum(env.T, env.R + gamma * V, "s a s_next, s a s_next -> s a")
    pi_better = q_values_for_every_state_action_pair.argmax(axis=1)
    return pi_better


# ## Find optimal policy

# In[30]:


def find_optimal_policy(env: Environment, gamma=0.99, max_iterations=10_000):
    """
    Args:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    """
    pi = np.zeros(shape=env.num_states, dtype=int)

    for i in range(max_iterations):
        V = policy_eval_exact(env, pi, gamma)
        pi_new = policy_improvement(env, V, gamma)
        if np.array_equal(pi_new, pi):
            return pi_new
        else:
            pi = pi_new
    else:
        print(f"Failed to converge after {max_iterations} steps.")
        return pi


# In[31]:


penalty = -1
norvig = Norvig(penalty)
pi_opt = find_optimal_policy(norvig, gamma=0.99)
norvig.render(pi_opt)

