import random

from typing import Literal
import gymnasium as gym
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import torch as t
import einops
import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import plotly.express as px
from gymnasium.wrappers import (
    ClipAction,
    FrameStack,
    GrayScaleObservation,
    NormalizeObservation,
    NormalizeReward,
    ResizeObservation,
    TransformObservation,
    TransformReward,
)
from IPython.display import display

MAIN = __name__ == "__main__"
Arr = np.ndarray

CONFIG = dict(displayModeBar=False)


def set_global_seeds(seed):
    """Sets random seeds in several different ways (to guarantee reproducibility)"""
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True


def moving_avg(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_rewards(
    all_rewards: list[np.ndarray],
    names: list[str],
    moving_avg_window: int | None = 15,
    filename: str | None = None,
):
    # names = ["<br>" + name.replace(", ", "<br>  ").replace("(", "<br>  ").replace(")", "") + "<br>" for name in names]
    names = [name.replace(", ", "<br>  │").replace("(", "<br>  │").replace(")", "") for name in names]

    fig = go.Figure(layout=dict(template="simple_white", title_text="Mean reward over all runs"))
    for rewards, name in zip(all_rewards, names):
        rewards_avg = rewards.mean(axis=0)
        if moving_avg_window is not None:
            rewards_avg = moving_avg(rewards_avg, moving_avg_window)
        fig.add_trace(go.Scatter(y=rewards_avg, mode="lines", name=name))
    fig.update_layout(height=450, width=750).show(config=CONFIG)
    if filename is not None:
        fig.write_html(filename, config=CONFIG)


def linear_schedule(
    current_step: int,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
    total_timesteps: int,
) -> float:
    """Return the appropriate epsilon for the current step.
    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    """
    duration = exploration_fraction * total_timesteps
    slope = (end_e - start_e) / duration
    return max(slope * current_step + start_e, end_e)


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str, video_log_freq: int = 25):
    """Return a function that returns an environment after setting up boilerplate."""

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    step_trigger=lambda x: (x % video_log_freq) == 0,  # Video every 25 episodes steps for env #1
                )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# %%
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

def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    """
    Finds the exact solution to the Bellman equation.
    """
    num_states = env.num_states

    transition_matrix = env.T[range(num_states), pi, :]  # shape [s, s_next]
    reward_matrix = env.R[range(num_states), pi, :]  # shape [s, s_next]

    r = einops.einsum(transition_matrix, reward_matrix, "s s_next, s s_next -> s")

    mat = np.eye(num_states) - gamma * transition_matrix

    return np.linalg.solve(mat, r)

def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
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
    
Arr = np.ndarray
cv2.ocl.setUseOpenCL(False)


def sum_rewards(rewards: list[int], gamma: float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]:  # reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward


def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def plot_buffer_items(df, title):
    fig = px.line(df, facet_row="variable", labels={"value": "", "index": "steps"}, title=title)
    fig.update_layout(template="simple_white")
    fig.layout.annotations = []
    fig.update_yaxes(matches=None)
    fig.show()


def arg_help(args, print_df=False, filename: str | None = None):
    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = DQNArgs()
        changed_args = []
    else:
        default_args = DQNArgs()
        changed_args = [key for key in default_args.__dict__ if getattr(default_args, key) != getattr(args, key)]
    df = pd.DataFrame([ARG_HELP_STRINGS]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", ["yes" if i in changed_args else "" for i in df.index])
        with pd.option_context("max_colwidth", 0, "display.width", 150, "display.colheader_justify", "left"):
            print(df)
    else:
        s = df.style.set_table_styles(
            [
                {"selector": "td", "props": "text-align: left;"},
                {"selector": "th", "props": "text-align: left;"},
            ]
        ).apply(
            lambda row: ["background-color: red" if row.name in changed_args else None]
            + [
                None,
            ]
            * (len(row) - 1),
            axis=1,
        )
        with pd.option_context("max_colwidth", 0):
            display(s)

    if filename is not None:
        s.set_table_styles(
            [
                {
                    "selector": "",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("font-size", "0.9em"),
                        ("font-family", "system-ui"),
                        ("background-color", "#1a1a1a"),
                        ("color", "#ffffff"),
                    ],
                },
                {"selector": "th, td", "props": [("padding", "5px 10px"), ("text-align", "left")]},
                {
                    "selector": "thead tr",
                    "props": [("background-color", "#4a4a4a"), ("color", "#ffffff"), ("text-align", "left")],
                },
                {"selector": "tbody tr", "props": [("border-bottom", "1px solid #404040")]},
                {"selector": "tbody tr:nth-of-type(even)", "props": [("background-color", "#2d2d2d")]},
                {"selector": "tbody tr:last-of-type", "props": [("border-bottom", "2px solid #4a4a4a")]},
                {"selector": "tbody tr:hover", "props": [("background-color", "#363636")]},
            ]
        ).to_html(filename)


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    run_name: str,
    mode: str = "classic-control",
    video_log_freq: int = None,
    video_save_path: str = None,
    **kwargs,
):
    """
    Return a function that returns an environment after setting up boilerplate.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0 and video_log_freq:
            env = gym.wrappers.RecordVideo(
                env,
                f"{video_save_path}/{run_name}",
                episode_trigger=lambda episode_id: episode_id % video_log_freq == 0,
                disable_logger=True,
            )

        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated | truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated | truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated | truncated:
            self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames so it's important to keep lives > 0,
            # so that we only reset once the environment advertises done
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = truncated = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def prepare_atari_env(env: gym.Env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    return env


def prepare_mujoco_env(env: gym.Env):
    env = ClipAction(env)
    env = NormalizeObservation(env)
    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = NormalizeReward(env)
    env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

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

ARG_HELP_STRINGS = dict(
    seed="seed of the experiment",
    env_id="the id of the environment",
    num_envs="number of environments to run in parallel",
    # mode="can be 'classic-control' or 'atari'",
    #
    use_wandb="whether to log to weights and biases",
    wandb_project_name="the name of this experiment (also used as wandb project name)",
    wandb_entity="the entity (team) of wandb's project",
    video_log_freq="number of episodes between each video capture (None means no capture this way)",
    #
    total_timesteps="total number of steps our agent will take in total, across training",
    steps_per_train="number of sampled actions (i.e. agent steps) in between each training step",
    trains_per_target_update="the number of training steps in between each target network update",
    buffer_size="the replay memory buffer size",
    #
    batch_size="the batch size of samples from the replay memory",
    learning_rate="the learning rate of the optimizer",
    #
    gamma="the discount factor gamma",
    exploration_fraction="the fraction of `total-timesteps` it takes from start-e to go end-e",
    start_e="the starting epsilon for exploration",
    end_e="the ending epsilon for exploration",
    #
    total_training_steps="the total number of training steps (total_timesteps - buffer_size) // steps_per_train",
    # video_save_path="the path we save videos to",
)

PPO_ARG_HELP_STRINGS = dict(
    seed="seed of the experiment",
    env_id="the id of the environment",
    mode="can be 'classic-control', 'atari' or 'mujoco'",
    #
    use_wandb="if toggled, this experiment will be tracked with Weights and Biases",
    video_log_freq="if not None, we log videos this many episodes apart (so shorter episodes mean more frequent logging)",
    wandb_project_name="the name of this experiment (also used as the wandb project name)",
    wandb_entity="the entity (team) of wandb's project",
    #
    total_timesteps="total timesteps of the experiments",
    num_envs="number of synchronized vector environments in our `envs` object (this is N in the '37 Implementational Details' post)",
    num_steps_per_rollout="number of steps taken in the rollout phase (this is M in the '37 Implementational Details' post)",
    num_minibatches="the number of minibatches you divide each batch up into",
    batches_per_learning_phase="how many times you train on the full batch of data generated in each rollout phase",
    #
    lr="the learning rate of the optimizer",
    max_grad_norm="value used in gradient clipping",
    #
    gamma="the discount factor gamma",
    gae_lambda="the discount factor used in our GAE estimation",
    clip_coef="the epsilon term used in the clipped surrogate objective function",
    ent_coef="coefficient of entropy bonus term",
    vf_coef="cofficient of value loss function",
    #
    batch_size="N * M in the '37 Implementational Details' post (calculated from other values in PPOArgs)",
    minibatch_size="the size of a single minibatch we perform a gradient step on (calculated from other values in PPOArgs)",
    total_phases="total number of phases during training (calculated from other values in PPOArgs)",
    total_training_steps="total number of minibatches we will perform an update step on during training (calculated from other values in PPOArgs)",
)

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


def ppo_arg_help(args, print_df=False):

    """Prints out a nicely displayed list of arguments, their default values, and what they mean."""
    if args is None:
        args = PPOArgs()
        changed_args = []
    else:
        default_args = PPOArgs()
        # print(default_args.__dict__)
        changed_args = [key for key in default_args.__dict__ if getattr(default_args, key) != getattr(args, key)]
    df = pd.DataFrame([PPO_ARG_HELP_STRINGS]).T
    df.columns = ["description"]
    df["default value"] = [repr(getattr(args, name)) for name in df.index]
    df.index.name = "arg"
    df = df[["default value", "description"]]
    if print_df:
        df.insert(1, "changed?", ["yes" if i in changed_args else "" for i in df.index])
        with pd.option_context("max_colwidth", 0, "display.width", 150, "display.colheader_justify", "left"):
            print(df)
    else:
        s = df.style.set_table_styles(
            [
                {"selector": "td", "props": "text-align: left;"},
                {"selector": "th", "props": "text-align: left;"},
            ]
        ).apply(
            lambda row: ["background-color: red" if row.name in changed_args else None]
            + [
                None,
            ]
            * (len(row) - 1),
            axis=1,
        )
        with pd.option_context("max_colwidth", 0):
            display(s)

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