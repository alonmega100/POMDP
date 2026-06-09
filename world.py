import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    def __init__(self):
        super().__init__()  # Fixed: added ()
        self.size = 10  # Fixed: changed self.grid_size to self.size

        # Define 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Fixed: Map integer actions (0, 1, 2, 3) to movements
        self._action_to_direction = {
            0: np.array([0, 1]),  # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),  # Move down (row + 1)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomly place the agent
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False

        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        # Initialize the plot on the first render call
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()  # Turn on interactive mode for live updates
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()

        # Set up grid boundaries and gridlines
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_xticks(np.arange(self.size))
        self.ax.set_yticks(np.arange(self.size))
        self.ax.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5)

        # Invert Y-axis so row 0 is at the top (standard for grid worlds)
        self.ax.invert_yaxis()

        # Draw the target (Red Star)
        # Note: self._target_location is [row, col], but plot needs [col, row] (x, y)
        self.ax.plot(self._target_location[1], self._target_location[0],
                     marker='*', color='red', markersize=15, label='Target')

        # Draw your agent pal (Blue Square)
        self.ax.plot(self._agent_location[1], self._agent_location[0],
                     marker='s', color='dodgerblue', markersize=12, label='Agent')

        self.ax.set_title("GridWorld 20x20")
        self.ax.legend(loc='upper right')

        # Pause briefly so the window has time to draw/refresh
        plt.draw()
        plt.pause(0.1)

    def close(self):
        if hasattr(self, 'fig') and self.fig is None:
            plt.close(self.fig)