import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from sensors import LineSensor, ConeSensor

class GridEnv(gym.Env):
    def __init__(self, sensors=None):
        super().__init__()
        self.size = 10

        # Define 4 discrete actions
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),  # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),  # Move down (row + 1)
        }

        # Configure sensors
        if sensors is None:
            self.sensors = [
                LineSensor(name="sensor_line", num_blocks=4),
                ConeSensor(name="sensor_cone", depth=3)
            ]
        else:
            self.sensors = sensors

        # Visited state representation (10x10 grid)
        self._visited_grid = np.zeros((self.size, self.size), dtype=np.float32)
        # Agent heading vector
        self._agent_heading = np.array([-1, 0])  # Defaults to Up

        # Define Gym Dict observation space
        obs_dict = {
            "agent": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=np.int32),
            "target": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=np.int32),
            "visited_memory": spaces.Box(low=0.0, high=1.0, shape=(self.size, self.size), dtype=np.float32)
        }
        for sensor in self.sensors:
            obs_dict[sensor.name] = sensor.observation_space

        self.observation_space = spaces.Dict(obs_dict)

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

        # Reset visited grid and mark current agent position as visited
        self._visited_grid = np.zeros((self.size, self.size), dtype=np.float32)
        self._visited_grid[self._agent_location[0], self._agent_location[1]] = 1.0

        # Reset heading (defaults to UP)
        self._agent_heading = np.array([-1, 0])

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
        obs = {
            "agent": self._agent_location.astype(np.int32),
            "target": self._target_location.astype(np.int32),
            "visited_memory": self._visited_grid.copy()
        }
        for sensor in self.sensors:
            obs[sensor.name] = sensor.get_observation(self)
        return obs

    def step(self, action):
        # Update heading to the direction of action
        direction = self._action_to_direction[action]
        self._agent_heading = direction

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        # Mark the new position as visited
        self._visited_grid[self._agent_location[0], self._agent_location[1]] = 1.0

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

        # 1. Draw visited cells (semi-transparent light green fill)
        for r in range(self.size):
            for c in range(self.size):
                if self._visited_grid[r, c] == 1.0:
                    # Note: x is col (c), y is row (r)
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, color='palegreen', alpha=0.35)
                    self.ax.add_patch(rect)

        # 2. Draw active sensor fields-of-view (colored dashed outlines)
        sensor_colors = ['gold', 'darkorange']
        for idx, sensor in enumerate(self.sensors):
            color = sensor_colors[idx % len(sensor_colors)]
            covered_cells = sensor.get_covered_cells(self)
            for cell in covered_cells:
                r, c = cell
                if 0 <= r < self.size and 0 <= c < self.size:
                    # Draw a slightly smaller rectangle border to look like scanning range
                    rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                         fill=False, edgecolor=color, linewidth=2, linestyle='--', alpha=0.7)
                    self.ax.add_patch(rect)

        # 3. Draw the target (Red Star)
        # Note: self._target_location is [row, col], but plot needs [col, row] (x, y)
        self.ax.plot(self._target_location[1], self._target_location[0],
                     marker='*', color='red', markersize=15, label='Target')

        # 4. Draw agent (Blue Square)
        self.ax.plot(self._agent_location[1], self._agent_location[0],
                     marker='s', color='dodgerblue', markersize=12, label='Agent')

        # 5. Draw agent heading direction (Blue Arrow)
        dy, dx = self._agent_heading
        self.ax.arrow(self._agent_location[1], self._agent_location[0], dx * 0.3, dy * 0.3,
                      head_width=0.25, head_length=0.25, fc='dodgerblue', ec='dodgerblue', zorder=5)

        self.ax.set_title("GridWorld 10x10 with Visited Memory & Sensors")
        
        # Build legend to show Visited and Sensor Ranges
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='Agent', markerfacecolor='dodgerblue', markersize=10),
            Line2D([0], [0], marker='*', color='w', label='Target', markerfacecolor='red', markersize=12),
            Patch(facecolor='palegreen', edgecolor='none', alpha=0.35, label='Visited Memory'),
            Line2D([0], [0], color='gold', linestyle='--', linewidth=2, label='Line Sensor Fov'),
            Line2D([0], [0], color='darkorange', linestyle='--', linewidth=2, label='Cone Sensor Fov')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        # Repaint without blocking the GUI thread
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)


if __name__ == "__main__":
    pass
    # env = GridEnv()
    # obs, info = env.reset()

    # print("Click the plot window, then press an arrow key (or 0=Right, 1=Up, "
    #       "2=Left, 3=Down). Press 'q' to quit.")
    # env.render()

    # key_to_action = {
    #     "right": 0, "up": 1, "left": 2, "down": 3,
    #     "0": 0, "1": 1, "2": 2, "3": 3,
    # }

    # def on_key(event):
    #     if event.key == "q":
    #         plt.close(env.fig)
    #         return

    #     if event.key not in key_to_action:
    #         return

    #     obs, reward, terminated, truncated, info = env.step(key_to_action[event.key])
    #     env.render()
    #     print(f"Agent Pos: {obs['agent']} | Distance to Target: {info['distance']}")

    #     if terminated:
    #         print("Success! Your pal reached the target!")
    #         env.ax.set_title("Success! Reached the target.")
    #         env.fig.canvas.draw_idle()

    # env.fig.canvas.mpl_connect("key_press_event", on_key)

    # # Hand control to the GUI event loop so the window stays responsive.
    # plt.ioff()
    # plt.show()