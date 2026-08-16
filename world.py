# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Optional
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from matplotlib.patches import Patch
# pyrefly: ignore [missing-import]
from matplotlib.lines import Line2D
from sensors import LineSensor, ConeSensor, FrontRowSensor, DiagonalSensor
from config import AVAILABLE_SENSORS


def create_sensor(sensor_key: str):
    """Factory function to instantiate a BaseSensor object based on string key in AVAILABLE_SENSORS."""
    if sensor_key not in AVAILABLE_SENSORS:
        raise ValueError(f"Unknown sensor key: '{sensor_key}'. Available keys: {list(AVAILABLE_SENSORS.keys())}")
    cfg = AVAILABLE_SENSORS[sensor_key]
    cls_name = cfg["class"]
    params = cfg["params"]

    if cls_name == "LineSensor":
        return LineSensor(name="sensor_line", **params)
    elif cls_name == "ConeSensor":
        return ConeSensor(name="sensor_cone", **params)
    elif cls_name == "FrontRowSensor":
        return FrontRowSensor(name="sensor_front_row", **params)
    elif cls_name == "DiagonalSensor":
        return DiagonalSensor(name="sensor_diagonal", **params)
    else:
        raise ValueError(f"Unknown sensor class name: {cls_name}")


class GridEnv(gym.Env):
    def __init__(self, sensors=None, active_sensors=None, hard_envs=None):
        super().__init__()
        self.size = 10
        self.hard_envs = hard_envs

        # Define 8 discrete actions
        self.action_space = spaces.Discrete(8)

        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }

        self._action_to_sensor_heading = {
            4: np.array([0, 1]),   # Point sensor right
            5: np.array([-1, 0]),  # Point sensor up
            6: np.array([0, -1]),  # Point sensor left
            7: np.array([1, 0]),   # Point sensor down
        }

        # Configure sensors
        if sensors is not None:
            self.sensors = sensors
            self.active_sensors = [s.name for s in sensors]
        else:
            if active_sensors is not None:
                if isinstance(active_sensors, (str, int)):
                    active_sensors = [active_sensors]

                sensor_keys = list(AVAILABLE_SENSORS.keys())
                resolved_sensors = []
                for s in active_sensors:
                    if isinstance(s, int):
                        # Convert integer index to string key for backwards compatibility
                        s = sensor_keys[s]
                    resolved_sensors.append(create_sensor(s))

                self.sensors = resolved_sensors
                self.active_sensors = list(active_sensors)
            else:
                self.sensors = [create_sensor(k) for k in AVAILABLE_SENSORS]
                self.active_sensors = list(AVAILABLE_SENSORS.keys())

        # Visited state representation (10x10 grid)
        self._visited_grid = np.zeros((self.size, self.size), dtype=np.float32)
        # Sensor heading vector (direction the sensor is pointed)
        self._sensor_heading = np.array([-1, 0])  # Defaults to Up

        # Define Gym Dict observation space
        obs_dict = {
            "agent": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "target": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "sensor_heading": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "visited_memory": spaces.Box(low=0.0, high=1.0, shape=(3, self.size, self.size), dtype=np.float32)
        }
        for sensor in self.sensors:
            obs_dict[sensor.name] = sensor.observation_space

        self.observation_space = spaces.Dict(obs_dict)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        if getattr(self, 'hard_envs', None) is not None and len(self.hard_envs) > 0:
            idx = self.np_random.integers(0, len(self.hard_envs))
            env_cfg = self.hard_envs[idx]
            self._agent_location = np.array(env_cfg['agent_start_pos'], dtype=int)
            self._target_location = np.array(env_cfg['target_pos'], dtype=int)
        else:
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

        # Reset sensor heading (defaults to UP)
        self._sensor_heading = np.array([-1, 0])

        # Scan the initial area
        self._update_visited_with_sensors()

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
        # 3x10x10 grid representation:
        # Channel 0: Was this place observed (0 if not, 1 if yes)
        # Channel 1: Is the robot here (1 at agent location, 0 otherwise)
        # Channel 2: Is the reward here (1 at target location ONLY IF observed by sensor/agent, 0 otherwise)
        grid = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # Channel 0: observed places
        grid[0] = self._visited_grid.copy()
        
        # Channel 1: robot location
        ar, ac = self._agent_location
        grid[1, ar, ac] = 1.0
        
        # Channel 2: reward location (ONLY if observed)
        tr, tc = self._target_location
        if self._visited_grid[tr, tc] == 1.0:
            grid[2, tr, tc] = 1.0
        
        obs = {
            "agent": self._agent_location.astype(np.float32) / (self.size - 1),
            "target": self._target_location.astype(np.float32) / (self.size - 1),
            "sensor_heading": self._sensor_heading.astype(np.float32),
            "visited_memory": grid
        }
        for sensor in self.sensors:
            obs[sensor.name] = sensor.get_observation(self)
        return obs

    def print_grid(self):
        obs = self._get_obs()
        grid = obs["visited_memory"]
        print("\n=== 10x10 Grid World Status (Ch0:Observed, Ch1:Robot, Ch2:Reward) ===")
        for r in range(self.size):
            row_vals = []
            for c in range(self.size):
                if grid.ndim == 3:
                    row_vals.append(f"[{int(grid[0, r, c])},{int(grid[1, r, c])},{int(grid[2, r, c])}]")
                else:
                    row_vals.append(f"{grid[r, c]:.2f}")
            print("  ".join(row_vals))
        print("=======================================================================\n")

    def _update_visited_with_sensors(self):
        for sensor in self.sensors:
            covered_cells = sensor.get_covered_cells(self)
            for cell in covered_cells:
                r, c = cell
                if 0 <= r < self.size and 0 <= c < self.size:
                    self._visited_grid[r, c] = 1.0

    def step(self, action):
        self._elapsed_steps += 1
        
        hit_wall = False
        # If action is movement (0, 1, 2, 3)
        if action in self._action_to_direction:
            direction = self._action_to_direction[action]
            next_location = self._agent_location + direction
            hit_wall = (
                next_location[0] < 0 or next_location[0] >= self.size or
                next_location[1] < 0 or next_location[1] >= self.size
            )
            self._agent_location = np.clip(next_location, 0, self.size - 1)
        # If action is point sensor (4, 5, 6, 7)
        elif action in self._action_to_sensor_heading:
            self._sensor_heading = self._action_to_sensor_heading[action]

        # Mark current agent position as visited and update seen cells with active sensors
        self._visited_grid[self._agent_location[0], self._agent_location[1]] = 1.0
        self._update_visited_with_sensors()

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = self._elapsed_steps >= 60

        if terminated:
            reward = 10.0
        elif hit_wall:
            reward = -1.0
        else:
            reward = -0.1

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

        # 5. Draw sensor heading direction (Blue Arrow)
        dy, dx = self._sensor_heading
        self.ax.arrow(self._agent_location[1], self._agent_location[0], dx * 0.3, dy * 0.3,
                      head_width=0.25, head_length=0.25, fc='dodgerblue', ec='dodgerblue', zorder=5)

        self.ax.set_title("GridWorld 10x10 with Visited Memory & Sensors")
        
        # Build legend to show Visited and Sensor Ranges
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='Agent', markerfacecolor='dodgerblue', markersize=10),
            Line2D([0], [0], marker='*', color='w', label='Target', markerfacecolor='red', markersize=12),
            Patch(facecolor='palegreen', edgecolor='none', alpha=0.35, label='Visited Memory'),
        ]
        
        # Dynamically add active sensors to the legend
        sensor_colors = ['gold', 'darkorange', 'mediumpurple', 'cyan', 'lime']
        for idx, sensor in enumerate(self.sensors):
            color = sensor_colors[idx % len(sensor_colors)]
            label_text = f"{sensor.name.replace('_', ' ').title()} FOV"
            legend_elements.append(
                Line2D([0], [0], color=color, linestyle='--', linewidth=2, label=label_text)
            )
                
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