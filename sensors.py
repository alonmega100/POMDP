import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BaseSensor:
    """Base class for all agent sensors in the GridEnv."""
    def __init__(self, name: str):
        self.name = name

    def get_observation(self, env) -> np.ndarray:
        """Computes and returns the sensor observation for the environment."""
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.Space:
        """Returns the gymnasium observation space for this sensor."""
        raise NotImplementedError

    def get_covered_cells(self, env) -> list:
        """Returns a list of coordinates (np.array of shape (2,)) covered by this sensor.
        Used for rendering and visualization.
        """
        raise NotImplementedError


class LineSensor(BaseSensor):
    """A line-of-sight sensor that looks N blocks straight ahead of the robot."""
    def __init__(self, name: str = "sensor_line", num_blocks: int = 4):
        super().__init__(name)
        self.num_blocks = num_blocks

    def get_covered_cells(self, env) -> list:
        cells = []
        heading = env._agent_heading
        for k in range(1, self.num_blocks + 1):
            cell = env._agent_location + k * heading
            cells.append(cell)
        return cells

    def get_observation(self, env) -> np.ndarray:
        # For each cell: [is_visited, is_boundary, is_target]
        obs = np.zeros((self.num_blocks, 3), dtype=np.float32)
        cells = self.get_covered_cells(env)
        for i, cell in enumerate(cells):
            r, c = cell
            if r < 0 or r >= env.size or c < 0 or c >= env.size:
                # Out of bounds
                obs[i, 1] = 1.0  # boundary
            else:
                # Inside bounds
                if env._visited_grid[r, c] == 1:
                    obs[i, 0] = 1.0  # visited
                if np.array_equal(cell, env._target_location):
                    obs[i, 2] = 1.0  # target
        return obs

    @property
    def observation_space(self) -> gym.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(self.num_blocks, 3), dtype=np.float32)


class ConeSensor(BaseSensor):
    """A cone-shaped sensor that expands outwards ahead of the robot.
    For a given depth D:
    - Distance 1: 1 cell (center)
    - Distance 2: 3 cells (center, left, right)
    - Distance d: 2d-1 cells (center, and offsets up to d-1 on both sides)
    Total number of cells observed is D^2.
    """
    def __init__(self, name: str = "sensor_cone", depth: int = 3):
        super().__init__(name)
        self.depth = depth
        self.num_cells = depth * depth

    def get_covered_cells(self, env) -> list:
        cells = []
        heading = env._agent_heading
        dr, dc = heading
        # Orthogonal left vector
        left_dir = np.array([-dc, dr])

        for d in range(1, self.depth + 1):
            center_cell = env._agent_location + d * heading
            for offset in range(-d + 1, d):
                cell = center_cell + offset * left_dir
                cells.append(cell)
        return cells

    def get_observation(self, env) -> np.ndarray:
        # For each cell: [is_visited, is_boundary, is_target]
        obs = np.zeros((self.num_cells, 3), dtype=np.float32)
        cells = self.get_covered_cells(env)
        for i, cell in enumerate(cells):
            r, c = cell
            if r < 0 or r >= env.size or c < 0 or c >= env.size:
                # Out of bounds
                obs[i, 1] = 1.0  # boundary
            else:
                # Inside bounds
                if env._visited_grid[r, c] == 1:
                    obs[i, 0] = 1.0  # visited
                if np.array_equal(cell, env._target_location):
                    obs[i, 2] = 1.0  # target
        return obs

    @property
    def observation_space(self) -> gym.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(self.num_cells, 3), dtype=np.float32)
