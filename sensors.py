# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
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
        heading = env._sensor_heading
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
        heading = env._sensor_heading
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


class FrontRowSensor(BaseSensor):
    """A sensor that covers a horizontal row directly in front of the robot.
    By default, width=3 covers 1 square straight ahead and 2 diagonals (like a chess pawn).
    Width is a hyperparameter (e.g. 3, 5, 7) controlling how many squares wide the row is.
    Distance controls how many steps ahead the row is located (default 1).
    """
    def __init__(self, name: str = "sensor_front_row", width: int = 3, distance: int = 1):
        super().__init__(name)
        self.width = width
        self.distance = distance
        self.num_cells = width

    def get_covered_cells(self, env) -> list:
        cells = []
        heading = env._sensor_heading
        dr, dc = heading
        # Orthogonal left vector
        left_dir = np.array([-dc, dr])

        center_cell = env._agent_location + self.distance * heading
        half_w = self.width // 2

        for offset in range(-half_w, half_w + 1):
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


class DiagonalSensor(BaseSensor):
    """A diagonal sensor that looks D blocks along diagonal ray(s) starting from the agent (like a bishop).
    By default, it shoots D blocks up-right (forward-right relative to sensor heading).
    
    Parameters:
    - name: sensor name (default "sensor_diagonal")
    - num_blocks: depth D (number of diagonal blocks to observe)
    - direction: 'right' (forward-right), 'left' (forward-left), 'both' (V-shape), or 'all' (X-shape)
    """
    def __init__(self, name: str = "sensor_diagonal", num_blocks: int = 4, direction: str = "right"):
        super().__init__(name)
        self.num_blocks = num_blocks
        self.direction = direction.lower()
        
        if self.direction in ["right", "forward_right"]:
            self.num_cells = num_blocks
        elif self.direction in ["left", "forward_left"]:
            self.num_cells = num_blocks
        elif self.direction in ["both", "v", "v_shape"]:
            self.num_cells = 2 * num_blocks
        elif self.direction in ["all", "x", "x_shape"]:
            self.num_cells = 4 * num_blocks
        else:
            self.num_cells = num_blocks

    def get_covered_cells(self, env) -> list:
        cells = []
        heading = env._sensor_heading  # [dr, dc]
        dr, dc = heading
        left_dir = np.array([-dc, dr])
        right_dir = np.array([dc, -dr])
        
        fwd_right = heading + right_dir
        fwd_left = heading + left_dir
        bwd_right = -heading + right_dir
        bwd_left = -heading + left_dir
        
        if self.direction in ["right", "forward_right"]:
            dirs_to_check = [fwd_right]
        elif self.direction in ["left", "forward_left"]:
            dirs_to_check = [fwd_left]
        elif self.direction in ["both", "v", "v_shape"]:
            dirs_to_check = [fwd_right, fwd_left]
        elif self.direction in ["all", "x", "x_shape"]:
            dirs_to_check = [fwd_right, fwd_left, bwd_right, bwd_left]
        else:
            dirs_to_check = [fwd_right]
            
        for vec in dirs_to_check:
            for k in range(1, self.num_blocks + 1):
                cell = env._agent_location + k * vec
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


# Alias BishopSensor to DiagonalSensor for convenience



