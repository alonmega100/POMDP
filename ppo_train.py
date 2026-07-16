#from openpyxl.cell import rich_text
from torch.nn.modules import padding
import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from gymnasium import spaces
# pyrefly: ignore [missing-import]
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from world import GridEnv
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import BaseCallback

# pyrefly: ignore [missing-import]
from stable_baselines3.common.policies import ActorCriticPolicy

class D4_Equivariant_CNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 9):
        # We output 9 features: 8 for action logits, 1 for critic value
        super().__init__(observation_space, features_dim)
        
        self.grid_key = "visited_memory" if "visited_memory" in observation_space.spaces else "image"
        grid_shape = observation_space[self.grid_key].shape
        
        if len(grid_shape) == 2:
            channels = 1
            height, width = grid_shape
        else:
            channels, height, width = grid_shape
        self.channels = channels
            
        print(f"Grid shape details - Channels: {channels}, Height: {height}, Width: {width}")
        assert height == 10, "Height must be 10 for a 10x10 grid"
        assert width == 10, "Width must be 10 for a 10x10 grid"
        self.critic_head_cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5),  # 16x6x6
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),        # 32x4x4
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),        # 32x2x2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Shared CNN layers preserving spatial relationships across 10x10 grid
        self.policy_head_cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),  # 32x10x10
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),        # 64x10x10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),        # 64x10x10
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 64),                        # Mid-size hidden layer
            nn.ReLU(),
            nn.Linear(64, 8)            
        )

        # Exact group inverse permutations P_{g^-1} for the 8 actions under D4 transformations.
        # Action indices: 0:Right, 1:Up, 2:Left, 3:Down (Move), 4:Right, 5:Up, 6:Left, 7:Down (Point)
        self.d4_inv_perms = [
            [0, 1, 2, 3, 4, 5, 6, 7],  # r0 (Identity)
            [1, 2, 3, 0, 5, 6, 7, 4],  # r1 (90 CCW)
            [2, 3, 0, 1, 6, 7, 4, 5],  # r2 (180)
            [3, 0, 1, 2, 7, 4, 5, 6],  # r3 (270 CCW)
            [2, 1, 0, 3, 6, 5, 4, 7],  # f0 (Horizontal flip across columns)
            [3, 2, 1, 0, 7, 6, 5, 4],  # f1 (90 CCW of horizontal flip)
            [0, 3, 2, 1, 4, 7, 6, 5],  # f2 (180 of horizontal flip = Vertical flip)
            [1, 0, 3, 2, 5, 4, 7, 6],  # f3 (270 CCW of horizontal flip)

        ]

    def get_d4_grids(self, x):
        """
        Takes a tensor `x` (B, C, H, W) and returns a list of 8 transformed tensors under D4.
        """
        r0 = x
        r1 = torch.rot90(x, k=1, dims=[-2, -1])
        r2 = torch.rot90(x, k=2, dims=[-2, -1])
        r3 = torch.rot90(x, k=3, dims=[-2, -1])
        
        flipped_x = torch.flip(x, dims=[-1])
        f0 = flipped_x
        f1 = torch.rot90(flipped_x, k=1, dims=[-2, -1])
        f2 = torch.rot90(flipped_x, k=2, dims=[-2, -1])
        f3 = torch.rot90(flipped_x, k=3, dims=[-2, -1])
        
        return [r0, r1, r2, r3, f0, f1, f2, f3]

    def forward(self, observations):
        # 1. Extract the raw grid from SB3 dict observation
        grid = observations[self.grid_key]

        if self.channels == 1:
            if len(grid.shape) == 2:
                grid = grid.unsqueeze(0).unsqueeze(0)
            elif len(grid.shape) == 3:
                grid = grid.unsqueeze(1)
        else:
            if len(grid.shape) == 3:
                grid = grid.unsqueeze(0)

        # 2. Get all 8 group transformations of the input grid
        d4_grids = self.get_d4_grids(grid)

        # 3. Stack and flatten into a batch of size (8 * B, C, H, W)
        stacked = torch.stack(d4_grids, dim=0)
        flat_batch = stacked.view(-1, *grid.shape[1:])

        # 4. Pass the batch through the CNN + Linear layers
        actor_features = self.policy_head_cnn(flat_batch)  # Shape: (8 * B, 8)

        # 5. Split actor logits (first 8) and critic value (last 1)
        #actor_features = shared_features[:, :8]  # Shape: (8 * B, 8)
        critic_features = self.critic_head_cnn(flat_batch)  # Shape: (8 * B, 1)

        # 6. Chunk into 8 groups for the 8 transformations
        split_actor = torch.chunk(actor_features, chunks=8, dim=0)  # 8 tensors of shape (B, 8)
        split_critic = torch.chunk(critic_features, chunks=8, dim=0)  # 8 tensors of shape (B, 1)

        # 7. Apply exact D4 inverse permutations to align direction logits
        aligned_actor = [
            split_actor[i][:, self.d4_inv_perms[i]] for i in range(8)
        ]

        # 8. Average over the 8 transformations to enforce strict equivariance/invariance
        actor_flat = torch.stack(aligned_actor).mean(dim=0)  # Shape: (B, 8) (Strictly D4-equivariant)
        critic_flat = torch.stack(split_critic).mean(dim=0)  # Shape: (B, 1) (Strictly D4-invariant)

        # 9. Concatenate actor logits and critic value
        combined = torch.cat([actor_flat, critic_flat], dim=1)  # Shape: (B, 9)
        return combined


class D4_EquivariantPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=D4_Equivariant_CNNPolicy,
            features_extractor_kwargs={"features_dim": 9},
            **kwargs
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        actor_logits = features[:, :8]
        critic_values = features[:, 8:9]
        
        distribution = self.action_dist.proba_distribution(actor_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, critic_values, log_prob

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        actor_logits = features[:, :8]
        return self.action_dist.proba_distribution(actor_logits)

    def predict_values(self, obs):
        features = self.extract_features(obs)
        critic_values = features[:, 8:9]
        return critic_values

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        actor_logits = features[:, :8]
        critic_values = features[:, 8:9]
        
        distribution = self.action_dist.proba_distribution(actor_logits)
        log_prob = distribution.log_prob(actions)
        return critic_values, log_prob, distribution.entropy()

    def _predict(self, observation, deterministic=False):
        distribution = self.get_distribution(observation)
        return distribution.get_actions(deterministic=deterministic)

class PeriodicSaveCallback(BaseCallback):
    """Custom callback to save the model periodically during training."""
    def __init__(self, save_path: str = "ppo_cnn_model", save_freq: int = 2000):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            print(f"\n[Checkpoint] Saved model to {self.save_path}.zip at step {self.num_timesteps}")
        return True

def train():
    # We initialize the environment with active_sensors=1 (ConeSensor) as active.
    # If we want to support both or any sensor configured in the env, we can.
    # Let's match the active sensor index in agent_run.py (currently set to 1 by the user)
    active_sensor_idx = 1
    print(f"Initializing GridEnv with active_sensors={active_sensor_idx} for training...")
    env = GridEnv(active_sensors=active_sensor_idx)
    
    model_path = "ppo_cnn_model.zip"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    monitor_file_path = os.path.join(log_dir, "monitor.csv")
    
    import shutil
    
    continuing = os.path.exists(model_path)
    backup_path = None
    
    if continuing:
        if os.path.exists(monitor_file_path):
            backup_path = monitor_file_path + ".backup"
            shutil.copyfile(monitor_file_path, backup_path)
            print("Backing up existing log file to append new episodes.")
    else:
        if os.path.exists(monitor_file_path):
            os.remove(monitor_file_path)
            print("No existing model found. Cleared old log files.")
            
    env = Monitor(env, monitor_file_path)
    
    if continuing and backup_path is not None:
        if os.path.exists(backup_path):
            if hasattr(env, "results_writer") and env.results_writer is not None:
                env.results_writer.file_handler.close()
            elif hasattr(env, "file_handler") and env.file_handler is not None:
                env.file_handler.close()
                
            shutil.copyfile(backup_path, monitor_file_path)
            
            new_handler = open(monitor_file_path, "a", newline="")
            if hasattr(env, "results_writer") and env.results_writer is not None:
                env.results_writer.file_handler = new_handler
                import csv
                env.results_writer.logger = csv.DictWriter(
                    new_handler,
                    fieldnames=env.results_writer.logger.fieldnames
                )
            elif hasattr(env, "file_handler") and env.file_handler is not None:
                env.file_handler = new_handler
                
            os.remove(backup_path)
            print("Successfully restored log history. New episodes will be appended.")
    
    # Detect if Intel GPU (XPU) is available, otherwise default to CPU
    # pyrefly: ignore [missing-import]

    device = "cuda" #if (hasattr(torch, "cuda") and torch.xpu.is_available()) else "cpu"
    print(f"Using device: {device}")
    assert device=="cuda", "Device aint cuda u biatch"

    model_path = "ppo_cnn_model.zip"
    model = None
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        try:
            model = PPO.load(model_path, env=env, device=device)
        except Exception as e:
            print(f"Could not load model due to architecture mismatch ({e}). Re-creating model from scratch...")
            model = None

    if model is None:
        print("Creating new model to start training from scratch...")
        # Instantiate the PPO agent with custom equivariant policy
        model = PPO(
            D4_EquivariantPolicy,
            env,
            verbose=1,
            learning_rate=1e-4,     # Good default learning rate for small discrete environments
            n_steps=256,            # Collect 256 steps per update
            batch_size=128,         # Minibatch size
            n_epochs=10,            # Number of epochs when optimizing the surrogate loss
            gamma=0.99,             # Discount factor
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,          # Encourage exploration in state space
            seed=42,
            device=device
        )

    print("Starting training of PPO agent...")
    # Instantiate the periodic save callback
    save_callback = PeriodicSaveCallback(save_path="ppo_cnn_model", save_freq=2000)
    
    # Train for more steps, preserving log step count if loading model, passing callback
    model.learn(total_timesteps=1000000, reset_num_timesteps=False, callback=save_callback)
    
    # Save the final trained model
    model.save("ppo_cnn_model")
    print("Training finished! Model saved to ppo_cnn_model.zip")

if __name__ == "__main__":
    train()
