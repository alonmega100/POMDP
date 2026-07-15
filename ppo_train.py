#from openpyxl.cell import rich_text
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
from sb3_contrib import RecurrentPPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import BaseCallback

# pyrefly: ignore [missing-import]
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
# pyrefly: ignore [missing-import]
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class D4_invariant_CNNPolicy(BaseFeaturesExtractor):
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
            
        print(f"Grid shape details - Channels: {channels}, Height: {height}, Width: {width}")
        assert height == 10, "Height must be 10 for a 10x10 grid"
        assert width == 10, "Width must be 10 for a 10x10 grid"
        
        # Shared CNN layers preserving spatial relationships across 10x10 grid
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),  # 10x10x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),        # 10x10x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),        # 10x10x64
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(),
        )
        
        # Actor head outputs 8 action logits directly (4 movement + 4 pointing)
        self.actor_head = nn.Linear(128, 8)
        
        # Critic head outputs 1 scalar value
        self.critic_head = nn.Linear(128, 1)
        
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

        if len(grid.shape) == 3:
            grid = grid.unsqueeze(1)  # Add channel dimension -> (B, 1, H, W)
            
        batch_size = grid.shape[0]
        
        # 2. Get all 8 group transformations of the input grid
        d4_grids = self.get_d4_grids(grid)
        
        # 3. Stack and flatten into a batch of size (8 * B, C, H, W)
        stacked = torch.stack(d4_grids, dim=0)
        flat_batch = stacked.view(-1, *grid.shape[1:])
        
        # 4. Pass the batch through the CNN + Linear layers
        shared_features = self.cnn(flat_batch)  # Shape: (8 * B, 128)
        
        # 5. Pass through Actor head
        actor_out = self.actor_head(shared_features)  # Shape: (8 * B, 8)
        split_actor = torch.chunk(actor_out, chunks=8, dim=0)  # List of 8 tensors of shape (B, 8)
        
        # 6. Apply exact D4 inverse permutations to align direction logits with original coordinates
        aligned_actor = [
            split_actor[i][:, self.d4_inv_perms[i]] for i in range(8)
        ]
        actor_flat = torch.stack(aligned_actor).mean(dim=0)  # Shape: (B, 8) (Strictly D4-equivariant)
        
        # 7. Pass through Critic head
        critic_out = self.critic_head(shared_features)  # Shape: (8 * B, 1)
        split_critic = torch.chunk(critic_out, chunks=8, dim=0)  # List of 8 tensors of shape (B, 1)
        
        # 8. Average over the 8 transformations to enforce strict D4-invariance
        critic_flat = torch.stack(split_critic).mean(dim=0)  # Shape: (B, 1) (Strictly D4-invariant)
        
        # 9. Concatenate actor logits and critic value
        combined = torch.cat([actor_flat, critic_flat], dim=1)  # Shape: (B, 9)
        return combined


class D4_EquivariantPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=D4_invariant_CNNPolicy,
            features_extractor_kwargs={"features_dim": 9},
            **kwargs
        )

    def forward(self, obs, lstm_states=None, episode_starts=None, deterministic=False):
        features = self.extract_features(obs)
        actor_logits = features[:, :8]
        critic_values = features[:, 8:9]
        
        distribution = self.action_dist.proba_distribution(actor_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, critic_values, log_prob, lstm_states

    def get_distribution(self, obs, lstm_states=None, episode_starts=None):
        features = self.extract_features(obs)
        actor_logits = features[:, :8]
        return self.action_dist.proba_distribution(actor_logits), lstm_states

    def predict_values(self, obs, lstm_states=None, episode_starts=None):
        features = self.extract_features(obs)
        critic_values = features[:, 8:9]
        return critic_values

    def evaluate_actions(self, obs, actions, lstm_states=None, episode_starts=None):
        features = self.extract_features(obs)
        actor_logits = features[:, :8]
        critic_values = features[:, 8:9]
        
        distribution = self.action_dist.proba_distribution(actor_logits)
        log_prob = distribution.log_prob(actions)
        return critic_values, log_prob, distribution.entropy()

    def _predict(self, observation, lstm_states=None, episode_starts=None, deterministic=False):
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

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
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    # Detect if Intel GPU (XPU) is available, otherwise default to CPU
    # pyrefly: ignore [missing-import]
    import torch
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Using device: {device}")

    model_path = "ppo_cnn_model.zip"
    model = None
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        try:
            model = RecurrentPPO.load(model_path, env=env, device=device)
        except Exception as e:
            print(f"Could not load model due to architecture mismatch ({e}). Re-creating model from scratch...")
            model = None

    if model is None:
        print("Creating new model to start training from scratch...")
        # Instantiate the RecurrentPPO agent with custom equivariant policy
        model = RecurrentPPO(
            D4_EquivariantPolicy,
            env,
            verbose=1,
            learning_rate=1e-3,     # Good default learning rate for small discrete environments
            n_steps=256,            # Collect 256 steps per update
            batch_size=64,          # Sequence minibatch size
            n_epochs=10,            # Number of epochs when optimizing the surrogate loss
            gamma=0.99,             # Discount factor
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,          # Encourage exploration in partially observable space
            seed=42,
            device=device
        )

    print("Starting training of RecurrentPPO agent...")
    # Instantiate the periodic save callback
    save_callback = PeriodicSaveCallback(save_path="ppo_cnn_model", save_freq=2000)
    
    # Train for more steps, preserving log step count if loading model, passing callback
    model.learn(total_timesteps=100000, reset_num_timesteps=False, callback=save_callback)
    
    # Save the final trained model
    model.save("ppo_cnn_model")
    print("Training finished! Model saved to ppo_cnn_model.zip")

if __name__ == "__main__":
    train()
