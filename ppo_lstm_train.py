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

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
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
        
        # Shared CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5),  # 10x10x1 -> 6x6x16
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3),  # 6x6x16 -> 4x4x8
            nn.ReLU(),
        )
        
        # Actor head outputs 2 channels (shape 2x2x2)
        self.actor_head = nn.Conv2d(8, 2, kernel_size=3)  # 4x4x8 -> 2x2x2
        
        # Critic head outputs 1 channel, then maps to scalar
        self.critic_head = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3),  # 4x4x8 -> 1x2x2
            nn.Flatten(),
            nn.Linear(4, 1)                  # 1x2x2 -> 1 scalar value
        )

    def apply_d4_elements(self, x):
        """
        Applies all 8 transformations of D4 to the batch of grids x.
        x shape: (B, C, H, W)
        """
        # The 4 rotations of the original grid
        r0 = x
        r1 = torch.rot90(x, k=1, dims=[-2, -1])
        r2 = torch.rot90(x, k=2, dims=[-2, -1])
        r3 = torch.rot90(x, k=3, dims=[-2, -1])
        
        # The 4 rotations of the horizontally flipped grid
        flipped_x = torch.flip(x, dims=[-1])
        f0 = flipped_x
        f1 = torch.rot90(flipped_x, k=1, dims=[-2, -1])
        f2 = torch.rot90(flipped_x, k=2, dims=[-2, -1])
        f3 = torch.rot90(flipped_x, k=3, dims=[-2, -1])
        
        return [r0, r1, r2, r3, f0, f1, f2, f3]

    def forward(self, observations):
        # 1. Extract the raw image/grid from SB3 dict. Shape: (B, H, W) or (B, C, H, W)
        x = observations[self.grid_key]
        if len(x.shape) == 3:
            x = x.unsqueeze(1) # Add channel dimension -> (B, 1, H, W)
            
        batch_size = x.shape[0]
        
        # 2. Get all 8 group transformations
        transformed_grids = self.apply_d4_elements(x)
        
        # 3. Stack and flatten into a giant batch of size (8 * B, C, H, W)
        stacked = torch.stack(transformed_grids, dim=0)
        flat_batch = stacked.view(-1, *x.shape[1:])
        
        # 4. Pass the massive batch through the CNN shared layers
        shared_features = self.cnn(flat_batch)  # Shape: (8 * B, 8, 4, 4)
        
        # 5. Pass through Actor head
        actor_out = self.actor_head(shared_features)  # Shape: (8 * B, 2, 2, 2)
        
        # 6. Reshape and isolate the 8 group actions for Actor
        actor_split = actor_out.view(8, batch_size, 2, 2, 2)
        
        # Apply inverse D4 transformations g^{-1} to each corresponding element
        y0 = actor_split[0]
        y1 = torch.rot90(actor_split[1], k=3, dims=[-2, -1])
        y2 = torch.rot90(actor_split[2], k=2, dims=[-2, -1])
        y3 = torch.rot90(actor_split[3], k=1, dims=[-2, -1])
        
        y4 = torch.flip(actor_split[4], dims=[-1])
        y5 = torch.flip(torch.rot90(actor_split[5], k=3, dims=[-2, -1]), dims=[-1])
        y6 = torch.flip(torch.rot90(actor_split[6], k=2, dims=[-2, -1]), dims=[-1])
        y7 = torch.flip(torch.rot90(actor_split[7], k=1, dims=[-2, -1]), dims=[-1])
        
        # Average the features over the 8 elements of D4 to enforce equivariance
        y_stacked = torch.stack([y0, y1, y2, y3, y4, y5, y6, y7], dim=0)
        actor_features = torch.mean(y_stacked, dim=0)  # Shape: (B, 2, 2, 2)
        
        # Flatten actor features to shape (B, 8)
        actor_flat = actor_features.reshape(batch_size, 8)
        
        # 7. Pass through Critic head
        critic_out = self.critic_head(shared_features)  # Shape: (8 * B, 1)
        
        # Reshape and isolate the 8 group actions for Critic
        critic_split = critic_out.view(8, batch_size, 1)
        
        # Average over the 8 transformations to enforce D4-invariance
        critic_flat = torch.mean(critic_split, dim=0)  # Shape: (B, 1)
        
        # 8. Concatenate actor logits and critic value
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
    def __init__(self, save_path: str = "ppo_lstm_model", save_freq: int = 2000):
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
    
    # Wrap in Monitor to track episode statistics during training
    env = Monitor(env)
    
    # Detect if Intel GPU (XPU) is available, otherwise default to CPU
    # pyrefly: ignore [missing-import]
    import torch
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Using device: {device}")

    model_path = "ppo_lstm_model.zip"
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
    save_callback = PeriodicSaveCallback(save_path="ppo_lstm_model", save_freq=2000)
    
    # Train for more steps, preserving log step count if loading model, passing callback
    model.learn(total_timesteps=100000, reset_num_timesteps=False, callback=save_callback)
    
    # Save the final trained model
    model.save("ppo_lstm_model")
    print("Training finished! Model saved to ppo_lstm_model.zip")

if __name__ == "__main__":
    train()
