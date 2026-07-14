import os
from world import GridEnv
# pyrefly: ignore [missing-import]
from sb3_contrib import RecurrentPPO
# pyrefly: ignore [missing-import]
from stable_baselines3.common.monitor import Monitor
# pyrefly: ignore [missing-import]
from stable_baselines3.common.callbacks import BaseCallback

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
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        model = RecurrentPPO.load(model_path, env=env, device=device)
    else:
        print("No existing model found. Creating new model to start training from scratch...")
        # Instantiate the RecurrentPPO agent
        # We use MultiInputLstmPolicy because the environment has a Dict observation space
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
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
