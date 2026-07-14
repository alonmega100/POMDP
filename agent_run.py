# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np  
from world import GridEnv

def print_sensor_obs(obs):
    print("\n--- Current Observations ---")
    # Denormalize coordinates back to 0-9 scale for readable display
    agent_loc = np.round(obs['agent'] * 9).astype(int)
    target_loc = np.round(obs['target'] * 9).astype(int)
    print(f"Agent Location: {agent_loc}")
    print(f"Target Location: {target_loc}")
    if 'heading' in obs:
        print(f"Agent Heading: {obs['heading'].tolist()}")
    
    # Visited memory summary
    visited_coords = np.argwhere(obs['visited_memory'] == 1.0)
    print(f"Total Unique Cells Visited: {len(visited_coords)}")
    print(f"Visited Coordinates:\n{visited_coords.tolist()}")
    
    # Line Sensor (4 blocks in front)
    # Each row is: [is_visited, is_boundary, is_target]
    if 'sensor_line' in obs:
        print("\nLine Sensor Output (4 blocks in front):")
        print("Format: [visited, boundary, target]")
        for i, val in enumerate(obs['sensor_line']):
            print(f"  Block {i+1}: {val.tolist()}")
        
    # Cone Sensor (depth 3, 9 cells)
    if 'sensor_cone' in obs:
        print("\nCone Sensor Output (9 blocks in front cone):")
        print("Format: [visited, boundary, target]")
        for i, val in enumerate(obs['sensor_cone']):
            print(f"  Cell {i+1}: {val.tolist()}")
    print("----------------------------\n")

# 1. Initialize the environment. You can specify a single sensor index (e.g., active_sensors=0 for LineSensor,
# active_sensors=1 for ConeSensor) or list them (e.g., active_sensors=[0, 1] or None for all sensors).
env = GridEnv(active_sensors=1)

# Load trained PPO LSTM model if available
import os
# pyrefly: ignore [missing-import]
from sb3_contrib import RecurrentPPO

def main():
    model_path = "ppo_lstm_model.zip"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = RecurrentPPO.load("ppo_lstm_model")
    else:
        print("No trained model found. Falling back to random walk baseline.")
        model = None

    # Run 5 episodes sequentially without reloading the model
    num_episodes = 5
    for ep in range(1, num_episodes + 1):
        print(f"\n==========================================")
        print(f"Starting Episode {ep} / {num_episodes}")
        print(f"==========================================\n")
        
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        # 2. Reset the environment
        observation, info = env.reset()
        print("--- Environment Initialized ---")
        print_sensor_obs(observation)

        # Render initial state
        env.render()
        plt.pause(0.5)

        # 3. Take steps
        for step_num in range(1, 61):
            if model is not None:
                # Predict action using the trained LSTM policy
                action, lstm_states = model.predict(
                    observation,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                action = int(action)
            else:
                # Sample a random action (0=Right, 1=Up, 2=Left, 3=Down)
                action = env.action_space.sample()

            # Step the environment forward
            observation, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step_num} | Action taken: {action} (0=Right, 1=Up, 2=Left, 3=Down) | Reward: {reward:.2f}")
            print_sensor_obs(observation)

            if model is not None:
                episode_starts = np.array([terminated or truncated])

            # Update the plot and pause briefly to animate it
            env.render()
            plt.pause(0.001)  # Pause to make visualization visible

            if terminated:
                print("Success! Target reached!")
                plt.pause(1.5)
                break
            if truncated:
                print("Failed! Step limit reached (60 steps).")
                plt.pause(1.5)
                break

    env.close()
if __name__ == "__main__":
    main()