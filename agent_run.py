# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

from config import ACTIVE_SENSOR, NUM_EPISODES_FOR_AGENT_RUN, SAVED_MODELS_DIR, PLAY_ON_HARD_ENVS_ONLY, REWARD_MODE
from world import GridEnv
import json
import os

def print_sensor_obs(obs):
    print("\n--- Current Observations ---")
    # Denormalize coordinates back to 0-9 scale for readable display
    agent_loc = np.round(obs['agent'] * 9).astype(int)
    target_loc = np.round(obs['target'] * 9).astype(int)
    print(f"Agent Location: {agent_loc}")
    print(f"Target Location: {target_loc}")
    if 'sensor_heading' in obs:
        print(f"Sensor Heading: {obs['sensor_heading'].tolist()}")
    
    # Visited memory summary
    visited_grid = obs['visited_memory'][0] if obs['visited_memory'].ndim == 3 else obs['visited_memory']
    visited_coords = np.argwhere(visited_grid == 1.0)
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
active_sensor = ACTIVE_SENSOR

hard_envs_list = None
if PLAY_ON_HARD_ENVS_ONLY:
    if not os.path.exists("hard_envs.json"):
        raise FileNotFoundError("PLAY_ON_HARD_ENVS_ONLY is True but 'hard_envs.json' not found. Run extract_hard_envs.py first.")
    with open("hard_envs.json", "r") as f:
        all_hard_envs = json.load(f)
    hard_envs_list = all_hard_envs.get(active_sensor, [])
    print(f"Loaded {len(hard_envs_list)} specific hard environments for sensor '{active_sensor}' for agent play.")
    if len(hard_envs_list) == 0:
        print(f"Warning: No hard environments found for '{active_sensor}'. Will fall back to random envs.")

env = GridEnv(active_sensors=active_sensor, hard_envs=hard_envs_list)

# Load trained PPO model if available
# pyrefly: ignore [missing-import]
from stable_baselines3 import PPO

def main():
    mode_tag = f"_{REWARD_MODE}"
    filename = f"ppo_cnn_model_sensor_{active_sensor}{mode_tag}.zip"
    model_path = os.path.join(SAVED_MODELS_DIR, filename)
    if not os.path.exists(model_path) and REWARD_MODE == "target":
        legacy_path = os.path.join(SAVED_MODELS_DIR, f"ppo_cnn_model_sensor_{active_sensor}.zip")
        if os.path.exists(legacy_path):
            model_path = legacy_path
    if not os.path.exists(model_path) and os.path.exists(filename):
        model_path = filename  # Fallback to root directory

    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        try:
            from ppo_train import D4_EquivariantPolicy
            model = PPO.load(model_path, custom_objects={"policy_class": D4_EquivariantPolicy})
        except Exception as e:
            print(f"Could not load model: {e}. Falling back to random walk baseline.")
            model = None
    else:
        print(f"No trained model found at {model_path}. Falling back to random walk baseline.")
        model = None

    # Run 5 episodes sequentially without reloading the model
    num_episodes = NUM_EPISODES_FOR_AGENT_RUN
    for ep in range(1, num_episodes + 1):
        print(f"\n==========================================")
        print(f"Starting Episode {ep} / {num_episodes}")
        print(f"==========================================\n")
        # 2. Reset the environment
        observation, info = env.reset()
        print("--- Environment Initialized ---")
        print_sensor_obs(observation)
        env.print_grid()

        # Render initial state
        env.render()
        plt.pause(0.5)


        # 3. Take steps
        for step_num in range(1, 61):
            if model is not None:
                # Predict action using the trained PPO policy
                action, _ = model.predict(
                    observation,
                    deterministic=True
                )
                action = int(action)
            else:
                # Sample a random action (0-3=Move, 4-7=Point Sensor)
                action = env.action_space.sample()

            # Step the environment forward
            observation, reward, terminated, truncated, info = env.step(action)

            action_names = {
                0: "Move Right", 1: "Move Up", 2: "Move Left", 3: "Move Down",
                4: "Point Sensor Right", 5: "Point Sensor Up", 6: "Point Sensor Left", 7: "Point Sensor Down"
            }
            action_desc = action_names.get(action, f"Unknown({action})")
            print(f"Step {step_num} | Action taken: {action} ({action_desc}) | Reward: {reward:.2f}")
            print_sensor_obs(observation)
            env.print_grid()

            # Update the plot and pause briefly to animate it
            env.render()
            plt.pause(0.001)  # Pause to make visualization visible

            if terminated:
                if REWARD_MODE == "exploration":
                    print("Success! Entire 10x10 map (100%) revealed!")
                else:
                    print("Success! Target reached!")
                plt.pause(1.5)
                break
            if truncated:
                coverage_pct = info.get('coverage_ratio', 0.0) * 100
                print(f"Step limit reached (60 steps). Total Coverage: {info.get('cells_discovered', '?')}/100 ({coverage_pct:.1f}%)")
                plt.pause(1.5)
                break

    env.close()
if __name__ == "__main__":
    main()
    