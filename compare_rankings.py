import numpy as np
import os
from stable_baselines3 import PPO
from ppo_train import D4_EquivariantPolicy
from config import AVAILABLE_SENSORS, SAVED_MODELS_DIR
from world import GridEnv

def get_deterministic_action(observation, size=10):
    """
    Returns a deterministic movement action if the target is observed.
    Otherwise returns None.
    """
    # Channel 2 of visited_memory is the target location (1 if observed, 0 otherwise)
    target_grid = observation["visited_memory"][2]
    
    # Check if target is found
    target_indices = np.where(target_grid == 1.0)
    if len(target_indices[0]) == 0:
        return None  # Target not seen yet
        
    target_r, target_c = target_indices[0][0], target_indices[1][0]
    
    # Get agent position (it's normalized by size-1)
    agent_pos_normalized = observation["agent"]
    agent_r = int(round(agent_pos_normalized[0] * (size - 1)))
    agent_c = int(round(agent_pos_normalized[1] * (size - 1)))
    
    # Move towards target (Manhattan distance)
    if agent_c < target_c:
        return 0 # Move right
    elif agent_r > target_r:
        return 1 # Move up
    elif agent_c > target_c:
        return 2 # Move left
    elif agent_r < target_r:
        return 3 # Move down
        
    return None # Already at target

def evaluate_models(train_mode, sensors, num_envs=100, device="cpu"):
    loaded_models = {}
    for sensor in sensors:
        if train_mode == "target":
            filename = f"ppo_cnn_model_sensor_{sensor}.zip"
        else:
            filename = f"ppo_cnn_model_sensor_{sensor}_{train_mode}.zip"
            
        model_path = os.path.join(SAVED_MODELS_DIR, filename)
        if os.path.exists(model_path):
            loaded_models[sensor] = PPO.load(
                model_path,
                device=device,
                custom_objects={"policy_class": D4_EquivariantPolicy}
            )
        else:
            print(f"Missing model for {sensor} in mode {train_mode}: {model_path}")
            loaded_models[sensor] = None

    results = {sensor: [] for sensor in sensors if loaded_models[sensor] is not None}
    
    for env_idx in range(num_envs):
        for sensor in results.keys():
            # We always evaluate on the TARGET task reward to see who finds it best/fastest
            env = GridEnv(active_sensors=sensor, reward_mode="target")
            observation, info = env.reset(seed=env_idx)
            model = loaded_models[sensor]
            
            total_reward = 0.0
            for _ in range(60):
                # 1. Try to get deterministic action to goal
                action = get_deterministic_action(observation, env.size)
                
                # 2. If target not seen, use the RL model (explore/search)
                if action is None:
                    rl_action, _ = model.predict(observation, deterministic=True)
                    action = int(rl_action)
                    
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            results[sensor].append(total_reward)
            env.close()
            
    means = {sensor: np.mean(rewards) for sensor, rewards in results.items()}
    return means

def main():
    sensors = ["line_3", "line_4", "cone_3", "cone_4", "front_row_3_1", "front_row_5_1", "front_row_3_2", "front_row_5_2", "diagonal_3", "diagonal_4"]
    
    print("Evaluating target models (with deterministic override)...")
    target_means = evaluate_models("target", sensors)
    
    print("\nEvaluating exploration models (with deterministic override)...")
    exploration_means = evaluate_models("exploration", sensors)
    
    # Sort sensors by mean reward
    target_ranking = sorted(target_means.items(), key=lambda x: x[1], reverse=True)
    exploration_ranking = sorted(exploration_means.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Target Task Ranking ---")
    for i, (sensor, reward) in enumerate(target_ranking):
        print(f"{i+1}. {sensor} (Reward: {reward:.2f})")
        
    print("\n--- Exploration Task Ranking ---")
    for i, (sensor, reward) in enumerate(exploration_ranking):
        print(f"{i+1}. {sensor} (Reward: {reward:.2f})")
        
    print("\n--- Ranking Comparison ---")
    print(f"{'Sensor':<15} | {'Target Rank':<12} | {'Exploration Rank':<16}")
    print("-" * 50)
    
    target_ranks = {sensor: i+1 for i, (sensor, _) in enumerate(target_ranking)}
    exploration_ranks = {sensor: i+1 for i, (sensor, _) in enumerate(exploration_ranking)}
    
    for sensor in sensors:
        if sensor in target_ranks and sensor in exploration_ranks:
            print(f"{sensor:<15} | {target_ranks[sensor]:<12} | {exploration_ranks[sensor]:<16}")

if __name__ == '__main__':
    main()
