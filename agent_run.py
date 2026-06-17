import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from world import GridEnv

def print_sensor_obs(obs):
    print("\n--- Current Observations ---")
    print(f"Agent Location: {obs['agent']}")
    print(f"Target Location: {obs['target']}")
    
    # Visited memory summary
    visited_coords = np.argwhere(obs['visited_memory'] == 1.0)
    print(f"Total Unique Cells Visited: {len(visited_coords)}")
    print(f"Visited Coordinates:\n{visited_coords.tolist()}")
    
    # Line Sensor (4 blocks in front)
    # Each row is: [is_visited, is_boundary, is_target]
    print("\nLine Sensor Output (4 blocks in front):")
    print("Format: [visited, boundary, target]")
    for i, val in enumerate(obs['sensor_line']):
        print(f"  Block {i+1}: {val.tolist()}")
        
    # Cone Sensor (depth 3, 9 cells)
    print("\nCone Sensor Output (9 blocks in front cone):")
    print("Format: [visited, boundary, target]")
    for i, val in enumerate(obs['sensor_cone']):
        print(f"  Cell {i+1}: {val.tolist()}")
    print("----------------------------\n")

# 1. Initialize the environment
env = GridEnv()

# 2. Reset the environment
observation, info = env.reset()
print("--- Environment Initialized with Dict Observation Space ---")
print_sensor_obs(observation)

# Render initial state
env.render()
plt.pause(0.5)

# 3. Take random steps
for step_num in range(1, 20):
    # Sample a random action (0=Right, 1=Up, 2=Left, 3=Down)
    action = env.action_space.sample()

    # Step the environment forward
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step_num} | Action taken: {action} (0=Right, 1=Up, 2=Left, 3=Down)")
    print_sensor_obs(observation)

    # Update the plot and pause briefly to animate it
    env.render()
    plt.pause(0.8)  # Pause to make visualization visible

    if terminated:
        print("Success! Target reached!")
        plt.pause(2.0)
        break

env.close()
