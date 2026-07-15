# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from world import GridEnv  # Imports your custom class

# 1. Initialize the environment
env = GridEnv(active_sensors=1)

# 2. Reset the environment to start
observation, info = env.reset()
print("--- Environment Initialized ---")
print(f"Initial Agent Position: {observation['agent']}")
print(f"Target Position: {observation['target']}")
print(f"Manhattan Distance: {info['distance']}\n")
env.print_grid()

# Render initial state
env.render()

# 3. Take random steps in the world
for step_num in range(1, 60):
    # Sample a random action (0-3: Move, 4-7: Point Sensor)
    action = env.action_space.sample()

    # Step the environment forward
    observation, reward, terminated, truncated, info = env.step(action)

    action_names = {
        0: "Move Right", 1: "Move Up", 2: "Move Left", 3: "Move Down",
        4: "Point Sensor Right", 5: "Point Sensor Up", 6: "Point Sensor Left", 7: "Point Sensor Down"
    }
    action_desc = action_names.get(action, f"Unknown({action})")

    print(f"Step {step_num} | Action taken: {action} ({action_desc})")
    print(f"New Agent Position: {observation['agent']}")
    print(f"Distance to Target: {info['distance']}")
    env.print_grid()
    print("-" * 30)

    # Update the plot and pause briefly to animate it
    env.render()
    plt.pause(0.1)  # Adjust speed of the animation here (in seconds)

    if terminated:
        print("Target reached!")
        plt.pause(1.5)
        break

env.close()