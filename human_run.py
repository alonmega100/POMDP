# pyrefly: ignore [missing-import]
import gymnasium as gym
from world import GridEnv  # Imports your custom class
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# 1. Initialize the environment
env = GridEnv(active_sensors=1)

# 2. Reset the environment to start
observation, info = env.reset()
print("--- Environment Initialized ---")
print(f"Initial Agent Position: {observation['agent']}")
print(f"Target Position: {observation['target']}")
print(f"Manhattan Distance: {info['distance']}\n")

# Render the initial state
env.render()

# 3. Play the game
print("Use keys to move: d=Right, w=Up, a=Left, s=Down. Type 'q' to quit.")
while True:
    user_input = input("Enter action (d,w,a,s): ")
    if user_input.lower() == 'q':
        break

    if user_input in ['d', 'w', 'a', 's']:
        action_map = {'d': 0, 'w': 1, 'a': 2, 's': 3}
        obs, reward, terminated, truncated, info = env.step(action_map[user_input])
        print(f"Agent: {obs['agent']} | Target: {obs['target']} | Dist: {info['distance']}")

        # Update the live plot
        env.render()

        if terminated:
            print("You won!")
            # Keep the plot open for 2 seconds so you can see your victory


            plt.pause(2)
            break
    else:
        print("Invalid input. Use d,w,a,s.")

env.close()