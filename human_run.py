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
print("Use keys to move: w=Up, a=Left, s=Down, d=Right.")
print("Use keys to point sensor: i=Up, j=Left, k=Down, l=Right.")
print("Type 'q' to quit.")
while True:
    user_input = input("Enter action (w,a,s,d or i,j,k,l): ").lower()
    if user_input == 'q':
        break

    if user_input in ['d', 'w', 'a', 's', 'l', 'i', 'j', 'k']:
        action_map = {
            'd': 0, 'w': 1, 'a': 2, 's': 3,  # Move
            'l': 4, 'i': 5, 'j': 6, 'k': 7   # Point sensor
        }
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
        print("Invalid input. Use w,a,s,d for movement or i,j,k,l for pointing.")

env.close()