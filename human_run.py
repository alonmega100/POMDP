import gymnasium as gym
from world import GridEnv  # Imports your custom class
import matplotlib.pyplot as plt
# 1. Initialize the environment
env = GridEnv()

# 2. Reset the environment to start
observation, info = env.reset()
print("--- Environment Initialized ---")
print(f"Initial Agent Position: {observation['agent']}")
print(f"Target Position: {observation['target']}")
print(f"Manhattan Distance: {info['distance']}\n")

# Render the initial state
env.render()

# 3. Play the game
print("Use numbers to move: 0=Right, 1=Up, 2=Left, 3=Down. Type 'q' to quit.")
while True:
    user_input = input("Enter action (0-3): ")
    if user_input.lower() == 'q':
        break

    if user_input in ['0', '1', '2', '3']:
        obs, reward, terminated, truncated, info = env.step(int(user_input))
        print(f"Agent: {obs['agent']} | Target: {obs['target']} | Dist: {info['distance']}")

        # Update the live plot
        env.render()

        if terminated:
            print("You won!")
            # Keep the plot open for 2 seconds so you can see your victory


            plt.pause(2)
            break
    else:
        print("Invalid input. Use 0, 1, 2, or 3.")

env.close()