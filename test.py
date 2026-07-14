#import gymnasium as gym
#from world import GridEnv
#import matplotlib.pyplot as plt
#
#env = GridEnv()
#obs, info = env.reset()
#
#print("Use numbers to move: 0=Right, 1=Up, 2=Left, 3=Down. Type 'q' to quit.")
#
## Initial render
#env.render()
#
#while True:
#    user_input = input("Enter action (0-3): ")
#    if user_input.lower() == 'q':
#        break
#
#    if user_input in ['0', '1', '2', '3']:
#        obs, reward, terminated, truncated, info = env.step(int(user_input))
#
#        # Re-render the grid with new positions
#        env.render()
#
#        print(f"Agent Pos: {obs['agent']} | Distance to Target: {info['distance']}")
#
#        if terminated:
#            print("Success! Your pal reached the target!")
#            plt.pause(2)  # Keep the window open for 2 seconds to celebrate
#            break
#    else:
#        print("Invalid input. Use 0, 1, 2, or 3.")
#
#env.close()