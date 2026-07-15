import os
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

def plot_learning_curve(log_dir="logs", window=10):
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"No monitor log file found at {monitor_file}.")
        print("Make sure you have started training to generate logs.")
        return
        
    # Read the monitor log file (skip the first line header which is metadata)
    df = pd.read_csv(monitor_file, skiprows=1)
    
    if len(df) == 0:
        print("Monitor log file is empty. Keep training to log episode metrics!")
        return

    # Cumulative timesteps
    df['timesteps'] = df['l'].cumsum()
    
    plt.figure(figsize=(10, 5))
    
    # Plot raw episode rewards with lower alpha
    plt.plot(df['timesteps'], df['r'], alpha=0.3, color='dodgerblue', label='Episode Reward')
    
    # Plot rolling average reward to smooth out the learning curve
    if len(df) >= window:
        df['rolling_r'] = df['r'].rolling(window=window).mean()
        plt.plot(df['timesteps'], df['rolling_r'], color='blue', linewidth=2, label=f'Rolling Mean (last {window} eps)')
        
    plt.title("PPO Agent Learning Curve (Reward vs. Timesteps)")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episode Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Save the learning curve plot
    output_path = "learning_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Learning curve plot saved successfully to {output_path}!")

if __name__ == "__main__":
    plot_learning_curve()
