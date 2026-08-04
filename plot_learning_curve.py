import os
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

try:
    from config import ACTIVE_SENSOR, WINDOW_SIZE, AVAILABLE_SENSORS
except ImportError:
    ACTIVE_SENSOR = "cone_4"
    WINDOW_SIZE = 5000
    AVAILABLE_SENSORS = {}

SENSOR_NAMES = {k: v.get("display_name", k) for k, v in AVAILABLE_SENSORS.items()}

def plot_learning_curve(sensor_id=None, window=WINDOW_SIZE):
    if sensor_id is None:
        sensor_id = ACTIVE_SENSOR

    log_dir = os.path.join("logs", f"sensor_{sensor_id}")
    monitor_file = os.path.join(log_dir, "monitor.csv")
    
    # Fallback check for legacy root logs directory
    if not os.path.exists(monitor_file):
        legacy_file = os.path.join("logs", "monitor.csv")
        if os.path.exists(legacy_file):
            monitor_file = legacy_file

    if not os.path.exists(monitor_file):
        print(f"No monitor log file found for sensor {sensor_id} at {monitor_file}.")
        print("Make sure you have started training for this sensor to generate logs.")
        return

    # Read the monitor log file (skip the first line header which is metadata)
    df = pd.read_csv(monitor_file, skiprows=1)
    
    if len(df) == 0:
        print(f"Monitor log file for sensor {sensor_id} is empty. Keep training to log episode metrics!")
        return

    # Cumulative timesteps
    df['timesteps'] = df['l'].cumsum()
    
    plt.figure(figsize=(10, 5))
    
    sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
    
    # Plot rolling average reward to smooth out the learning curve
    smooth_window = min(window, max(1, len(df) // 5))
    if len(df) >= 2:
        df['rolling_r'] = df['r'].rolling(window=smooth_window, min_periods=1).mean()
        plt.plot(df['timesteps'], df['rolling_r'], color='blue', linewidth=2, label=f'Rolling Mean ({smooth_window} eps)')
    else:
        plt.plot(df['timesteps'], df['r'], color='blue', linewidth=2, label='Episode Reward')
        
    plt.title(f"PPO Agent Learning Curve - {sensor_name}")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episode Reward (Moving Average)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Save the learning curve plot
    output_path = f"learning_curve_sensor_{sensor_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig("learning_curve.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Learning curve plot saved successfully to {output_path} and learning_curve.png!")

def plot_all_sensors_comparison(window=WINDOW_SIZE):
    logs_base = "logs"
    if not os.path.exists(logs_base):
        return

    sensor_dirs = [
        d for d in os.listdir(logs_base) 
        if os.path.isdir(os.path.join(logs_base, d)) and d.startswith("sensor_")
    ]
    
    if not sensor_dirs:
        return

    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    plotted_any = False
    for idx, sdir in enumerate(sorted(sensor_dirs)):
        monitor_file = os.path.join(logs_base, sdir, "monitor.csv")
        if not os.path.exists(monitor_file):
            continue

        try:
            df = pd.read_csv(monitor_file, skiprows=1)
            if len(df) == 0:
                continue

            sensor_key = sdir.replace("sensor_", "")
            label_name = SENSOR_NAMES.get(sensor_key, f"Sensor {sensor_key}")

            df['timesteps'] = df['l'].cumsum()
            smooth_window = min(window, max(1, len(df) // 5))
            df['rolling_r'] = df['r'].rolling(window=smooth_window, min_periods=1).mean()
            
            color = colors[idx % len(colors)]
            plt.plot(df['timesteps'], df['rolling_r'], label=label_name, linewidth=2, color=color)
            plotted_any = True
        except Exception as e:
            print(f"Error reading {monitor_file}: {e}")

    if plotted_any:
        plt.title(f"Multi-Sensor PPO Learning Curve Comparison (window_size={WINDOW_SIZE})")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Episode Reward")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        output_path = "learning_curve_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Multi-sensor comparison plot saved successfully to {output_path}!")

if __name__ == "__main__":
    plot_learning_curve()
    plot_all_sensors_comparison()

