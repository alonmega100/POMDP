import matplotlib.pyplot as plt
import numpy as np
import os
import csv

from config import AVAILABLE_SENSORS, NUMBER_OF_ENVIRONMENTS, SAVED_MODELS_DIR
from world import GridEnv


def evaluate_sensors(num_environments, sensors=None, device="cpu"):
    """
    Generates num_environments environments and runs all sensors in each of them.
    Returns a matrix where each row is an environment, each column is a sensor,
    and the value is the total reward the sensor achieved.
    Also saves a CSV with detailed metrics and snapshots of the environments.
    """
    if sensors is None:
        sensors = list(AVAILABLE_SENSORS.keys())
        
    matrix = np.zeros((num_environments, len(sensors)))
    
    try:
        from stable_baselines3 import PPO
        from ppo_train import D4_EquivariantPolicy
        has_sb3 = True
    except ImportError:
        has_sb3 = False
        
    # Pre-load models for all sensors ONCE to avoid disk I/O inside the evaluation loop
    loaded_models = {}
    if has_sb3:
        for sensor in sensors:
            filename = f"ppo_cnn_model_sensor_{sensor}.zip"
            model_path = os.path.join(SAVED_MODELS_DIR, filename)
            if not os.path.exists(model_path) and os.path.exists(filename):
                model_path = filename
                
            if os.path.exists(model_path):
                print(f"Loading model for sensor {sensor} from '{model_path}' on device '{device}'...")
                loaded_models[sensor] = PPO.load(
                    model_path,
                    device=device,
                    custom_objects={"policy_class": D4_EquivariantPolicy}
                )
            else:
                loaded_models[sensor] = None
    
    # Create directory for environment snapshots
    snapshots_dir = "env_snapshots"
    os.makedirs(snapshots_dir, exist_ok=True)
    
    results_data = []
    
    for env_idx in range(num_environments):
        seed = env_idx  # Use env_idx as seed to ensure the same env for all sensors
        env_dict = {"env_index": env_idx}
        
        for s_idx, sensor in enumerate(sensors):
            env = GridEnv(active_sensors=sensor)
            observation, info = env.reset(seed=seed)
            
            # Save environment info & snapshot on the first sensor run (since the env is identical)
            if s_idx == 0:
                env_dict["agent_start_pos"] = (int(env._agent_location[0]), int(env._agent_location[1]))
                env_dict["target_pos"] = (int(env._target_location[0]), int(env._target_location[1]))
                
                # Render and save image of t=0
                env.render()
                plt.savefig(os.path.join(snapshots_dir, f"env_{env_idx}.jpg"), format='jpg', dpi=150)
            
            model = loaded_models.get(sensor, None)
                
            total_reward = 0.0
            steps = 0
            success = False
            
            for _ in range(60):
                steps += 1
                if model is not None:
                    action, _ = model.predict(observation, deterministic=True)
                    action = int(action)
                else:
                    action = env.action_space.sample()
                    
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated:
                    success = True
                    break
                if truncated:
                    break
                    
            env.close()
            matrix[env_idx, s_idx] = total_reward
            
            # Record detailed metrics for this sensor
            env_dict[f"sensor_{sensor}_reward"] = total_reward

            
        results_data.append(env_dict)
            
    # Save all results to a CSV file (Wide Format for Excel)
    csv_filename = f"sensor_comparison_results_{num_environments}x{len(sensors)}.csv"
    if results_data:
        keys = results_data[0].keys()
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results_data)
            
    print(f"Results saved to {csv_filename}")
    print(f"Environment snapshots saved in '{snapshots_dir}/'")
    
    return matrix

def analyze_results():

    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from statsmodels.stats.multitest import multipletests
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import combinations
    import os
    import glob

    # 1. Load your dataset
    csv_files = glob.glob('sensor_comparison_results_*.csv')
    if not csv_files:
        print("No results CSV found to analyze.")
        return
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Analyzing results from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)

    # Define sensor reward columns dynamically
    sensor_cols = [c for c in df.columns if c.startswith('sensor_') and c.endswith('_reward')]
    sensor_names = [c.replace('sensor_', '').replace('_reward', '') for c in sensor_cols]

    # ==========================================
    # STEP 1: Calculate Summary Statistics
    # ==========================================
    summary_list = []
    for col in sensor_cols:
        data = df[col]
        mean_val = data.mean()
        std_val = data.std()
        sem_val = stats.sem(data)
        # 95% Confidence Interval half-width
        ci95 = sem_val * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)

        summary_list.append({
            'Sensor': col.replace('sensor_', '').replace('_reward', ''),
            'Mean Reward': mean_val,
            'Median Reward': data.median(),
            'Std Dev': std_val,
            '95% CI (±)': ci95
        })

    summary_df = pd.DataFrame(summary_list).sort_values(by='Mean Reward', ascending=False)

    print("="*60)
    print(" 1. SENSOR PERFORMANCE SUMMARY (Ranked Best to Worst)")
    print("="*60)
    print(summary_df.to_string(index=False))

    # ==========================================
    # STEP 2: Global Test (Friedman Test)
    # ==========================================
    # Checks if ANY sensor performs significantly differently overall across the 100 seeds
    friedman_stat, friedman_p = stats.friedmanchisquare(*[df[c] for c in sensor_cols])

    print("\n" + "="*60)
    print(" 2. GLOBAL OVERALL DIFFERENCE (Friedman Test)")
    print("="*60)
    print(f"Friedman Statistic: {friedman_stat:.4f}")
    print(f"p-value:            {friedman_p:.4e}")

    if friedman_p < 0.05:
        print("Conclusion:        Significant overall difference detected (p < 0.05).")
        print("                   Proceeding to pairwise comparisons...")
    else:
        print("Conclusion:        No statistically significant difference detected between sensors.")

    # ==========================================
    # STEP 3: Pairwise Comparisons (Post-Hoc)
    # ==========================================
    # Compare all 6 possible pairs (sensor 0 vs 1, 0 vs 2, etc.)
    pairs = list(combinations(sensor_cols, 2))
    raw_p_values = []
    pair_labels = []

    for s1, s2 in pairs:
        # Wilcoxon signed-rank test for paired environments
        stat, p = stats.wilcoxon(df[s1], df[s2])
        raw_p_values.append(p)
        pair_labels.append(f"{s1.replace('sensor_', '').replace('_reward', '')} vs {s2.replace('sensor_', '').replace('_reward', '')}")

    # Apply Holm-Bonferroni correction to prevent false positives across 6 tests
    reject, p_adjusted, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')

    pairwise_df = pd.DataFrame({
        'Comparison': pair_labels,
        'Raw p-value': raw_p_values,
        'Adjusted p-value': p_adjusted,
        'Is Difference Real? (p < 0.05)': reject
    })

    print("\n" + "="*60)
    print(" 3. PAIRWISE SENSOR COMPARISONS (Wilcoxon + Holm-Bonferroni)")
    print("="*60)
    print(pairwise_df.to_string(index=False))

    # ==========================================
    # STEP 4: Visualizations
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot A: Mean Reward with 95% Confidence Intervals
    means = summary_df['Mean Reward']
    errors = summary_df['95% CI (±)']
    labels = summary_df['Sensor']

    axes[0].bar(labels, means, yerr=errors, capsize=6, color='skyblue', edgecolor='black')
    axes[0].set_ylabel('Mean Accumulated Reward')
    axes[0].set_title('Mean Reward per Sensor (with 95% Confidence Intervals)')
    # axes[0].set_ylim(bottom=df[sensor_cols].min().min() * 0.9)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True)
    axes[0].set_ylim(0,11)

    # Plot B: Boxplot showing performance distribution & consistency
    sns.boxplot(data=df[sensor_cols], ax=axes[1], palette='Set2')
    axes[1].set_xticklabels([c.replace('sensor_', '').replace('_reward', '') for c in sensor_cols])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Accumulated Reward')
    axes[1].set_ylim(-6,11)
    axes[1].grid(True)
    axes[1].set_title(f'Reward Distribution across {NUMBER_OF_ENVIRONMENTS} Environments')

    plt.tight_layout()
    plt.show()
def main():
    # print(f"Generating {NUMBER_OF_ENVIRONMENTS} environments and running sensors...")
    # results_matrix = evaluate_sensors(NUMBER_OF_ENVIRONMENTS, device="cpu")
    # print("\nResults Matrix (Rows = Environments, Columns = Sensors):")
    # print(results_matrix)
    # return results_matrix
    analyze_results()

if __name__ == "__main__":
    main()
