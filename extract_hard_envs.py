import pandas as pd
import glob
import os
import json
import ast

def main():
    csv_files = glob.glob('sensor_comparison_results_*.csv')
    if not csv_files:
        print("No results CSV found to analyze.")
        return
    
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Extracting hard environments from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)

    sensor_cols = [c for c in df.columns if c.startswith('sensor_') and c.endswith('_reward')]
    sensor_names = [c.replace('sensor_', '').replace('_reward', '') for c in sensor_cols]
    
    # Initialize dictionary for each sensor
    hard_envs_dict = {name: [] for name in sensor_names}
    seen_envs_dict = {name: set() for name in sensor_names}
    
    for _, row in df.iterrows():
        agent_start_pos = ast.literal_eval(row['agent_start_pos'])
        target_pos = ast.literal_eval(row['target_pos'])
        env_key = (tuple(agent_start_pos), tuple(target_pos))
        
        for col, name in zip(sensor_cols, sensor_names):
            if row[col] <= -5.9:
                if env_key not in seen_envs_dict[name]:
                    seen_envs_dict[name].add(env_key)
                    hard_envs_dict[name].append({
                        'agent_start_pos': agent_start_pos,
                        'target_pos': target_pos
                    })
                
    for name, envs in hard_envs_dict.items():
        print(f"Sensor '{name}' failed in {len(envs)} environments.")
        
    with open('hard_envs.json', 'w') as f:
        json.dump(hard_envs_dict, f, indent=4)
        
    print(f"Saved sensor-specific hard environments to 'hard_envs.json'")

if __name__ == "__main__":
    main()
