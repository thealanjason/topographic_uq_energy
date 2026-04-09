import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

iterations = 50
energy_results = []

plt.figure(figsize=(12, 6))

print("--- Ensemble Energy Analysis ---")

for i in range(iterations):
    filename = f'results_iter_{i}.csv'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename} - File not found.")
        continue
        
    df = pd.read_csv(filename)
    
    # 1. Extract the Raw Metrics
    # Get total hardware power (in milliwatts)
    df_power = df[df['metric'] == 'nvml_instant_power'][['timestamp', 'resource_id', 'value']].rename(columns={'value': 'total_mw'})
    # Get process utilization (percentage 0-100)
    df_util = df[df['metric'] == 'nvml_sm_utilization'][['timestamp', 'resource_id', 'value']].rename(columns={'value': 'sm_percent'})
    
    # 2. Merge them together (Matching exact time and exact GPU)
    df_merged = pd.merge(df_power, df_util, on=['timestamp', 'resource_id'])
    
    if df_merged.empty:
        print(f"Iteration {i}: Merge failed. No overlapping data.")
        continue

    # 3. Format Time
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
    df_merged = df_merged.sort_values('timestamp')
    df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()

    # 4. The Math Engine Replacement
    # Isolated Watts = (Total Hardware Watts) * (Process Share %)
    df_merged['power_watts'] = (df_merged['total_mw'] / 1000.0) * (df_merged['sm_percent'] / 100.0)

    # 5. Integrate to Total Joules
    total_joules = np.trapezoid(df_merged['power_watts'], df_merged['time_sec'])
    energy_results.append(total_joules)
    
    print(f"Iteration {i}: {total_joules:.2f} Joules (Duration: {df_merged['time_sec'].iloc[-1]:.2f}s)")
    
    # Plot this iteration's curve
    plt.plot(df_merged['time_sec'], df_merged['power_watts'], alpha=0.5, linewidth=1.5, label=f'Run {i}')

# --- Statistical Analysis ---
if energy_results:
    energy_array = np.array(energy_results)
    mean_energy = np.mean(energy_array)
    std_dev_energy = np.std(energy_array)
    cov = (std_dev_energy / mean_energy) * 100

    print("\n--- Final Statistics ---")
    print(f"Mean Energy Cost: {mean_energy:.2f} Joules")
    print(f"Standard Deviation: ±{std_dev_energy:.2f} Joules")
    print(f"Coefficient of Variation: {cov:.2f}%")

    # --- Finalize the Graph ---
    plt.title(f"SynXFlow Energy Footprint Variance (L40S GPU)\nTopographic Noise: $\sigma$ = 0.5m | Runs: {iterations}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Isolated Process Power (Watts)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('ensemble_power_variance.png', dpi=300)
    print("\nGraph saved as 'ensemble_power_variance.png'")
else:
    print("\nNo valid energy data could be calculated.")