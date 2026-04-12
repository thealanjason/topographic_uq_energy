import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

iterations = 10
energy_results = []

plt.figure(figsize=(12, 6))

print("--- Ensemble Total Energy Analysis (CPU + GPU) ---")

for i in range(iterations):
    filename = f'results_iter_{i}.csv'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename} - File not found.")
        continue
        
    df = pd.read_csv(filename)
    
    # 1. Extract the Pre-Calculated Joules from Alumet
    df_gpu = df[df['metric'] == 'attributed_energy_gpu'][['timestamp', 'value']].rename(columns={'value': 'gpu_joules'})
    df_cpu = df[df['metric'] == 'attributed_energy_cpu'][['timestamp', 'value']].rename(columns={'value': 'cpu_joules'})
    
    if df_gpu.empty or df_cpu.empty:
        print(f"Iteration {i}: Missing CPU or GPU data.")
        continue

    # 2. Format Time and Sort (Required for interpolation)
    df_gpu['timestamp'] = pd.to_datetime(df_gpu['timestamp'])
    df_cpu['timestamp'] = pd.to_datetime(df_cpu['timestamp'])
    
    df_gpu = df_gpu.sort_values('timestamp')
    df_cpu = df_cpu.sort_values('timestamp')

    # 3. Interpolation
    # merge_asof aligns the CPU timeline to the closest GPU timestamp
    df_merged = pd.merge_asof(df_gpu, df_cpu, on='timestamp', direction='nearest')

    # 4. Total Energy Calculation 
    df_merged['total_joules'] = df_merged['gpu_joules'] + df_merged['cpu_joules']
    
    # Because Alumet outputs the Joules used per tick, Total Energy is just the sum
    run_energy = df_merged['total_joules'].sum()
    energy_results.append(run_energy)

    # 5. Math for the Graph (Converting Joules back to Watts for plotting)
    df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()
    df_merged['dt'] = df_merged['time_sec'].diff()
    
    # Power (Watts) = Joules consumed in this tick / Time of this tick
    df_merged['power_watts'] = df_merged['total_joules'] / df_merged['dt']
    
    # Clean up the first row (NaN) 
    df_merged['power_watts'] = df_merged['power_watts'].fillna(0)
    
    print(f"Iteration {i}: {run_energy:.2f} Total Joules (Duration: {df_merged['time_sec'].iloc[-1]:.2f}s)")
    
    # Plot this iteration's curve
    plt.plot(df_merged['time_sec'], df_merged['power_watts'], alpha=0.5, linewidth=1.5, label=f'Run {i}')

# --- Statistical Analysis ---
if energy_results:
    energy_array = np.array(energy_results)
    mean_energy = np.mean(energy_array)
    std_dev_energy = np.std(energy_array)
    cov = (std_dev_energy / mean_energy) * 100

    print("\n--- Final Statistics ---")
    print(f"Mean Energy Cost (CPU+GPU): {mean_energy:.2f} Joules")
    print(f"Standard Deviation: ±{std_dev_energy:.2f} Joules")
    print(f"Coefficient of Variation: {cov:.2f}%")

    # --- Finalize the Graph ---
    plt.title(f"SynXFlow Total Energy Footprint Variance (CPU + GPU)\nTopographic Noise: $\sigma$ = 0.5m | Runs: {iterations}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Combined Process Power (Watts)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('../plots/ensemble_power_variance.png', dpi=300)
    print("\nGraph saved as 'ensemble_power_variance.png'")
else:
    print("\nNo valid energy data could be calculated.")