import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

iterations = 2
energy_results = []

# Create a figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

print("--- Ensemble Total Energy Analysis (CPU + GPU) ---")

for i in range(iterations):
    filename = f'results_iter_{i}.csv'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename} - File not found.")
        continue
        
    df = pd.read_csv(filename)
    
    # 1. Extract the Joules from Alumet
    df_gpu = df[df['metric'] == 'attributed_energy_gpu'][['timestamp', 'value']].rename(columns={'value': 'gpu_joules'})
    df_cpu = df[df['metric'] == 'attributed_energy_cpu'][['timestamp', 'value']].rename(columns={'value': 'cpu_joules'})
    
    if df_gpu.empty or df_cpu.empty:
        print(f"Iteration {i}: Missing CPU or GPU data.")
        continue

    # 2. Format Time and Sort
    df_gpu['timestamp'] = pd.to_datetime(df_gpu['timestamp'])
    df_cpu['timestamp'] = pd.to_datetime(df_cpu['timestamp'])
    
    df_gpu = df_gpu.sort_values('timestamp')
    df_cpu = df_cpu.sort_values('timestamp')

    # 3. Existing Interpolation Logic (preserved as requested)
    df_merged = pd.merge_asof(df_gpu, df_cpu, on='timestamp', direction='nearest')

    # 4. Cumulative Energy Calculation
    # total_joules is the energy used in that specific 100ms tick
    df_merged['total_joules'] = df_merged['gpu_joules'] + df_merged['cpu_joules']
    
    # Cumulative sum gives the total "bill" paid up to that point in time
    df_merged['cum_energy'] = df_merged['total_joules'].cumsum()
    
    # Record final total for stats
    run_total = df_merged['cum_energy'].iloc[-1]
    energy_results.append(run_total)

    # 5. Time alignment for X-axis
    df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()
    
    print(f"Iteration {i}: {run_total:.2f} Total Joules (Duration: {df_merged['time_sec'].iloc[-1]:.2f}s)")
    
    # Plot 1: Energy Curve
    ax1.plot(df_merged['time_sec'], df_merged['cum_energy'], alpha=0.7, linewidth=1.5, label=f'Run {i}')

# --- Finalize Subplot 1: Cumulative Energy vs Time ---
ax1.set_title("Cumulative Energy Consumption (CPU + GPU)")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Total Energy Consumed (Joules)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- Finalize Subplot 2: Energy vs Iterations ---
if energy_results:
    iters = range(len(energy_results))
    ax2.scatter(iters, energy_results, color='orange', edgecolors='black', s=80, zorder=3)
    ax2.plot(iters, energy_results, color='orange', alpha=0.4, linestyle='--')
    
    # Add a horizontal line for the mean
    mean_val = np.mean(energy_results)
    ax2.axhline(mean_val, color='red', linestyle=':', label=f'Mean: {mean_val:.2f} J')
    
    ax2.set_title("Total Energy Cost per Iteration")
    ax2.set_xlabel("Iteration Number")
    ax2.set_ylabel("Final Joules")
    ax2.set_xticks(iters)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

# --- Print Final Statistics ---
if energy_results:
    energy_array = np.array(energy_results)
    mean_energy = np.mean(energy_array)
    std_dev_energy = np.std(energy_array)
    cov = (std_dev_energy / mean_energy) * 100

    print("\n--- Final Statistics ---")
    print(f"Mean Energy Cost: {mean_energy:.2f} Joules")
    print(f"Standard Deviation: ±{std_dev_energy:.2f} Joules")
    print(f"Coefficient of Variation: {cov:.2f}%")

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/energy_cost_analysis.png', dpi=300)
print("\nDouble-pane plot saved as 'energy_cost_analysis.png'")