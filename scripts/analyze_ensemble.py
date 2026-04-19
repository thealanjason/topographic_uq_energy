import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

iterations = 5
energy_results = []

# Create a figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

print("--- Ensemble Total Energy Analysis (CPU + GPU) ---")

for i in range(iterations):
    filename = f'results_iter_{i}.csv'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename} - File not found.")
        continue
        
    df = pd.read_csv(filename, sep=';')
    
    # 1. Extract the Joules from Alumet
    df_gpu_raw = df[df['metric'].str.contains('attributed_energy_gpu', na=False)][['timestamp', 'value']]
    df_cpu = df[df['metric'].str.contains('attributed_energy_cpu', na=False)][['timestamp', 'value']].rename(columns={'value': 'cpu_joules'})
    
    if df_gpu_raw.empty or df_cpu.empty:
        print(f"Iteration {i}: Missing CPU or GPU data.")
        continue

    # 2. Format Time and Sort
    df_gpu_raw['timestamp'] = pd.to_datetime(df_gpu_raw['timestamp'])
    df_cpu['timestamp'] = pd.to_datetime(df_cpu['timestamp'])
    
    # Squash the 4 GPU rows per timestamp into a single unified GPU reading
    df_gpu = df_gpu_raw.groupby('timestamp', as_index=False).sum().rename(columns={'value': 'gpu_joules'})

    df_gpu = df_gpu.sort_values('timestamp')
    df_cpu = df_cpu.sort_values('timestamp')

    # 3. Cumulative Energy Calculation
    # Calculate the running total while timelines are still independent
    df_gpu['gpu_cum'] = df_gpu['gpu_joules'].cumsum()
    df_cpu['cpu_cum'] = df_cpu['cpu_joules'].cumsum()

    # Set timestamps as the index for alignment
    df_gpu.set_index('timestamp', inplace=True)
    df_cpu.set_index('timestamp', inplace=True)

    # Perform an Outer Join (Union) on the cumulative columns
    df_merged = df_cpu[['cpu_cum']].join(df_gpu[['gpu_cum']], how='outer')
    
    # Forward-fill the running totals
    # Then fill early NaNs with 0
    df_merged = df_merged.ffill().fillna(0)

    # Reset index to get the 'timestamp' column back for plotting
    df_merged = df_merged.reset_index()

    # 4. Final Energy Calculation
    # The total bill is the sum of the two running totals at any given microsecond
    df_merged['cum_energy'] = df_merged['cpu_cum'] + df_merged['gpu_cum']
    
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