import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import yaml
from pathlib import Path
from PIL import Image


def _align_cumulative_energy_to_timeline(df_metric: pd.DataFrame, timeline: pd.DatetimeIndex, value_name: str) -> pd.Series:
    """
    Align a cumulative metric to a shared timeline.

    Missing values between the earlier start/end of this stream and later start/end of the other stream are returned as 0.
    Between observed samples, values are linearly interpolated on timestamp.
    After the last observed sample, the cumulative value is carried forward so the total does not drop back to zero.
    """
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    source = (
        df_metric.groupby("timestamp", as_index=True)["value"]
        .sum()
        .sort_index()
        .astype(float)
    )
    interpolation_index = pd.DatetimeIndex(source.index.union(timeline).sort_values())
    aligned = source.reindex(interpolation_index)

    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned.loc[aligned.index < first_valid] = 0.0
    aligned = aligned.interpolate(method="time", limit_area="inside")
    aligned = aligned.ffill().fillna(0.0)
    aligned = aligned.reindex(timeline)
    aligned.name = value_name
    return aligned


def _build_total_energy_timeline(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Build timestamps for attributed total energy.

    Mirrors Alumet's energy-attribution interpolation plug-in (https://github.com/alumet-dev/alumet/tree/main/plugins/energy-attribution): 
    one timeseries is the reference and remains unchanged, while other timeseries are interpolated onto its timestamps.
    CPU timestamps are the reference when CPU data exists; GPU timestamps are used only for GPU-only data.
    """
    cpu_index = pd.DatetimeIndex(pd.Index(cpu_pid["timestamp"]).unique()).sort_values()
    gpu_index = pd.DatetimeIndex(pd.Index(gpu_pid["timestamp"]).unique()).sort_values()

    if cpu_index.empty:
        return gpu_index
    return cpu_index


def _create_gif_from_pngs(image_dir: Path, output_path: Path, duration_ms: int = 800) -> None:
    """Build a GIF slideshow from PNG images in a directory."""
    image_paths = sorted(
        image_dir.glob("*.png"),
        key=lambda path: (
            int(match.group(1)) if (match := re.search(r"_iter_(\d+)", path.stem)) else float("inf"),
            path.name,
        ),
    )

    if not image_paths:
        print(f"No PNG files found in {image_dir}; skipping GIF creation.")
        return

    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            frame = img.convert("RGBA")
            frames.append(frame.copy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"GIF saved to '{output_path}' from {len(frames)} frame(s).")

# ================================================

config_file = 'config.yml'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found!")

with open(config_file, 'r') as file:
    cfg = yaml.safe_load(file)

iterations = cfg['monte_carlo']['iterations']
analysis_visualization_cfg = cfg.get('analysis', {}).get('visualization', {})
generate_gif = analysis_visualization_cfg.get('generate_gif', False)
gif_duration_ms = analysis_visualization_cfg.get('duration_ms', 800)
energy_results = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
print("--- Ensemble Total Energy Analysis (CPU + GPU) ---")

for i in range(iterations):
    filename = f'ensemble_results/iter_{i}/telemetry.csv'
    
    if not os.path.exists(filename):
        continue
        
    df = pd.read_csv(filename, sep=';')
    
    # 1. Extract Joules & Isolate the Process
    df_gpu_raw = df[(df['metric'].str.contains('attributed_energy_gpu', na=False)) & (df['consumer_kind'] == 'process')]
    df_cpu_raw = df[(df['metric'].str.contains('attributed_energy_cpu', na=False)) & (df['consumer_kind'] == 'process')]
    
    if df_gpu_raw.empty and df_cpu_raw.empty:
        print(f"Iteration {i}: Missing CPU or GPU data.")
        continue

    # We directly copy the remaining columns to prevent memory warnings.
    df_gpu_raw = df_gpu_raw[['timestamp', 'value']].copy()
    df_cpu_raw = df_cpu_raw[['timestamp', 'value']].copy()

    # 2. Format Time
    df_gpu_raw['timestamp'] = pd.to_datetime(df_gpu_raw['timestamp']).dt.floor('100ms')
    df_cpu_raw['timestamp'] = pd.to_datetime(df_cpu_raw['timestamp']).dt.floor('100ms')
    
    # 3. Squash Duplicates
    df_gpu = df_gpu_raw.groupby('timestamp', as_index=False).sum()
    df_cpu = df_cpu_raw.groupby('timestamp', as_index=False).sum()

    df_gpu = df_gpu.sort_values('timestamp')
    df_cpu = df_cpu.sort_values('timestamp')

    # 4. Cumulative Energy Calculation
    df_gpu['value'] = df_gpu['value'].cumsum()
    df_cpu['value'] = df_cpu['value'].cumsum()

    # 5. Timeline allignment
    timeline = _build_total_energy_timeline(df_cpu, df_gpu)
    
    if timeline.empty:
        continue

    cpu_aligned = _align_cumulative_energy_to_timeline(df_cpu, timeline, "cpu_cum")
    gpu_aligned = _align_cumulative_energy_to_timeline(df_gpu, timeline, "gpu_cum")

    df_merged = pd.DataFrame({
        "timestamp": timeline,
        "cpu_cum": cpu_aligned.to_numpy(),
        "gpu_cum": gpu_aligned.to_numpy(),
    })

    df_merged.dropna(subset=["cpu_cum", "gpu_cum"], inplace=True)

    if df_merged.empty:
        print(f"Iteration {i}: No overlapping CPU/GPU data. Skipping.")
        continue

    # 6. Final Energy Calculation
    df_merged['cum_energy'] = df_merged['cpu_cum'] + df_merged['gpu_cum']
    
    run_total = df_merged['cum_energy'].iloc[-1]
    cpu_total = df_merged['cpu_cum'].iloc[-1]
    gpu_total = df_merged['gpu_cum'].iloc[-1]
    energy_results.append(run_total)

    df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()
    
    print(f"Iteration {i}: {run_total:.2f} Total J [CPU: {cpu_total:.2f} J | GPU: {gpu_total:.2f} J] (Duration: {df_merged['time_sec'].iloc[-1]:.2f}s)")
    line_label = 'Monte Carlo Iterations' if i == 0 else None
    ax1.plot(df_merged['time_sec'], df_merged['cum_energy'], alpha=0.7, linewidth=1.5, label=line_label)

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

if generate_gif:
    _create_gif_from_pngs(Path("plots/dem_3d"), Path("plots/dem_3d.gif"), duration_ms=gif_duration_ms)
    _create_gif_from_pngs(Path("plots/water_height"), Path("plots/water_height.gif"), duration_ms=gif_duration_ms)
else:
    print("GIF creation skipped because analysis.visualization.generate_gif is false.")