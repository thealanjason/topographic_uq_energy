import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import yaml
from pathlib import Path
from PIL import Image
from scipy import stats


def _extract_pid_from_execution_log(iteration: int) -> int:
    """
    Extract the SynxFlow process PID from execution.log.
    
    Searches for the line: "Child process '...' spawned with pid XXXXX."
    Returns the PID as an integer, or None if not found.
    """
    log_file = f'ensemble_results/iter_{iteration}/execution.log'
    
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'spawned with pid (\d+)', line)
            if match:
                return int(match.group(1))
    
    return None


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


def load_config(config_file: str = 'config.yml') -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found!")
    
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def collect_energy_data(iterations: int) -> tuple[list[float], list[dict]]:
    """
    Collect energy data from all iterations.
    
    Returns:
        Tuple of (energy_results list, iteration_data list with detailed timelines)
    """
    energy_results = []
    iteration_data = []
    
    print("--- Ensemble Total Energy Analysis (CPU + GPU) ---")
    
    for i in range(iterations):
        filename = f'ensemble_results/iter_{i}/telemetry.csv'
        
        if not os.path.exists(filename):
            continue
        
        # Extract the target process PID from execution.log
        target_pid = _extract_pid_from_execution_log(i)
        if target_pid is None:
            raise FileNotFoundError(f"Iteration {i}: Could not extract PID from execution.log. Execution log missing or PID not found.")
            
        df = pd.read_csv(filename, sep=';', dtype={'resource_id': 'str'})
        
        # 1. Extract Joules & Isolate the Process by PID
        df_gpu_raw = df[(df['metric'].str.contains('attributed_energy_gpu', na=False)) 
                        & (df['consumer_kind'] == 'process')
                        & (df['consumer_id'] == target_pid)]
        df_cpu_raw = df[(df['metric'].str.contains('attributed_energy_cpu', na=False)) 
                        & (df['consumer_kind'] == 'process')
                        & (df['consumer_id'] == target_pid)]
        
        if df_gpu_raw.empty and df_cpu_raw.empty:
            print(f"Iteration {i}: Missing CPU or GPU data for PID {target_pid}.")
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

        # 5. Timeline alignment
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
        df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()
        
        run_total = df_merged['cum_energy'].iloc[-1]
        cpu_total = df_merged['cpu_cum'].iloc[-1]
        gpu_total = df_merged['gpu_cum'].iloc[-1]
        
        energy_results.append(run_total)
        iteration_data.append({
            'iteration': i,
            'df': df_merged,
            'cpu_total': cpu_total,
            'gpu_total': gpu_total,
            'run_total': run_total,
        })
        
        print(f"Iteration {i}: {run_total:.2f} Total J [CPU: {cpu_total:.2f} J | GPU: {gpu_total:.2f} J] (Duration: {df_merged['time_sec'].iloc[-1]:.2f}s)")
    
    return energy_results, iteration_data


def plot_cumulative_energy(iteration_data: list[dict], config: dict) -> None:
    """Generate and save cumulative energy vs time plot."""
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    
    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)
    
    cumulative_plot_cfg = analysis_visualization_cfg.get('cumulative_plot', {})
    cumulative_linewidth = cumulative_plot_cfg.get('linewidth', 1.5)
    cumulative_alpha = cumulative_plot_cfg.get('alpha', 0.7)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, data in enumerate(iteration_data):
        df_merged = data['df']
        line_label = 'Monte Carlo Iterations' if idx == 0 else None
        ax.plot(df_merged['time_sec'], df_merged['cum_energy'], 
                alpha=cumulative_alpha, linewidth=cumulative_linewidth, label=line_label)
    
    ax.set_title("Cumulative Energy Consumption (CPU + GPU)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Total Energy Consumed (Joules)")
    if grid_enabled:
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    ax.legend()
    
    fig.tight_layout()
    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/cumulative_energy_consumption.png', dpi=dpi)
    plt.close(fig)
    print("Plot saved as 'cumulative_energy_consumption.png'")


def plot_energy_per_iteration(energy_results: list[float], config: dict) -> None:
    """Generate and save energy cost per iteration plot."""
    if not energy_results:
        return
    
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    
    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)
    
    iteration_plot_cfg = analysis_visualization_cfg.get('iteration_plot', {})
    scatter_color = iteration_plot_cfg.get('scatter_color', 'orange')
    scatter_edgecolor = iteration_plot_cfg.get('scatter_edgecolor', 'black')
    scatter_size = iteration_plot_cfg.get('scatter_size', 80)
    line_color = iteration_plot_cfg.get('line_color', 'orange')
    line_alpha = iteration_plot_cfg.get('line_alpha', 0.4)
    line_style = iteration_plot_cfg.get('line_style', '--')
    iter_mean_color = iteration_plot_cfg.get('mean_color', 'red')
    iter_mean_linestyle = iteration_plot_cfg.get('mean_linestyle', ':')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    iters = range(len(energy_results))
    ax.scatter(iters, energy_results, color=scatter_color, edgecolors=scatter_edgecolor, 
               s=scatter_size, zorder=3)
    ax.plot(iters, energy_results, color=line_color, alpha=line_alpha, linestyle=line_style)
    
    mean_val = np.mean(energy_results)
    ax.axhline(mean_val, color=iter_mean_color, linestyle=iter_mean_linestyle, 
               label=f'Mean: {mean_val:.2f} J')
    
    ax.set_title("Total Energy Cost per Iteration")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Final Joules")
    if grid_enabled:
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig('plots/energy_cost_per_iteration.png', dpi=dpi)
    plt.close(fig)
    print("Plot saved as 'energy_cost_per_iteration.png'")


def plot_energy_distribution(energy_results: list[float], config: dict) -> None:
    """Generate and save energy distribution plot."""
    if not energy_results:
        return
    
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    
    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)
    
    distribution_plot_cfg = analysis_visualization_cfg.get('distribution_plot', {})
    hist_color = distribution_plot_cfg.get('hist_color', 'steelblue')
    hist_edgecolor = distribution_plot_cfg.get('hist_edgecolor', 'black')
    hist_alpha = distribution_plot_cfg.get('hist_alpha', 0.8)
    kde_color = distribution_plot_cfg.get('kde_color', 'darkblue')
    kde_linewidth = distribution_plot_cfg.get('kde_linewidth', 2)
    kde_bw_method = distribution_plot_cfg.get('kde_bw_method', 0.3)
    dist_mean_color = distribution_plot_cfg.get('mean_color', 'red')
    dist_mean_linestyle = distribution_plot_cfg.get('mean_linestyle', ':')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Freedman-Diaconis rule for bin width
    q75, q25 = np.percentile(energy_results, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(energy_results) ** (1/3)) if iqr > 0 else (max(energy_results) - min(energy_results)) / 10
    bins = max(3, int(np.ceil((max(energy_results) - min(energy_results)) / bin_width)))
    
    n, bins_edges, patches = ax.hist(
        energy_results,
        bins=bins,
        color=hist_color,
        edgecolor=hist_edgecolor,
        alpha=hist_alpha,
    )

    # Add Kernel Density Estimate smooth line overlay
    ax_twin = None
    if len(energy_results) > 1:
        kde = stats.gaussian_kde(energy_results, bw_method=kde_bw_method)
        x_range = np.linspace(min(energy_results), max(energy_results), 200)
        kde_values = kde(x_range)
        # Scale KDE to match histogram height
        kde_values = kde_values * n.sum() * (bins_edges[1] - bins_edges[0])
        ax_twin = ax.twinx()
        ax_twin.plot(x_range, kde_values, color=kde_color, linewidth=kde_linewidth, 
                     label='Kernel Density Estimate')
        ax_twin.set_ylabel('Density', fontsize=10)

    mean_val = np.mean(energy_results)
    ax.axvline(mean_val, color=dist_mean_color, linestyle=dist_mean_linestyle, 
               label=f'Mean: {mean_val:.2f} J')

    ax.set_title("Distribution of Total Energy Cost")
    ax.set_xlabel("Final Joules")
    ax.set_ylabel("Count")
    if grid_enabled:
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax_twin is not None:
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend()
    
    fig.tight_layout()
    fig.savefig('plots/energy_cost_distribution.png', dpi=dpi)
    plt.close(fig)
    print("Plot saved as 'energy_cost_distribution.png'")


def print_statistics(energy_results: list[float]) -> None:
    """Print statistical summary of energy results."""
    if not energy_results:
        return
    
    energy_array = np.array(energy_results)
    mean_energy = np.mean(energy_array)
    std_dev_energy = np.std(energy_array)
    cov = (std_dev_energy / mean_energy) * 100

    print("\n--- Final Statistics ---")
    print(f"Mean Energy Cost: {mean_energy:.2f} Joules")
    print(f"Standard Deviation: ±{std_dev_energy:.2f} Joules")
    print(f"Coefficient of Variation: {cov:.2f}%")


def generate_gifs(config: dict) -> None:
    """Generate GIF animations from PNG sequences if configured."""
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    generate_gif = analysis_visualization_cfg.get('generate_gif', False)
    gif_duration_ms = analysis_visualization_cfg.get('duration_ms', 800)
    
    if generate_gif:
        _create_gif_from_pngs(Path("plots/dem_3d"), Path("plots/dem_3d.gif"), duration_ms=gif_duration_ms)
        _create_gif_from_pngs(Path("plots/water_height"), Path("plots/water_height.gif"), duration_ms=gif_duration_ms)
    else:
        print("GIF creation skipped because analysis.visualization.generate_gif is false.")


def main() -> None:
    """Main analysis orchestration function."""
    cfg = load_config('config.yml')
    iterations = cfg['monte_carlo']['iterations']
    
    # Collect energy data from all iterations
    energy_results, iteration_data = collect_energy_data(iterations)
    
    # Generate plots if data was collected
    if energy_results:
        plot_cumulative_energy(iteration_data, cfg)
        plot_energy_per_iteration(energy_results, cfg)
        plot_energy_distribution(energy_results, cfg)
        print_statistics(energy_results)
    
    # Generate GIFs
    generate_gifs(cfg)


if __name__ == "__main__":
    main()