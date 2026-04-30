import numpy as np
import os
import subprocess
import yaml
import synxflow.IO as IO
import multiprocessing
import re

# --- Load Configuration ---
config_file = 'config.yml'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found!")

with open(config_file, 'r') as file:
    cfg = yaml.safe_load(file)

# --- Research Parameters (from YAML) ---
iterations = cfg['monte_carlo']['iterations']
std_dev = cfg['monte_carlo']['std_dev']
log_filename = cfg['monte_carlo']['log_filename']

# 0. Clean up the old log file before starting a new ensemble
if os.path.exists(log_filename):
    os.remove(log_filename)

# --- Dynamic Hardware Detection ---
detected_cores = multiprocessing.cpu_count()
print(f"Hardware detected: {detected_cores} logical cores. Calibrating telemetry...")

config_path = 'alumet-config.toml'
if os.path.exists(config_path):
    with open(config_path, 'r') as file:
        config_text = file.read()

    # Safely inject the core count into the CPU formula only
    dynamic_formula = f'expr = "cpu_energy * (cpu_usage / 100.0) / {detected_cores}.0"'
    config_text = re.sub(r'expr\s*=\s*"cpu_energy[^"]+"', dynamic_formula, config_text)

    with open(config_path, 'w') as file:
        file.write(config_text)
# ----------------------------------

# 1. Locate the pristine baseline map
base_dem_path = cfg['dem']

if not os.path.exists(base_dem_path):
    raise FileNotFoundError(f"Baseline DEM not found at: {base_dem_path}")

print(f"Starting Monte Carlo Ensemble: {iterations} runs, sigma={std_dev}m")

dem = IO.Raster(base_dem_path)
original_elevation = dem.array

for i in range(iterations):
    print(f"\n==========================================")
    print(f"      Running Iteration {i+1} / {iterations}      ")
    print(f"==========================================")

    # 2. Inject Gaussian Noise
    noise = np.random.normal(0, std_dev, original_elevation.shape)
    noisy_elevation = original_elevation + noise

    # 3. Save the noisy map for this specific run
    noisy_filename = f'DEM_noisy_{i}.gz'
    dem.array = noisy_elevation 
    dem.write(noisy_filename)

    # 4. Execute the Simulation & Measurement Pipeline
    cmd = f"micromamba run -n env-model alumet-agent --config alumet-config.toml exec python scripts/flood_model.py --dem {noisy_filename} --config {config_file} 2>&1 | tee -a {log_filename}"
    
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True)

    # 5. Archive the Telemetry
    archive_csv = f'results_iter_{i}.csv'
    if os.path.exists('alumet-gpu-test.csv'):
        os.rename('alumet-gpu-test.csv', archive_csv)
        print(f"Saved energy telemetry to {archive_csv}")
    else:
        print(f"WARNING: Telemetry missing for iteration {i}!")

    # 6. Cleanup
    if os.path.exists(noisy_filename):
        os.remove(noisy_filename)

print("\nEnsemble complete!")