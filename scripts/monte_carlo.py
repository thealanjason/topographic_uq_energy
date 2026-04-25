import numpy as np
import os
import subprocess
import yaml
from synxflow import IO
from synxflow.IO.demo_functions import get_sample_data

# --- Load Configuration ---
config_file = 'config.yml'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found!")

with open(config_file, 'r') as file:
    cfg = yaml.safe_load(file)

# --- Research Parameters (from YAML) ---
iterations = cfg['monte_carlo']['iterations']
std_dev = cfg['monte_carlo']['std_dev']

# 0. Setup Master Output Directory
BASE_OUT_DIR = "ensemble_results"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# 1. Dynamically locate the pristine baseline map
base_dem_config = cfg['files']['baseline_dem']

if base_dem_config == 'demo':
    # Fallback to the SynXFlow built-in data
    _, _, data_path = get_sample_data()
    base_dem_path = os.path.join(data_path, 'DEM.gz')
else:
    # Use your custom file
    base_dem_path = base_dem_config

if not os.path.exists(base_dem_path):
    raise FileNotFoundError(f"Baseline DEM not found at: {base_dem_path}")

print(f"Starting Monte Carlo Ensemble: {iterations} runs, sigma={std_dev}m")

dem = IO.Raster(base_dem_path)
original_elevation = dem.array

for i in range(iterations):
    print(f"\n==========================================")
    print(f"      Running Iteration {i+1} / {iterations}      ")
    print(f"==========================================")

    # Create a dedicated folder for this specific iteration
    iter_dir = os.path.join(BASE_OUT_DIR, f"iter_{i}")
    os.makedirs(iter_dir, exist_ok=True)

    # 2. Inject Gaussian Noise
    noise = np.random.normal(0, std_dev, original_elevation.shape)
    noisy_elevation = original_elevation + noise

    # 3. Save the noisy map for this specific run
    noisy_filename = f'DEM_noisy_{i}.gz'
    dem.array = noisy_elevation 
    dem.write(noisy_filename)

    # 4. Execute the Simulation & Measurement Pipeline
    iter_log = os.path.join(iter_dir, "execution.log")
    
    cmd = f"alumet-agent --config alumet-config.toml exec micromamba run -n env-model python scripts/gaia_flood_test.py --dem {noisy_filename} --config {config_file} 2>&1 | tee {iter_log}"    
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True)

    # 5. Archive the Telemetry
    archive_csv = os.path.join(iter_dir, "telemetry.csv")
    if os.path.exists('alumet-gpu-test.csv'):
        os.rename('alumet-gpu-test.csv', archive_csv)
        print(f"Saved energy telemetry to {archive_csv}")
    else:
        print(f"WARNING: Telemetry missing for iteration {i}!")
    
    # 6. Cleanup
    os.remove(noisy_filename)

print("\nEnsemble complete!")