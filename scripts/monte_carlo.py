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
    _, _, data_path = get_sample_data()
    base_dem_path = os.path.join(data_path, 'DEM.gz')
else:
    base_dem_path = base_dem_config

if not os.path.exists(base_dem_path):
    raise FileNotFoundError(f"Baseline DEM not found at: {base_dem_path}")

print(f"Starting Monte Carlo Ensemble: {iterations} runs, sigma={std_dev}m")

dem = IO.Raster(base_dem_path)
original_elevation = dem.array

# Path to the shared Gaia Alumet binary
alumet_bin = "/Users/share/alumet/alumet-agent"
#alumet_bin = "/Users/share/alumet/alumet-agent-v0.9.3"

for i in range(iterations):
    print(f"\n==========================================")
    print(f"      Running Iteration {i+1} / {iterations}      ")
    print(f"==========================================")

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
    python_exe = subprocess.check_output(
        "micromamba run -n env-model which python", shell=True, text=True
    ).strip()
    
    iter_log = os.path.join(iter_dir, "execution.log")
    archive_csv = os.path.join(iter_dir, "telemetry.csv")
    
    cmd = (
        f"{alumet_bin} --config alumet-config.toml "
        f"exec {python_exe} scripts/gaia_flood_test.py --dem {noisy_filename} --config {config_file} "
        f"2>&1 | tee {iter_log}"
    )
    
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True)

    # 5. Archive Telemetry
    default_alumet_output = 'alumet-output.csv' 
    
    if os.path.exists(default_alumet_output):
        os.rename(default_alumet_output, archive_csv)
        print(f"Saved energy telemetry to {archive_csv}")
    else:
        print(f"WARNING: Telemetry missing ({default_alumet_output} not found) for iteration {i}!")
    
    # 6. Cleanup
    if os.path.exists(noisy_filename):
        os.remove(noisy_filename)

print("\nEnsemble complete!")