import numpy as np
import os
import subprocess
from synxflow import IO
from synxflow.IO.demo_functions import get_sample_data

# --- Research Parameters ---
iterations = 5
std_dev = 0.5
log_filename = 'alumet_execution.log'

# 0. Clean up the old log file before starting a new ensemble
if os.path.exists(log_filename):
    os.remove(log_filename)

# 1. Dynamically locate the pristine baseline map
dem_file, demo_data, data_path = get_sample_data()
base_dem_path = os.path.join(data_path, 'DEM.gz')

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
    cmd = f"micromamba run -n env-model alumet-agent --config scripts/alumet-config.toml exec python scripts/gaia_flood_test.py --dem {noisy_filename} 2>&1 | tee -a {log_filename}"
    
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
    os.remove(noisy_filename)

print("\nEnsemble complete!")