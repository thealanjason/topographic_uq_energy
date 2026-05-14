import numpy as np
import os
import subprocess
import tomllib
import yaml
import rasterio

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

# 1. Locate the pristine baseline map
base_dem_path = cfg['dem']

if not os.path.exists(base_dem_path):
    raise FileNotFoundError(f"Baseline DEM not found at: {base_dem_path}")

print(f"Starting Monte Carlo Ensemble: {iterations} runs, sigma={std_dev}m")

abs_dem_path = os.path.abspath(base_dem_path)
vfs_path = f'/vsigzip/{abs_dem_path}' if base_dem_path.endswith('.gz') else base_dem_path

# Read the map using Rasterio
with rasterio.open(vfs_path) as src:
    original_elevation = src.read(1)  # Read the first band (elevation data)
    dem_meta = src.meta.copy()  # Save the coordinates/projection data for later

# Force the output to be a standard TIFF to avoid compression write errors
dem_meta.update(driver='GTiff')

# Path to the Alumet binary (from config)
alumet_bin = cfg['monte_carlo']['alumet_bin']

if not os.path.exists(alumet_bin):
    raise FileNotFoundError(f"Alumet binary not found at: {alumet_bin}")

# Read the CSV output name from Alumet's own config so the filename stays single-sourced.
with open('alumet-config.toml', 'rb') as file:
    alumet_cfg = tomllib.load(file)

default_alumet_output = alumet_cfg['plugins']['csv']['output_path']

for i in range(iterations):
    print(f"\n==========================================")
    print(f"      Running Iteration {i+1} / {iterations}      ")
    print(f"==========================================")

    iter_dir = os.path.join(BASE_OUT_DIR, f"iter_{i}")
    os.makedirs(iter_dir, exist_ok=True)

    # 2. Inject Gaussian Noise
    noise = np.random.normal(0, std_dev, original_elevation.shape)
    noisy_elevation = original_elevation + noise

    nodata_val = dem_meta.get('nodata')
    if nodata_val is not None:
        # Revert any no-data pixels back to their exact original value
        noisy_elevation[original_elevation == nodata_val] = nodata_val

    # 3. Save the noisy map using Rasterio (as a safe .tif)
    noisy_filename = f'DEM_noisy_{i}.tif'
    with rasterio.open(noisy_filename, 'w', **dem_meta) as dst:
        dst.write(noisy_elevation.astype(dem_meta['dtype']), 1)

    # 4. Execute the Simulation & Measurement Pipeline
    python_exe = subprocess.check_output(
        "micromamba run -n env-model which python", shell=True, text=True
    ).strip()
    
    iter_log = os.path.join(iter_dir, "execution.log")
    archive_csv = os.path.join(iter_dir, "telemetry.csv")

    # Remove any leftover telemetry from a previous run before starting a new iteration.
    if os.path.exists(default_alumet_output):
        os.remove(default_alumet_output)
    
    cmd = (
        f"{alumet_bin} --config alumet-config.toml "
        f"exec {python_exe} scripts/flood_model.py --dem {noisy_filename} --config {config_file} "
        f"2>&1 | tee {iter_log}"
    )
    
    print(f"Executing: {cmd}")
    # Use bash -o pipefail to catch failures in the pipeline, and check=True to raise on failure
    subprocess.run(f"bash -o pipefail -c \"{cmd}\"", shell=True, check=True)

    # 5. Archive Telemetry
    if os.path.exists(default_alumet_output):
        os.rename(default_alumet_output, archive_csv)
        print(f"Saved energy telemetry to {archive_csv}")
    else:
        print(f"WARNING: Telemetry missing ({default_alumet_output} not found) for iteration {i}!")
    
    # 6. Cleanup
    if os.path.exists(noisy_filename):
        os.remove(noisy_filename)

print("\nEnsemble complete!")