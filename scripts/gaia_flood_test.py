import argparse
import os
import numpy as np
import pandas as pd
import yaml
from synxflow import IO, flood

# --- 0. Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="Run SynXFlow Flood Simulation")
parser.add_argument('--dem', type=str, default=None, help='Path to the DEM file to use')
parser.add_argument('--config', type=str, default='config.yml', help='Path to the YAML config file')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"Configuration file {args.config} not found!")

with open(args.config, 'r') as file:
    cfg = yaml.safe_load(file)

# 1. Setup Data Paths
# Check if the orchestrator passed a custom noisy DEM.
if args.dem and os.path.exists(args.dem):
    target_dem_path = args.dem
    print(f"Loading NOISY DEM: {target_dem_path}")
else:
    # Directly read the baseline DEM path from the config
    target_dem_path = cfg['files']['baseline_dem']
    print(f"Loading BASELINE DEM: {target_dem_path}")

DEM = IO.Raster(target_dem_path)

current_mean = np.nanmean(DEM.array)
print(f"--> VERIFICATION: Map loaded successfully.")
print(f"--> VERIFICATION: Mean Elevation is {current_mean:.6f} meters")

case_folder = os.path.join(os.getcwd(), 'gaia_flood_case')
case_input = IO.InputModel(DEM, num_of_sections=1, case_folder=case_folder)

# 2. Add Water Sources (Discharge and Rain)
box_upstream = np.array(cfg['boundaries']['box_upstream'])
box_downstream = np.array(cfg['boundaries']['box_downstream'])
discharge_values = np.array(cfg['boundaries']['discharge_values'])
downstream_h = np.array(cfg['boundaries']['downstream_h'])

bound_list = [
    {'polyPoints': box_upstream, 'type': 'open', 'hU': discharge_values},
    {'polyPoints': box_downstream, 'type': 'open', 'h': downstream_h}
]
case_input.set_boundary_condition(boundary_list=bound_list)

# Directly read the rain files from the config
rain_mask_path = cfg['files']['rain_mask']
rain_source_path = cfg['files']['rain_source']

rain_mask = IO.Raster(rain_mask_path)
rain_source = pd.read_csv(rain_source_path, header=None).to_numpy()
case_input.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)

# 3. Add Friction
# Read the landcover file from the config
landcover_path = cfg['files']['landcover']
landcover = IO.Raster(landcover_path)

case_input.set_landcover(landcover)
case_input.set_grid_parameter(manning={
    'param_value': cfg['friction']['param_value'], 
    'land_value': cfg['friction']['land_value'], 
    'default_value': cfg['friction']['default_value']
})

# 4. Settings and Execution
# Pull simulation settings from the YAML config
case_input.set_initial_condition('h0', cfg['settings']['h0'])
case_input.set_gauges_position(np.array(cfg['settings']['gauges_position']))
case_input.set_runtime(cfg['settings']['runtime'])

print("Writing files and starting GPU simulation...")
case_input.write_input_files()
flood.run(case_folder)
print("Simulation complete.")