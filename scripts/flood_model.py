import argparse
import os
import warnings
import sys
from pathlib import Path

# Suppress numpy/matplotlib compatibility warnings early
warnings.filterwarnings('ignore')

# Patch numpy.warnings if it doesn't exist to prevent AttributeError
import numpy as np
if not hasattr(np, 'warnings'):
    import warnings as _warnings_module
    np.warnings = _warnings_module

import pandas as pd
import yaml
import copy
from synxflow import IO, flood
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #For Pandas Warnings from SynXflow

# --- 0. Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="Run SynXFlow Flood Simulation")
parser.add_argument('--dem', type=str, default=None, help='Path to the DEM file to use')
parser.add_argument('--config', type=str, default='config.yml', help='Path to the YAML config file')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"Configuration file {args.config} not found!")

with open(args.config, 'r') as file:
    cfg = yaml.safe_load(file)

project_root = os.path.dirname(os.path.abspath(args.config))

# 1. Get DEM Path. Prefer path provided as arguement
if args.dem and os.path.exists(args.dem):
    target_dem_path = args.dem
    print(f"Loading DEM: {target_dem_path}")
elif os.path.exists(cfg['dem']):
    target_dem_path = cfg['dem']
    print(f"Loading DEM: {target_dem_path}")
else:
    raise FileNotFoundError(f"DEM file not found! Please check provided path")


DEM = IO.Raster(target_dem_path)


model = cfg['model']
if model["debug"]:
    # Debug line for comparing means of elevations
    current_mean = np.nanmean(DEM.array)
    print(f"--> VERIFICATION: Map loaded successfully.")
    print(f"--> VERIFICATION: Mean Elevation is {current_mean:.6f} meters")

case_folder = os.path.join(os.getcwd(), 'gaia_flood_case')
case_input = IO.InputModel(DEM, num_of_sections=1, case_folder=case_folder)

# 2. Add Water Sources (Discharge boundaries)
bc = model['boundary_conditions']
box_upstream = np.array(bc['box_upstream'])
box_downstream = np.array(bc['box_downstream'])
discharge_values = np.array(bc['discharge_values'])
downstream_h = np.array(bc['downstream_h'])

bound_list = [
    {'polyPoints': box_upstream, 'type': 'open', 'hU': discharge_values},
    {'polyPoints': box_downstream, 'type': 'open', 'h': downstream_h}
]
case_input.set_boundary_condition(boundary_list=bound_list)

# --- CONDITIONAL: Rainfall Module ---
if model['rainfall'].get('on', True):
    print("--> VERIFICATION: Rainfall module ENABLED.")
    rain_mask_path = model['rainfall']['mask']
    rain_source_path = model['rainfall']['source']

    if os.path.exists(rain_mask_path):
        print(f"Loading rain mask: {rain_mask_path}")
    else:
        raise FileNotFoundError(f"Rainfall initialization failed: rain_mask file not found at '{rain_mask_path}'")

    if os.path.exists(rain_source_path):
        print(f"Loading rain source: {rain_source_path}")
    else:
        raise FileNotFoundError(f"Rainfall initialization failed: rain_source CSV not found at '{rain_source_path}'")

    rain_mask = IO.Raster(rain_mask_path)
    rain_source = pd.read_csv(rain_source_path, header=None).to_numpy()
    case_input.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)
else:
    print("--> VERIFICATION: Rainfall module DISABLED. Simulating dry weather.")

# --- CONDITIONAL: Friction & Landcover Module ---
default_friction = model['friction']['default']

if model['landcover'].get('on', True):
    print("--> VERIFICATION: Landcover module ENABLED. Applying heterogeneous friction.")
    landcover_path = model['landcover']['mask']

    if os.path.exists(landcover_path):
        print(f"Loading landcover: {landcover_path}")
    else:
        raise FileNotFoundError(f"Landcover initialization failed: landcover file not found at '{landcover_path}'")

    landcover = IO.Raster(landcover_path)

    case_input.set_landcover(landcover)
    case_input.set_grid_parameter(manning={
        'land_value': model['friction']['landcover']['value'],
        'param_value': model['friction']['landcover']['friction'],
        'default_value': default_friction
    })
else:
    print(f"--> VERIFICATION: Landcover module DISABLED. Applying uniform friction ({default_friction}).")
    
    # Create a synthetic landcover map from the DEM to satisfy the physics engine
    dummy_landcover = copy.copy(DEM)
    dummy_landcover.array = np.where(np.isnan(DEM.array), np.nan, 0.0)

    case_input.set_landcover(dummy_landcover)
    case_input.set_grid_parameter(manning={
        'param_value': [default_friction], 
        'land_value': [0], 
        'default_value': default_friction
    })

# Settings and Execution
case_input.set_initial_condition('h0', model['initial_conditions']['h0'])
case_input.set_gauges_position(np.array(model['observation']['gauges_position']))
solver = model['solver']
case_input.set_runtime([solver['start_time'], solver['end_time'], solver['log_interval'], solver['save_interval']])

print("\nWriting files and starting GPU simulation...")
case_input.write_input_files()
flood.run(case_folder)
print("Simulation complete.")

# 5. Visualization (if enabled in config)
# Prefer model.visualize in config.yml, but also support legacy visualization.visualize.
visualize_enabled = model.get('visualize', cfg.get('visualization', {}).get('visualize', False))

if visualize_enabled:
    print("\nGenerating visualizations...")
    try:
        # Save figures in the project root and keep names unique per run.
        run_tag = Path(target_dem_path).stem
        if run_tag.startswith('DEM_noisy_'):
            run_tag = run_tag.replace('DEM_noisy_', 'iter_')
        output_dir = project_root
        
        # Suppress warnings during visualization generation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 5a. Visualize the DEM (3D representation using hillshade)
            dem_hillshade_filename = os.path.join(output_dir, f'dem_3d_{run_tag}.png')
            IO.grid_show.hillshade(
                DEM,
                figsize=(14, 10),
                azdeg=315,
                altdeg=45,
                vert_exag=2,
                cmap='gray',
                blend_mode='overlay',
                alpha=1.0,
                scale_ratio=1
            )
            # Note: hillshade doesn't have a figname parameter, so we save manually
            import matplotlib.pyplot as plt
            plt.savefig(dem_hillshade_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - DEM 3D hillshade saved: {dem_hillshade_filename}")
            
            # 5b. Visualize water depth map
            sim_output_dir = os.path.join(case_folder, 'output')
            h_max_path = os.path.join(sim_output_dir, 'h_max_3600.asc')
            h_final_path = os.path.join(sim_output_dir, 'h_3600.asc')
            output_h_path = h_max_path if os.path.exists(h_max_path) else h_final_path
            
            if os.path.exists(output_h_path):
                h_raster = IO.Raster(output_h_path)
                
                # Save water depth map
                mapshow_filename = os.path.join(output_dir, f'water_height_{run_tag}.png')
                IO.grid_show.mapshow(
                    h_raster,
                    figname=mapshow_filename,
                    figsize=(12, 10),
                    dpi=300,
                    title='Simulated Water Depth',
                    cax=True,
                    cax_str='Depth (m)'
                )
                print(f"  - Water depth map saved: {mapshow_filename}")
            else:
                print(f"  WARNING: Output water depth file not found at {output_h_path}")
        
        print(f"Visualizations saved to {output_dir}")
            
    except Exception as e:
        print(f"WARNING: Visualization generation failed: {str(e)}")
        import traceback
        traceback.print_exc()