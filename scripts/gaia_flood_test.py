import os
import numpy as np
import pandas as pd
from synxflow import IO, flood
from synxflow.IO.demo_functions import get_sample_data

# 1. Setup Data Paths
dem_file, demo_data, data_path = get_sample_data()
DEM = IO.Raster(os.path.join(data_path, 'DEM.gz'))
case_folder = os.path.join(os.getcwd(), 'gaia_flood_case')
case_input = IO.InputModel(DEM, num_of_sections=1, case_folder=case_folder)

# 2. Add Water Sources (Discharge and Rain)
box_upstream = np.array([[1427, 195], [1446, 243]])
box_downstream = np.array([[58, 1645], [72, 1170]])
discharge_values = np.array([[0, 100], [3600, 100]]) 
bound_list = [
    {'polyPoints': box_upstream, 'type': 'open', 'hU': discharge_values},
    {'polyPoints': box_downstream, 'type': 'open', 'h': np.array([[0, 5], [3600, 5]])}
]
case_input.set_boundary_condition(boundary_list=bound_list)

rain_mask = IO.Raster(os.path.join(data_path, 'rain_mask.gz'))
rain_source = pd.read_csv(os.path.join(data_path, 'rain_source.csv'), header=None).to_numpy()
case_input.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)

# 3. Add Friction
landcover = IO.Raster(os.path.join(data_path, 'landcover.gz'))
case_input.set_landcover(landcover)
case_input.set_grid_parameter(manning={
    'param_value': [0.035, 0.055], 
    'land_value': [0, 1], 
    'default_value': 0.035
})

# 4. Settings and Execution
case_input.set_initial_condition('h0', 0.0)
case_input.set_gauges_position(np.array([[560, 1030], [1140, 330]]))
case_input.set_runtime([0, 3600, 600, 1200]) # 1-hour simulation

print("Writing files and starting GPU simulation...")
case_input.write_input_files()
flood.run(case_folder)
print("Simulation complete.")