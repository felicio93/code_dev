import glob
import os
import re
import numpy as np
import xarray as xr
import pandas as pd
from collections import defaultdict
from scipy.spatial import KDTree
import ocsmesh

# --- Function Definitions (should be at the top of your script) ---
def inverse_distance_weights(distances: np.ndarray, power: float = 1.0) -> np.ndarray:
    safe_distances = np.maximum(distances, 1e-6)
    weights = 1.0 / np.power(safe_distances, power)
    return weights / weights.sum(axis=1, keepdims=True)

class NearestSpatialLocator:
    def __init__(self, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
        self.tree = KDTree(np.column_stack((x_coords, y_coords)))
    def query(self, lon: np.ndarray, lat: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        points = np.column_stack((lon, lat))
        distances, indices = self.tree.query(points, k=k)
        return distances, indices

def get_file_number(filepath):
    match = re.search(r'out2d_(\d+)\.nc', os.path.basename(filepath))
    return int(match.group(1)) if match else -1
# --- End of Function Definitions ---


# 1. Initial Setup
input_csv_dir = r"./extracted_files"
model_output_dir = r"/work2/noaa/nos-surge/felicioc/BeringSea/R11b/outputs"
processed_output_dir = r"./processed_output"
os.makedirs(processed_output_dir, exist_ok=True)

hgrid = ocsmesh.Mesh.open("/work2/noaa/nos-surge/felicioc/BeringSea/R11b/hgrid.gr3", crs=4326)
x_mod, y_mod = hgrid.coord[:, 0], hgrid.coord[:, 1]
model_tree = NearestSpatialLocator(x_coords=x_mod, y_coords=y_mod)
model_depths = hgrid.value.flatten() 

list_of_dates_str = ['2019-08-01', '2019-08-11', '2019-08-21', '2019-08-31',
                     '2019-09-10', '2019-09-20', '2019-09-30',
                     '2019-10-10', '2019-10-19', '2019-10-29']

# 2. Pre-computation: Scan all model files to map dates to a LIST of file paths
print("--- Pre-scanning model files to map dates... ---")
date_to_files_map = defaultdict(list)
search_pattern = os.path.join(model_output_dir, "out2d_*.nc")
file_paths = sorted(glob.glob(search_pattern), key=get_file_number)

for file_path in file_paths:
    with xr.open_dataset(file_path, cache=False) as ds:
        file_dates = np.unique(pd.to_datetime(ds.time.values).date)
        for d in file_dates:
            if file_path not in date_to_files_map[d]:
                date_to_files_map[d].append(file_path)
print("--- Date mapping complete. ---")


# 3. Main Loop: Iterate through each target date
for date_str in list_of_dates_str:
    print(f"\n--- Processing date: {date_str} ---")
    try:
        date_obj = pd.to_datetime(date_str).date()
        csv_filename = f"pline_06_{date_obj.strftime('%Y%m%d')}.csv"
        csv_filepath = os.path.join(input_csv_dir, csv_filename)

        if not os.path.exists(csv_filepath):
            print(f"Warning: Input file not found, skipping: {csv_filepath}")
            continue
        
        df = pd.read_csv(csv_filepath)

        nodes_weight, nodes_query = model_tree.query(
            lon=df['Longitude (^oE)'], 
            lat=df['Latitude (^oN)']
        )
        idw = inverse_distance_weights(nodes_weight)

        model_file_paths = date_to_files_map.get(date_obj)
        if not model_file_paths:
            print(f"Warning: No model data found for date {date_str}. Skipping.")
            continue
            
        print(f"Found model data in: {[os.path.basename(p) for p in model_file_paths]}")
        
        ## NEW ## Calculate the interpolated depth
        flat_indices_depth = nodes_query.flatten()
        selected_depths_1d = model_depths[flat_indices_depth]
        selected_depths = np.reshape(selected_depths_1d, nodes_query.shape)
        final_depth_values = np.sum(selected_depths * idw, axis=1)

        with xr.open_mfdataset(model_file_paths) as ds:
            model_elevation = ds['elevation']
            daily_data = model_elevation.sel(time=date_str)
            daily_average = daily_data.mean(dim='time')

            original_shape = nodes_query.shape
            flat_indices_elev = nodes_query.flatten()
            
            selected_1d = daily_average.isel(nSCHISM_hgrid_node=flat_indices_elev)
            selected_data = np.reshape(selected_1d.values, original_shape)

            time_size = len(daily_data.time)
            values = np.sum(selected_data * idw, axis=1) * (time_size / 24.0)

        output_df = df.copy()
        output_df['SCHISM ADT (m)'] = values
        output_df['SCHISM_Depth'] = final_depth_values ## NEW ## Add the new column

        output_filename = f"pline_model_06_{date_obj.strftime('%Y%m%d')}.csv"
        output_filepath = os.path.join(processed_output_dir, output_filename)
        
        output_df.to_csv(output_filepath, index=False)
        print(f"-> Successfully saved output to {output_filepath}")

    except Exception as e:
        print(f"An error occurred while processing {date_str}: {e}")

print("\n--- All tasks complete ---")

# 4. Final Averaging Step
print("\n--- Averaging all generated CSV files... ---")
generated_files = sorted(glob.glob(os.path.join(processed_output_dir, "pline_model_06_*.csv")))

if len(generated_files) > 0:
    final_df_template = pd.read_csv(generated_files[0])
    
    ## NEW ## Add 'SCHISM_Depth' to the list of columns to average
    columns_to_average = ['Observation ADT (m)', 'Model ADT (m)', 'SCHISM ADT (m)', 'SCHISM_Depth']
    
    data_to_average = []
    for f in generated_files:
        df = pd.read_csv(f)
        data_to_average.append(df[columns_to_average].values)
        
    stacked_data = np.array(data_to_average)
    mean_values = np.nanmean(stacked_data, axis=0)
    
    final_averaged_df = final_df_template.copy()
    final_averaged_df[columns_to_average] = mean_values
    
    final_output_path = os.path.join(processed_output_dir, "pline_model_06_AVERAGE_ALL.csv")
    final_averaged_df.to_csv(final_output_path, index=False)
    
    print(f"-> Successfully saved final averaged output to {final_output_path}")
else:
    print("No processed files found to average.")