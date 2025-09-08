import netCDF4 as nc
import numpy as np
import glob
import re
import sys
import os
from mpi4py import MPI

def calculate_average_surface_value(filename, rank, variable_name):
    """
    Opens a SCHISM NetCDF file, extracts a surface variable (temp or salinity),
    and calculates the time-averaged value for each node for that single file.

    Args:
        filename (str): The path to the NetCDF file.
        rank (int): The MPI rank of the process, used for logging.
        variable_name (str): The name of the variable to process ('temperature' or 'salinity').

    Returns:
        numpy.ndarray: A 1D array containing the average surface
                       value for each horizontal grid node,
                       or None if an error occurs.
    """
    dataset = None  # Initialize dataset to None
    try:
        # Open the NetCDF file in read mode
        dataset = nc.Dataset(filename, 'r')
        print(f"[Rank {rank}] Successfully opened: {os.path.basename(filename)}")

        # --- 1. Access the specified variable ---
        if variable_name not in dataset.variables:
            print(f"[Rank {rank}] Error: '{variable_name}' variable not found in {filename}.", file=sys.stderr)
            return None
        
        data_var = dataset.variables[variable_name]
        
        # --- 2. Extract and average the surface data ---
        surface_data_all_timesteps = data_var[:, :, -1]
        average_surface_value_per_node = np.mean(surface_data_all_timesteps, axis=0)
        
        return average_surface_value_per_node

    except FileNotFoundError:
        print(f"[Rank {rank}] Error: The file '{filename}' was not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[Rank {rank}] An unexpected error occurred while processing {filename}: {e}", file=sys.stderr)
        return None
    finally:
        # --- 3. Close the NetCDF file ---
        if dataset:
            dataset.close()

def natural_sort_key(filename: str):
    """
    Generate a key for natural sorting of filenames (e.g., file10 comes after file2).
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', filename)]


if __name__ == '__main__':
    # --- 1. Initialize MPI ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    file_paths = r"/work2/noaa/nos-surge/felicioc/BeringSea/R11b/outputs"
    variable = None#'salinity'

    # --- 2. Rank 0 validates input and distributes files ---
    if rank == 0:
        # Check for command-line argument
        if len(sys.argv) != 2 or sys.argv[1] not in ['temperature', 'salinity']:
            print("Usage: mpiexec -n <cores> python average_surface_temp.py <variable>", file=sys.stderr)
            print("       <variable> must be 'temperature' or 'salinity'", file=sys.stderr)
            # Signal other processes to abort by setting variable to None
            variable = None
        else:
            variable = sys.argv[1]
            print(f"[Rank 0] Variable to process set to: '{variable}'")
            model_output_dir = r"/work2/noaa/nos-surge/felicioc/BeringSea/R11b/outputs"
            print(f"[Rank 0] Searching for files in: {model_output_dir}")
            search_pattern = os.path.join(model_output_dir, f"{variable}_*.nc")
            file_paths = sorted(glob.glob(search_pattern), key=natural_sort_key)
            file_paths = file_paths[31:]
            if not file_paths:
                print("[Rank 0] Error: No files found matching the pattern. Exiting.", file=sys.stderr)
    
    # Broadcast variable name. If it's None, all processes will exit.
    variable = comm.bcast(variable, root=0)
    if not variable:
        comm.Abort()
        sys.exit(1)

    # Broadcast the list of file paths from rank 0 to all other processes
    file_paths = comm.bcast(file_paths, root=0)
    if not file_paths:
        # This condition is hit if rank 0 found no files
        if rank == 0:
            print("[Rank 0] Aborting due to no files found.")
        comm.Abort()
        sys.exit(1)
    
    # --- 3. Each process calculates its share of work ---
    # Divide the files among the processes
    files_for_this_rank = np.array_split(file_paths, size)[rank]
    
    local_averages = []
    print(f"[Rank {rank}] Assigned {len(files_for_this_rank)} files.")

    # --- 4. Each process runs the calculation on its assigned files ---
    for file_path in files_for_this_rank:
        avg_value_array = calculate_average_surface_value(file_path, rank, variable)
        if avg_value_array is not None:
            local_averages.append(avg_value_array)
            
    # --- 5. Gather all results to Rank 0 ---
    # Each process sends its list of local_averages to rank 0.
    # The result on rank 0 is a list of lists.
    all_results = comm.gather(local_averages, root=0)
    
    # --- 6. Rank 0 calculates the final average and saves the result ---
    if rank == 0:
        print("\n[Rank 0] --- Aggregating results from all processes ---")
        
        # Flatten the list of lists into a single list of all file averages
        all_files_averages = [item for sublist in all_results for item in sublist]

        if all_files_averages:
            # Calculate the final average across all files from all processes
            final_average_value = np.mean(all_files_averages, axis=0)
            
            print("\n--- Overall Calculation Complete ---")
            print(f"Successfully processed {len(all_files_averages)} files across {size} processes.")
            print(f"The final array contains the grand-average surface {variable} for {len(final_average_value)} nodes.")
            
            # Save the final array to a CSV file
            output_filename = f'final_average_surface_{variable}.csv'
            np.savetxt(output_filename, final_average_value, delimiter=',', fmt='%.8f')
            print(f"\nFinal average {variable} array saved to: {os.path.abspath(output_filename)}")
            
        else:
            print("\n--- Calculation Failed ---")
            print("Could not process any of the input files successfully.")

