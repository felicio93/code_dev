#!/usr/bin/env python3
"""
This script serves as a template for parallel mesh plotting.

Before running remember to load:
module load openmpi

You can run it on a single node:
srun -N 1 -n 80 python avg_3D.py > mpi_output.log 2>&1

or multi-node:
srun -N 2 -n 130 python avg_3D.py > mpi_output.log 2>&1

For memory-intensive data, it's wise to rune the srun:
srun -N 4 -n 20 python avg_3D.py > mpi_output.log 2>&1

"""
import os
import sys
import time
import re
import glob
from mpi4py import MPI
import xarray as xr
import numpy as np

def natural_sort_key(s):
    """Key function for natural sorting of filenames."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

# FIX #1: Added 'chunks' to handle large input files without OOM errors.
def process_single_file(file_path, var_name, day_index, output_dir):
    """
    Worker function that opens, averages, and saves a single file using chunking
    for memory efficiency.
    """
    try:
        # Open the dataset in "chunks" to avoid loading it all into memory.
        # This processes the time dimension in blocks. 'auto' is also a good option.
        with xr.open_dataset(file_path, chunks={'time': 6}) as ds:
            # Get the first time step from the original dataset
            first_time = ds['time'].isel(time=0)

            # This .mean() operation will now happen chunk by chunk
            time_avg = ds[var_name].mean(dim='time', keepdims=True)

            # Assign the correct coordinate using the .values attribute
            time_avg = time_avg.assign_coords({'time': [first_time.values]})

            # Create a descriptive output filename
            output_filename = os.path.join(output_dir, f"{var_name}_avg_{day_index:03d}.nc")
            
            # .to_netcdf() will trigger the computation and save the result
            time_avg.to_netcdf(output_filename)

    except Exception as e:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"Rank {rank} ERROR processing file {file_path}: {e}", file=sys.stderr)
        sys.stderr.flush()

# FIX #2: Replaced open_mfdataset with a loop for memory efficiency.
def create_final_average_parallel(comm, input_dir, file_pattern, var_name, output_filename):
    """
    Calculates a grand average in parallel using a memory-efficient loop
    instead of open_mfdataset.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    daily_files = None
    if rank == 0:
        search_path = os.path.join(input_dir, file_pattern)
        daily_files = sorted(glob.glob(search_path))
        print(f"Rank 0 found {len(daily_files)} files for final averaging of '{var_name}'.")
    daily_files = comm.bcast(daily_files, root=0)

    if not daily_files:
        if rank == 0:
            print(f"Warning: No files found for pattern {file_pattern}. Skipping.")
        return

    # Distribute files to each rank
    files_per_rank = len(daily_files) // size
    remainder = len(daily_files) % size
    start_idx = rank * files_per_rank + min(rank, remainder)
    end_idx = start_idx + files_per_rank + (1 if rank < remainder else 0)
    my_files = daily_files[start_idx:end_idx]

    # Instead of open_mfdataset, loop through files one by one.
    local_sum_da = None
    local_count = 0
    
    if my_files:
        for i, file_path in enumerate(my_files):
            with xr.open_dataset(file_path) as ds:
                if i == 0:
                    # On the first file, initialize the sum
                    local_sum_da = ds[var_name].copy(deep=True)
                else:
                    # On subsequent files, add to the existing sum
                    # local_sum_da += ds[var_name]
                    local_sum_da.data += ds[var_name].values
                # Each daily file represents one time step in the grand average
                local_count += 1
    
    total_count = comm.allreduce(local_count, op=MPI.SUM)
    if total_count == 0:
        if rank == 0: print("Total count is 0, cannot compute average.")
        return

    if local_sum_da is not None:
        local_sum_np = local_sum_da.to_numpy()
    else:
        # Get template shape from rank 0 to handle ranks with no files
        if rank == 0:
             with xr.open_dataset(daily_files[0]) as ds_template:
                  template_shape = ds_template[var_name].shape
        else:
            template_shape = None
        template_shape = comm.bcast(template_shape, root=0)
        local_sum_np = np.zeros(template_shape, dtype=np.float32)

    global_sum_np = np.empty_like(local_sum_np) if rank == 0 else None
    comm.Reduce(local_sum_np, global_sum_np, op=MPI.SUM, root=0)

    if rank == 0:
        print("Rank 0 received global sum. Calculating final average...")
        final_average_np = global_sum_np / total_count
        
        with xr.open_dataset(daily_files[0]) as ds_template:
            final_ds = ds_template.copy(deep=True)
            final_ds[var_name].data = final_average_np
            
            final_output_path = os.path.join(input_dir, output_filename)
            final_ds.to_netcdf(final_output_path)
            print(f"Successfully saved final parallel average to {final_output_path}")

def main():
    start_time = time.time()
    
    start_day = 61
    end_day = 92
    
    rdir = r"/work2/noaa/nos-surge/felicioc/BeringSea/R11a/outputs/"
    output_dir = r"/work2/noaa/nos-surge/felicioc/BeringSea/P11/vertical_avg/hourly_to_daily/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    task_list = None
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Rank 0: Searching for files in {rdir} for days {start_day}-{end_day}...")
        
        task_list = []
        pattern = re.compile(r'_(\d+)\.nc$')
        
        try:
            for filename in os.listdir(rdir):
                match = pattern.search(filename)
                if match:
                    file_num = int(match.group(1))
                    if start_day <= file_num <= end_day:
                        file_path = os.path.join(rdir, filename)
                        if filename.startswith('horizontalVelX_'):
                            task_list.append((file_path, 'horizontalVelX', file_num))
                        elif filename.startswith('temperature_'):
                            task_list.append((file_path, 'temperature', file_num))
            
            task_list.sort(key=lambda x: (x[2], x[0]))
            print(f"Rank 0: Found {len(task_list)} total tasks to distribute.")

        except FileNotFoundError:
            print(f"Rank 0 ERROR: Input directory not found: {rdir}", file=sys.stderr)
            task_list = []
            
    task_list = comm.bcast(task_list, root=0)
    
    if not task_list:
        if rank == 0:
            print("No files to process. Exiting cleanly.")
        return
    
    comm.Barrier()

    tasks_per_rank = len(task_list) // size
    remainder = len(task_list) % size
    start_idx = rank * tasks_per_rank + min(rank, remainder)
    end_idx = start_idx + tasks_per_rank + (1 if rank < remainder else 0)
    tasks_to_process = task_list[start_idx:end_idx]

    for file_path, var_name, day_idx in tasks_to_process:
        process_single_file(file_path, var_name, day_idx, output_dir)
    
    print(f"Rank {rank} finished daily processing and is waiting at barrier.")
    sys.stdout.flush()
    comm.Barrier()

    if rank == 0: print("\n--- Starting Final Grand Average Calculation ---")
    
    create_final_average_parallel(comm, output_dir, "horizontalVelX_avg_*.nc", "horizontalVelX", "FINAL_grand_average_hvelx.nc")
    comm.Barrier()
    create_final_average_parallel(comm, output_dir, "temperature_avg_*.nc", "temperature", "FINAL_grand_average_temp.nc")
    
    comm.Barrier()
    if rank == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal time taken for all steps: {elapsed_time:.2f} seconds.")
        sys.stdout.flush()

# FIX #3: Added the standard entry point to execute the main() function.
if __name__ == "__main__":
    main()
