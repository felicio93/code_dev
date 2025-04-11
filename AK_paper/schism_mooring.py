#!/usr/bin/env python3
"""
Before running remember to load:
module load openmpi

You can run it on a single node:
srun -N 1 -n 80 python schism_mooring.py > mpi_output_moor.log 2>&1

or multi-node:
srun -N 2 -n 160 python schism_mooring.py > mpi_output_moor.log 2>&1

I recommend you use as many nodes needed for: -n > number of .nc output files

"""

import os
import sys
import time
import re

from mpi4py import MPI
import xarray as xr
import numpy as np



def natural_sort_key(s):
    """Key function for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def merge_ds(z_file,T_file,S_file,i,schism_idx,chunk_size,rank):

    try:
        z = xr.open_dataset(z_file,chunks={'time': chunk_size})
        T = xr.open_dataset(T_file,chunks={'time': chunk_size})
        S = xr.open_dataset(S_file,chunks={'time': chunk_size})
    except Exception as e:
        print(f"Rank {rank} failed to read one of the files: {e}")
        sys.stdout.flush()
        raise e
    
    dsz = z.zCoordinates[:,schism_idx,:]
    dsT = T.temperature[:,schism_idx,:]
    dsS = S.salinity[:,schism_idx,:]

    merged_ds = xr.merge([dsz,dsT,dsS])

    return merged_ds
    
def main():
    start_time = time.time()

    ### Enter inputs here:
    rdir = r"/work2/noaa/nosofs/felicioc/BeringSea/R07/outputs/"#r"/work2/noaa/nos-surge/felicioc/BeringSea/R09a/outputs/"
    wdir = r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/mooring_R07/"#r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/mooring_R09a/"
    schism_idx = [989944,1054772,793218] #idx of nodes to be extracted
    chunk_size = 100
    ###

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()

    ##This is just a check:
    #if rank == 0:
    #    print(f"Rank {rank} is starting the task... size={size}")
    #else:
    #    print(f"Rank {rank} is doing its part. size={size}")
    #sys.stdout.flush()

    print(f"This is just a check: from rank {rank} out of {size} ranks")
    sys.stdout.flush()

    ###if you want to plot 3D variables, you will use something like:
    z_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('zCoordinates_')], key=natural_sort_key)
    T_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('temperature_')], key=natural_sort_key)
    S_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('salinity_')], key=natural_sort_key)
    file_list = [(z_files[i], T_files[i], S_files[i], i) for i in range(0, len(z_files))]

    # Distribute file processing across ranks
    files_per_rank = len(file_list) // size
    remainder = len(file_list) % size
    start_idx = rank * files_per_rank + min(rank, remainder)
    end_idx = start_idx + files_per_rank + (1 if rank < remainder else 0)
    files_to_process = file_list[start_idx:end_idx]

    # Each rank processes its assigned files
    merged_ds = None
    for files in files_to_process:
        print(f"Rank {rank} started for files: {files}")

        merged_ds = merge_ds(files[0],files[1],files[2],files[3],schism_idx,chunk_size,rank)
        sys.stdout.flush()
 
    comm.Barrier()
    all_merged_datasets = comm.gather(merged_ds, root=0)


    if rank == 0:
        # Remove None values if any rank didn't produce a valid dataset
        all_merged_datasets = [ds for ds in all_merged_datasets if ds is not None]
        
        # Concatenate all datasets (e.g., along the 'time' dimension, adjust as needed)
        final_merged_ds = xr.concat(all_merged_datasets, dim="time")
        final_merged_ds = final_merged_ds.sortby('time')

        final_merged_ds.to_netcdf(wdir+f"merged_final.nc")

        # End the timer and calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Print the total elapsed time for the script
        print(f"Total time taken for the script to run: {elapsed_time:.2f} seconds.")
        sys.stdout.flush()

    MPI.Finalize()

if __name__ == '__main__':
    main()

