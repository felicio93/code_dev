#!/usr/bin/env python3
"""
This script serves as a template for parallel mesh plotting.

Before running remember to load:
module load openmpi

You can run it on a single node:
srun -N 1 -n 80 python hs_coupled.py > mpi_output.log 2>&1

or multi-node:
srun -N 2 -n 130 python hs_coupled.py > mpi_output.log 2>&1

I recommend you use as many nodes needed for: -n > number of .nc output files


Other variables (other than Hs) or 
unstructured models (ADCIRC, FVCOM, etc.) 
can be easily added by changing the plot func.

"""

import os
import sys
import time
import re

from mpi4py import MPI
import xarray as xr
import numpy as np

import numpy.typing as npt
import typing as T

import matplotlib.pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.tri as tri
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2


def natural_sort_key(s):
    """Key function for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def split_quads(face_nodes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """
    https://gist.github.com/pmav99/5ded91f18ef096b080b2ed45598c7d1c
    copied from: https://github.com/ec-jrc/Thalassa/blob/master/thalassa/utils.py
    """
    if face_nodes.shape[-1] != 4:
        return face_nodes
    existing_triangles = face_nodes[:, :3]
    quad_indexes = np.nonzero(~np.isnan(face_nodes).any(axis=1))
    quads = face_nodes[quad_indexes]
    quads_first_column = quads[:, 0]
    quads_last_two_columns = quads[:, -2:]
    new_triangles = np.c_[quads_first_column, quads_last_two_columns]
    new_face_nodes = T.cast(
        npt.NDArray[np.int_],
        np.r_[existing_triangles, new_triangles].astype(int),
    )
    return new_face_nodes

def fixed_connectivity_tri(data_dir):
    ds_tri = xr.open_dataset(
        f"{data_dir}out2d_1.nc",
        chunks={},
        engine='h5netcdf',
        drop_variables=['vvel4.5',
                        'uvel4.5',
                        'vvel_bottom',
                        'uvel_bottom',
                        'vvel_surface',
                        'uvel_surface',
                        'salt_bottom',
                        'temp_bottom',
                        'precipitationRate',
                        'evaporationRate',
                        'windSpeedX',
                        'windSpeedY',
                        'windStressX',
                        'windStressY',
                        'dryFlagElement',
                        'dryFlagSide',
                        'dryFlagNode',
                        ]
    )
    connect=np.array(ds_tri['SCHISM_hgrid_face_nodes'][:])-1
    connect_tri=split_quads(np.array(connect))
    x, y = np.array(ds_tri['SCHISM_hgrid_node_x']), np.array(ds_tri['SCHISM_hgrid_node_y'])
    depth=np.array(ds_tri.variables['depth'])

    return x,y,connect_tri,depth


def create_video_from_images(image_folder, output_file, fps=18):
    """Creates an MP4 video from a folder of images.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_file (str): Name of the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the video.
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
    images.sort(key=natural_sort_key)

    if not images:
        print("No images found in the folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    video.release()
    print("Video created successfully!")


def plot(file0,
         file,
         f,
         vmin,
         vmax,
         interv,
         x,
         y,
         triangulation,
         wdir,
         rank,
         depth=None,
         isobaths=None,
         ):
    """
    Change this plot function with your own plot func.
    This template function is used to plot Hs for SCHISM+WWMIII.
    2 Subplots are created, one for the entire domain and another 
    for a zoomed in area around Atka Island
    """

    # Making sure the files are opened correctly
    try:
        ds0 = xr.open_dataset(file0,
                            engine='h5netcdf',
                            drop_variables=['minimum_depth',
                                            'SCHISM_hgrid',
                                            'crs',
                                            'depth',
                                            'SCHISM_hgrid_face_x',
                                            'SCHISM_hgrid_face_y',
                                            'SCHISM_hgrid_edge_x',
                                            'SCHISM_hgrid_edge_y',
                                            'SCHISM_hgrid_face_nodes',
                                            'SCHISM_hgrid_edge_nodes',
                                            'meanWavePeriod',
                                            'meanWaveDirection',
                                            'bottom_index_node',
                                            'evaporationRate',
                                            'precipitationRate',
                                            'windSpeedX',
                                            'windSpeedY',
                                            'dryFlagElement',
                                            'dryFlagSide',
                                            'elevation',
                                            'zeta',
                                            'peakPeriod',
                                            'dominantDirection',
                                            'dryFlagElement',
                                            'dryFlagSide',
                                            ])
        hs0 = np.array(ds0.sigWaveHeight[:,:])
        ts0 = np.array(ds0.time)

        ds = xr.open_dataset(file,
                            engine='h5netcdf',
                            drop_variables=['minimum_depth',
                                            'SCHISM_hgrid',
                                            'crs',
                                            'depth',
                                            'bottom_index_node',
                                            'evaporationRate',
                                            'precipitationRate',
                                            'windSpeedX',
                                            'windSpeedY',
                                            'dryFlagElement',
                                            'dryFlagSide',
                                            'elevation',
                                            'zeta',
                                            'peakPeriod',
                                            'dominantDirection',
                                            'dryFlagElement',
                                            'dryFlagSide',
                                            ])
        hs = np.array(ds.sigWaveHeight[:,:])
        ts = np.array(ds.time)
    except Exception as e:
        print(f"Rank {rank} failed to read {file}: {e}")
        sys.stdout.flush()
        raise e 

    # Iterating over the timesteps:
    for hr_idx,t in enumerate(ts):

        i_time = str(ts[hr_idx]).split(".")[0]

        # Creating my own colormap:
        cmap = plb.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create fields subplots:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(211)
        
        tp = ax.tripcolor(triangulation,hs0[hr_idx],shading='flat',cmap=cmap,norm=norm)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.2)
        cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))        
        cb.set_label("Hs (m): Uncoupled Model")

        ax.set_xlim(185.7,191.5)
        ax.set_ylim(51.75,53.25)
        ax.set_facecolor('gray')
        ax.set_title(f'Significant Wave Height (Hs) at {i_time}')  


        ax2 = fig.add_subplot(212,)
        tp2 = ax2.tripcolor(triangulation,((hs[hr_idx]-hs0[hr_idx])/hs0[hr_idx])*100,shading='flat',cmap='RdBu_r',vmin=-50,vmax=50)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="5%", pad=0.2)
        cb2 = fig.colorbar(tp2,
                      cax=cax2,
                      )
        cb2.set_label("% Change in Hs: (Coupled-Uncoupled)/Uncoupled")        
        ax2.set_xlim(185.7,191.5)
        ax2.set_ylim(51.75,53.25)
        ax2.set_facecolor('gray')

        if depth is not None and isobaths is not None:
            depth[np.isnan(depth)] = -9999
            #ax.tricontour(triangulation, depth, levels=isomask, linewidths=.5, colors="white")
            ax.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
            ax2.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
    
        # Saving plots:
        fig.tight_layout()
        #print(f"saving file: {wdir}{f:03}_{hr_idx:03}.jpeg, from rank: {rank}")
        #sys.stdout.flush()
        fig.savefig(f"{wdir}{f:03}_{hr_idx:03}.jpeg",dpi=100)
     
        plt.close(fig)


def main():
    start_time = time.time()

    ### Enter inputs here:
    rdir0 = r"/work2/noaa/nos-surge/felicioc/BeringSea/R09a/outputs/"
    rdir = r"/work2/noaa/nos-surge/felicioc/BeringSea/R09b/outputs/"
    interv=0.2
    vmin, vmax = 0,8
    wdir = r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/hs_fig_coup_a_b_zoom/"
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

    # Break quads and gather the mesh info (constant for all outputs):
    x,y,connect_tri,depth = fixed_connectivity_tri(f'{rdir}')
    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

    # Distribute file processing across ranks
    # surface_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('out2d_')], key=natural_sort_key)
    # file_list = [(surface_files[i], i) for i in range(len(surface_files))]

    # if you want to plot 3D variables, you will use something like:
    surface_files0 = sorted([f'{rdir0}/{i}' for i in os.listdir(rdir0) if i.startswith('out2d_')], key=natural_sort_key)
    surface_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('out2d_')], key=natural_sort_key)
    # X_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('horizontalVelX_')])
    # Y_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('horizontalVelY_')])
    # file_list = [(surface_files[i], X_files[i], Y_files[i], i) for i in range(0, len(surface_files))]
    file_list = [(surface_files0[i], surface_files[i], i) for i in range(0, len(surface_files))]

    # Distribute file processing across ranks
    files_per_rank = len(file_list) // size
    remainder = len(file_list) % size
    start_idx = rank * files_per_rank + min(rank, remainder)
    end_idx = start_idx + files_per_rank + (1 if rank < remainder else 0)
    files_to_process = file_list[start_idx:end_idx]

    # Each rank processes its assigned files
    for files in files_to_process:
        print(f"Rank {rank} started for files: {files}")
        sys.stdout.flush()

        plot(files[0], files[1],files[2], vmin, vmax, interv, x, y, triangulation, wdir, rank, depth, isobaths = [200,2000])
    
    comm.Barrier()

    if rank == 0:
        create_video_from_images(wdir, "movie_hsR09coup.mp4")

        # End the timer and calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Print the total elapsed time for the script
        print(f"Total time taken for the script to run: {elapsed_time:.2f} seconds.")
        sys.stdout.flush()

    MPI.Finalize()

if __name__ == '__main__':
    main()

