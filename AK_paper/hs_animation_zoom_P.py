
"""
module load openmpi

run as: srun -N 1 -n 80 python hs_animation_zoom_P.py > mpi_output.log 2>&1
"""

from mpi4py import MPI
import os
import sys

#from __future__ import annotations
import xarray as xr
import numpy as np

#import ocsmesh

import pathlib

import matplotlib.pyplot
import netCDF4
import pandas as pd

import numpy.typing as npt
import typing as T

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.tri as tri
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2


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


def create_video_from_images(image_folder, output_file, fps=6):
    """Creates an MP4 video from a folder of images.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_file (str): Name of the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the video.
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
    images.sort()

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


def plot(file,
         f,
         vmin,
         vmax,
         interv,
         x,
         y,
         triangulation,
         wdir,
         rank
         ):
    """
    Change this plot function with your own plot func.
    This template function is used to plot Hs for SCHISM+WWMIII.
    2 Subplots are created, one for the entire domain and another 
    for a zoomed in area around Atka Island
    """


    try:
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


    for hr_idx,t in enumerate(ts):

        i_time = str(ts[hr_idx]).split(".")[0]

        cmap = plb.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create fields plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(211)
        
        tp = ax.tripcolor(triangulation,hs[hr_idx],shading='flat',cmap=cmap,norm=norm)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.2)
        cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))        
        cb.set_label("Hs (m)")

        ax.set_xlim(x.min(),x.max())
        ax.set_ylim(y.min(),y.max())
        ax.set_facecolor('lightgray')
        ax.set_title(i_time)  


        ax2 = fig.add_subplot(212,)
        tp2 = ax2.tripcolor(triangulation,hs[hr_idx],shading='flat',cmap=cmap,norm=norm)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="5%", pad=0.2)
        cb2 = fig.colorbar(tp2,
                      cax=cax2,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
        cb2.set_label("Hs (m)")           
        ax2.set_xlim(180+4.6,180+7.1)
        ax2.set_ylim(51.95,52.45)
        ax2.set_facecolor('lightgray')


        # Show the plot
        fig.tight_layout()
        print(f"saving file: {wdir}{f:03}_{hr_idx:03}.jpeg, from rank: {rank}")
        sys.stdout.flush()
        fig.savefig(f"{wdir}{f:03}_{hr_idx:03}.jpeg",dpi=300)
     
        plt.close(fig)


def main():

    ### Enter inputs here:
    rdir = r"/work2/noaa/nosofs/felicioc/BeringSea/R08/outputs/"
    interv=0.5
    vmin, vmax = 0,10
    wdir = r"/work2/noaa/nosofs/felicioc/BeringSea/P08/hs_fig_p/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This is just a check:
    # if rank == 0:
    #     print(f"Rank {rank} is starting the task...")clear
    # else:
    #     print(f"Rank {rank} is doing its part.")
    # sys.stdout.flush()

    # Break quads and gather the mesh info (constant for all outputs):
    x,y,connect_tri,depth = fixed_connectivity_tri(f'{rdir}')
    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

    # Distribute file processing across ranks
    surface_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('out2d_')])
    #file_list = [(surface_files[i], i) for i in range(0, len(surface_files))]
    file_list = [(surface_files[i], i) for i in range(0, 90)]


    # if you want to plot 3D variables, you will use something like:
    # surface_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('out2d_')])
    # X_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('horizontalVelX_')])
    # Y_files = sorted([f'{rdir}/{i}' for i in os.listdir(rdir) if i.startswith('horizontalVelY_')])
    # file_list = [(surface_files[i], X_files[i], Y_files[i], i) for i in range(0, len(surface_files))]

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

        plot(files[0], files[1], vmin, vmax, interv, x, y, triangulation, wdir, rank)

    MPI.Finalize()

if __name__ == '__main__':
    main()

