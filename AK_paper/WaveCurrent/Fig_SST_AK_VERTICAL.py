#!/usr/bin/python3

import os
import xarray as xr
import numpy as np
import numpy.typing as npt
import typing as T
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pylab as plb
import matplotlib.gridspec as gridspec

def dates_range(start_date, end_date):
    """
    This function takes the start and end dates and returns
    all the dates in between.
    """
    dates = []
    for i in range(
        int(
            (
                datetime.strptime(end_date, "%Y%m%d")
                - datetime.strptime(start_date, "%Y%m%d")
            ).days
        )
        + 1
    ):
        date = datetime.strptime(start_date, "%Y%m%d") + timedelta(days=i)
        #dates.append(date.strftime("%m/%d/%y"))
        dates.append(date)

    return dates



def open_schism(date, n, data_dir):
    """
    """
    n=n+1
    try:
        ds = xr.open_dataset(
                f"{data_dir}temperature_{n}.nc",
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
        print("Model data found for: ", date)
    except:
        print("No model data found for: ", date)
        pass

    return ds

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


def plot_arctic(
               triangulation,
               ds1,
               ds2,
               v1,
               v2,
               depth=None,
               isobaths=None,
               isomask=None,
               output_dir="",
               n=0,
               long_name="name of the var",
               time_str="time str",
               vmin=None,
               vmax=None,
               latmin=None,
               latmax=None,
               lonmin=None,
               lonmax=None,
               interv=None):
    
    cmap = plb.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 2])
    ax = fig.add_subplot(gs[0]) # Top Left
    ax2 = fig.add_subplot(gs[1:]) # Top Right
    ax3 = fig.add_subplot(gs[2]) # Bottom row, spanning both columns


    tp = ax.tripcolor(triangulation,ds1,shading='flat',cmap=cmap,norm=norm)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
    cb.set_label('Temperature (degC)')

    ax.set_title(f"SCHISM Standalone at time: {time_str} 12:00:00")
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    ax.plot(179.758, 57.495, 'bx', markersize=16)


    tp3 = ax3.tripcolor(triangulation,ds2,shading='flat',cmap=cmap,norm=norm)
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.2)
    cb3 = fig.colorbar(tp3,
                    cax=cax3,
                    boundaries=np.arange(vmin,vmax,interv),
                    cmap=cmap,norm=norm,
                    ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
    cb3.set_label('Temperature (degC)')

    ax3.set_title(f"SCHISM+WWM Decoupled at time: {time_str} 12:00:00")
    ax3.set_xlim(lonmin,lonmax)
    ax3.set_ylim(latmin,latmax)
    ax3.set_xlim(lonmin,lonmax)
    ax3.set_ylim(latmin,latmax)
    ax3.plot(179.758, 57.495, 'rx', markersize=16)


    ax2.scatter(v1, [i for i in range(1,52)], label='SCHISM Standalone', color='b', s=10)
    ax2.scatter(v2,[i for i in range(1,52)], label='SCHISM+WWM Decoupled', color='r', s=10)

    ax2.plot(v1, [i for i in range(1,52)], color='b')  # Line for observed
    ax2.plot(v2,[i for i in range(1,52)], color='r')  # Line for modeled

    ax2.legend(loc=4)
    ax2.set_ylabel("Vertical Layer #")
    ax2.set_xlabel('Temperature (degC)')
    ax3.set_xlim(0,11)


    if depth is not None and isobaths is not None:
        depth[np.isnan(depth)] = -9999
        #ax.tricontour(triangulation, depth, levels=isomask, linewidths=.5, colors="white")
        ax.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
        ax3.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")

    fig.tight_layout()
    fig.savefig(output_dir+'{:04d}.jpeg'.format(n),dpi=300)
    plt.clf()

def main(var,start_date,end_date,output_dir,isobaths,data_dir1,data_dir2,isomask,node_sel):

    # ds = concat_arctic(start_date,end_date,data_dir)
    if var == 'temperature':
        long_name="Sea Surface Temperature (degC)"

    dates = dates_range(start_date, end_date)    
    times = [dd.strftime("%m/%d/%y") for dd in dates]
    x,y,connect_tri,depth = fixed_connectivity_tri(data_dir1)
    lonmin,lonmax=x.min(),x.max()
    latmin,latmax=y.min(),y.max()

    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

    for n, date in enumerate(dates):
        ds1 = open_schism(date, n, data_dir1)
        v1 = np.array(ds1.variables[var][0,node_sel+1,:])
        ds1 = np.array(ds1.variables[var][0,:,-1])
        ds2 = open_schism(date, n, data_dir2)
        v2 = np.array(ds2.variables[var][0,node_sel+1,:])
        ds2 = np.array(ds2.variables[var][0,:,-1])

        print("creating plot ",n," of ", len(dates))

        time_str=times[n]


        if var == 'temperature':
            plot_arctic(triangulation,
                        ds1,
                        ds2,
                        v1,
                        v2,
                        depth=depth,
                        isobaths=isobaths,
                        isomask=isomask,
                        output_dir=output_dir,
                        n=n,
                        long_name=long_name,
                        time_str=time_str,
                        latmin=latmin,
                        latmax=latmax,
                        lonmin=lonmin,
                        lonmax=lonmax,
                        vmin=0,
                        vmax=15,
                        interv=.5)

    files = os.listdir(output_dir)
    files = sorted(files)
    images = []
    for file in files:
        images.append(imageio.imread(output_dir+file))
        # os.remove(output_dir/file)
    imageio.mimsave(output_dir+f'{start_date}_{end_date}_{var}.gif',images, fps = 2)

if __name__ == '__main__':

    # Enter your inputs here:
    var = 'temperature'
    start_date='20190701'
    end_date='20190730'
   # isomask=[-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,]
    isomask=None
    isobaths=[0,200,2000]#None

    node_sel = 398428

    data_dir1=r"/work2/noaa/nosofs/felicioc/BeringSea/R07/outputs/"
    data_dir2=r"/work2/noaa/nos-surge/felicioc/BeringSea/R09a/outputs/"
    #data_dir=r"/work2/noaa/nosofs/felicioc/BeringSea/R07/outputs/"
    output_dir=r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/sst_ak_VERTICAL/"
    #output_dir=r"/work2/noaa/nos-surge/felicioc/BeringSea/P09/sst_ak_R07/"

    main(var,start_date,end_date,output_dir,isobaths,data_dir1,data_dir2,isomask,node_sel)
