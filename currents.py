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
import math

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
        dates.append(date)

    return dates

def open_schism(date, n, data_dir):
    """
    """
    n=n+1
    try:
        dsx = xr.open_dataset(
            f"{data_dir}/horizontalVelX_{n}.nc",
            chunks={},
            engine='h5netcdf',
            )
        dsy = xr.open_dataset(
            f"{data_dir}/horizontalVelY_{n}.nc",
            chunks={},
            engine='h5netcdf',
            )
        print("Model data found for: ", date)
    except:
        print("No model data found for: ", date)
        pass
    
    return dsx,dsy

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

def plot_vel(
               triangulation,
               vel,
               ang,
               depth=None,
               isobaths=None,
               output_dir="",
               n=0,
               long_name1="name of the var",
               long_name2="name of the var",
               time_str="time str",
               sat_time="time str",
               vmin=None,
               vmax=None,
               latmin=None,
               latmax=None,
               lonmin=None,
               lonmax=None,
               interv=None):
    
    cmap = plb.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[:13] = [[0.6, 0.6, 0.6, 1] for i in cmaplist[:13]]
    cmaplist[13:26] = [[0.4, 0.4, 0.4, 1] for i in cmaplist[13:26]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(211)
    tp = ax.tripcolor(triangulation,vel,shading='flat',cmap=cmap,norm=norm)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,
                      # norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,.1)])+1))
    cb.set_label(long_name1)

    ax.set_title(time_str)
    ax.set_facecolor('grey')
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)



    ax2 = fig.add_subplot(212)
    tp2 = ax2.tripcolor(triangulation,ang,shading='flat',cmap='hsv')
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.2)
    cb2 = fig.colorbar(tp2,
                       cax=cax2,
                       boundaries=np.arange(0,360,1),
                       ticks=np.linspace(0,360,len([i for i in np.arange(0,360,45)])+1)
                      )
    cb2.set_label(long_name2)

    ax2.set_title(sat_time)
    ax2.set_facecolor('grey')
    ax2.set_xlim(lonmin,lonmax)
    ax2.set_ylim(latmin,latmax)
    ax2.set_xlim(lonmin,lonmax)
    ax2.set_ylim(latmin,latmax)

    if depth is not None and isobaths is not None:
        depth[np.isnan(depth)] = -9999
        ax.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
        ax2.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")

    fig.tight_layout()
    # fig.show()
    fig.savefig(output_dir+'vel_dir.jpeg')
    plt.clf()




data_dir = r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\Summer2019_AK_Run\O04/"
output_dir = r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\Summer2019_AK_Run\Fig_Currents/"

start_date='20190702'
end_date='20190704'

long_name1="Current Velocity (m/s)"
long_name2="Current Direction (deg)"
isobaths=[50,100,200,500,1000,2000]#None

dates = dates_range(start_date, end_date)    
times = [dd.strftime("%m/%d/%y") for dd in dates]
x,y,connect_tri,depth = fixed_connectivity_tri(data_dir)
lonmin,lonmax=x.min(),x.max()
latmin,latmax=y.min(),y.max()
triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

x_vel_all,y_vel_all=[],[]
for n, date in enumerate(dates):
    dsx,dsy = open_schism(date, n, data_dir)
    x_vel = np.array(dsx.variables['horizontalVelX'][0,:,-1])
    y_vel = np.array(dsy.variables['horizontalVelY'][0,:,-1])
    x_vel_all.append(x_vel)
    y_vel_all.append(y_vel)

x_avg = np.mean(np.array(x_vel_all), axis=0)
y_avg = np.mean(np.array(y_vel_all), axis=0)

vel = (x_avg**2+y_avg**2)**.5
ang = np.array([math.atan2(y_avg[i],x_avg[i]) / math.pi * 180 % 360.0 for i in range(len(y_avg))])
ang[vel < .05] = np.nan

plot_vel(triangulation,
            vel,
            ang,
            depth=depth,
            isobaths=isobaths,
            output_dir=output_dir,
            n=0,
            long_name1=long_name1,
            long_name2=long_name2,
            time_str=f"Average from {start_date} to {end_date}",
            sat_time=f"Average from {start_date} to {end_date}",
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            vmin=0,
            vmax=.5,
            interv=.025)
