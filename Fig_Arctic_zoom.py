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



def open_schism(n, data_dir):
    """
    """
    n=n+1
    try:
        ds_t = xr.open_dataset(
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
        print("Model TEMPERATURE data found for: ", n)
    except:
        print("No model TEMPERATURE data found for: ", n)
        pass
    try:
        ds_s = xr.open_dataset(
                f"{data_dir}salinity_{n}.nc",
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
        print("Model SALINITY data found for: ", n)
    except:
        print("No model SALINITY data found for: ", n)
        pass

    try:
        ds_h = xr.open_dataset(
                f"{data_dir}out2d_{n}.nc",
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
        print("Model ELEVATION data found for: ", n)
    except:
        print("No model ELEVATION data found for: ", n)
        pass

    return ds_t,ds_s,ds_h

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

def concat_LEO(sat_date, path_sat):
    """
    concatenates all LEO sat data on path_sat directory,
    sat_date is the list of names of the files to be concat, e.g. 20180801.nc
    """
    
    ds_sat=[]
    for date in sat_date:
        try:
            ds_sat.append(xr.open_dataset(r"{}/leosst_{}.nc".format(path_sat,date),engine='h5netcdf'))
        except:
            print("No satellite data found for: ", date)
    ds_sat = xr.concat(ds_sat,dim="time",data_vars="all")

    return ds_sat


def plot_arctic(
               triangulation,
                z_th,
                z_sh,
	            z_hh,
               depth=None,
               isobaths=None,
               output_dir="",
               n=0,
               long_name_t="name of the var",
               long_name_s="name of the var",
               long_name_h="name of the var",
               time_str="time str",
               vmin_t=None,
               vmax_t=None,
               vmin_s=None,
               vmax_s=None,
               vmin_h=None,
               vmax_h=None,
               latmin=None,
               latmax=None,
               lonmin=None,
               lonmax=None,
               interv_t=None,
               interv_s=None,
               interv_h=None,
               ):
    
    cmap = plb.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    ## SST Plots
    bounds_t = np.linspace(vmin_t, vmax_t, len([i for i in np.arange(vmin_t, vmax_t, interv_t)])+1)
    norm_t = mpl.colors.BoundaryNorm(bounds_t, cmap.N)
    
    fig = plt.figure(figsize=(30,10))

    ax = fig.add_subplot(131)
    tp = ax.tripcolor(triangulation,z_th,shading='flat',cmap=cmap,norm=norm_t)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin_t,vmax_t,interv_t),
                      cmap=cmap,norm=norm_t,
                      ticks=np.linspace(vmin_t,vmax_t,len([i for i in np.arange(vmin_t,vmax_t,1)])+1))
    cb.set_label(long_name_t)

    ax.set_title(time_str+" ROMS Hotstart")
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)


    ## SSS Plots
    bounds_s = np.linspace(vmin_s, vmax_s, len([i for i in np.arange(vmin_s, vmax_s, interv_s)])+1)
    norm_s = mpl.colors.BoundaryNorm(bounds_s, cmap.N)
    
    ax2 = fig.add_subplot(132)
    tp2 = ax2.tripcolor(triangulation,z_sh,shading='flat',cmap=cmap,norm=norm_s)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.2)
    cb2 = fig.colorbar(tp2,
                      cax=cax2,
                      boundaries=np.arange(vmin_s,vmax_s,interv_s),
                      cmap=cmap,norm=norm_s,
                      ticks=np.linspace(vmin_s,vmax_s,len([i for i in np.arange(vmin_s,vmax_s,.5)])+1))
    cb2.set_label(long_name_s)

    ax2.set_title(time_str+" ROMS Hotstart")
    ax2.set_xlim(lonmin,lonmax)
    ax2.set_ylim(latmin,latmax)
    ax2.set_xlim(lonmin,lonmax)
    ax2.set_ylim(latmin,latmax)


    ## SSH Plots
    bounds_h = np.linspace(vmin_h, vmax_h, len([i for i in np.arange(vmin_h, vmax_h, interv_h)])+1)
    norm_h = mpl.colors.BoundaryNorm(bounds_h, cmap.N)
    
    ax3 = fig.add_subplot(133)
    tp3 = ax3.tripcolor(triangulation,z_hh,shading='flat',cmap=cmap,norm=norm_h)
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.2)
    cb3 = fig.colorbar(tp3,
                      cax=cax3,
                      boundaries=np.arange(vmin_h,vmax_h,interv_h),
                      cmap=cmap,norm=norm_h,
                      ticks=np.linspace(vmin_h,vmax_h,len([i for i in np.arange(vmin_h,vmax_h,.1)])+1))
    cb3.set_label(long_name_h)

    ax3.set_title(time_str+" ROMS Hotstart")
    ax3.set_xlim(lonmin,lonmax)
    ax3.set_ylim(latmin,latmax)
    ax3.set_xlim(lonmin,lonmax)
    ax3.set_ylim(latmin,latmax)

    if depth is not None and isobaths is not None:
        depth[np.isnan(depth)] = -9999
        ax.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
        ax2.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
        ax3.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")

        # Add a line
        x_line = [169.02, 169.02]
        y_line = [59.05, 60.55]
        ax.plot(x_line, y_line, color='k')
        ax2.plot(x_line, y_line, color='k')
        ax3.plot(x_line, y_line, color='k')

    fig.tight_layout()
    fig.savefig(output_dir+'{:04d}.jpeg'.format(n))
    plt.close()
    plt.clf()

def main(timesteps,output_dir,isobaths,data_dir):

    long_name_t="Sea Surface Temperature (degC)"
    long_name_s="Sea Surface Salinity (PSU)"
    long_name_h="Sea Surface Height (m)"

    #dates = dates_range(start_date, end_date)    
    #times = [dd.strftime("%m/%d/%Y %H:%M") for dd in dates]
    x,y,connect_tri,depth = fixed_connectivity_tri(data_dir)
    lonmin,lonmax=168.3,170 #x.min(),x.max()
    latmin,latmax=59,60.6 #y.min(),y.max()

    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

    for n in range(timesteps):
        ds_th,ds_sh,ds_hh = open_schism(n, data_dir)
        time_str=str(np.array(ds_th.variables["time"])[0]).split(".")[0]
        print(time_str)

        ds_th = np.array(ds_th.variables["temperature"][0,:,-1])
        ds_sh = np.array(ds_sh.variables["salinity"][0,:,-1])
        ds_hh = np.array(ds_hh.variables["elevation"][0,:])

        z_th = ds_th
        z_sh = ds_sh
        z_hh = ds_hh

        plot_arctic(triangulation,
                    z_th,
                    z_sh,
                    z_hh,
                    depth=depth,
                    isobaths=isobaths,
                    output_dir=output_dir,
                    n=n,
                    long_name_t=long_name_t,
                    long_name_s=long_name_s,
                    long_name_h=long_name_h,
                    time_str=time_str,
                    latmin=latmin,
                    latmax=latmax,
                    lonmin=lonmin,
                    lonmax=lonmax,
                    vmin_t=0,
                    vmax_t=12,
                    vmin_s=27,
                    vmax_s=33.5,
                    vmin_h=-1,
                    vmax_h=1,
                    interv_t=.5,
                    interv_s=.25,
                    interv_h=.05,
                    )

    files = os.listdir(output_dir)
    files = sorted(files)
    images = []
    for file in files:
        images.append(imageio.imread(output_dir+file))
        # os.remove(output_dir/file)
    imageio.mimsave(output_dir+f'sstsssssh.gif',images, fps = 2)

if __name__ == '__main__':

    # Enter your inputs here:
    # var = 'temperature'
    #start_date='20190702'
    #end_date='20191030'
    timesteps=168
    isobaths=[200,2000]#None

    data_dir=r"/work2/noaa/nosofs/felicioc/BeringSea/R04a_hr/outputs/"
    #data_dir_r=r"/work2/noaa/nosofs/felicioc/BeringSea/O04/"
    output_dir=r"/work2/noaa/nosofs/felicioc/BeringSea/P04/sst_sss_zoom/"
    
    main(timesteps,output_dir,isobaths,data_dir)
