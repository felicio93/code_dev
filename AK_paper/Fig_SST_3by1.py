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
               triangulation1,
               triangulation2,
               z1,
               z2,
               x_sat,
               y_sat, 
               z_sat_n,
               depth1=None,
               depth2=None,
               isobaths=None,
               output_dir="",
               n=0,
               long_name="name of the var",
               time_str="time str",
               sat_time="time str",
               vmin=None,
               vmax=None,
               latmin1=None,
               latmax1=None,
               lonmin1=None,
               lonmax1=None,
               interv=None):
    
    cmap = plb.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    fig = plt.figure(figsize=(5,14))

    ax = fig.add_subplot(311)
    tp = ax.tripcolor(triangulation1,z1,shading='flat',cmap=cmap,norm=norm)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
    cb.set_label(long_name)

    ax.set_title(time_str)
    # ax.set_facecolor('grey')
    ax.set_xlim(lonmin1,lonmax1)
    ax.set_ylim(latmin1,latmax1)
    ax.set_xlim(lonmin1,lonmax1)
    ax.set_ylim(latmin1,latmax1)


    ax2 = fig.add_subplot(312)
    tp2 = ax2.tripcolor(triangulation2,z2,shading='flat',cmap=cmap,norm=norm)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.2)
    cb2 = fig.colorbar(tp2,
                      cax=cax2,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
    cb2.set_label(long_name)

    # ax2.set_title(time_str)
    # ax.set_facecolor('grey')
    ax2.set_xlim(lonmin1,lonmax1)
    ax2.set_ylim(latmin1,latmax1)
    ax2.set_xlim(lonmin1,lonmax1)
    ax2.set_ylim(latmin1,latmax1)


    ax3 = fig.add_subplot(313)
    tp3 = ax3.pcolor(x_sat,y_sat,z_sat_n,cmap=cmap,norm=norm)
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.2)
    cb3 = fig.colorbar(tp3,
                       cax=cax3,
                       boundaries=np.arange(vmin,vmax,interv),
                       cmap=cmap,
                       norm=norm,
                       ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
    cb3.set_label(long_name)

    # ax2.set_title(sat_time)
    # ax.set_facecolor('grey')
    ax3.set_xlim(lonmin1,lonmax1)
    ax3.set_ylim(latmin1,latmax1)
    ax3.set_xlim(lonmin1,lonmax1)
    ax3.set_ylim(latmin1,latmax1)

    if depth1 is not None and isobaths is not None:
        depth1[np.isnan(depth1)] = -9999
        depth2[np.isnan(depth2)] = -9999
        ax.tricontour(triangulation1, depth1, levels=isobaths, linewidths=.5, colors="k")
        ax2.tricontour(triangulation2, depth2, levels=isobaths, linewidths=.5, colors="k")

    fig.tight_layout()
    fig.savefig(output_dir+'{:04d}.jpeg'.format(n),dpi=300)
    plt.clf()

def main(var,start_date,end_date,output_dir,path_sat,isobaths,data_dir1,data_dir2):

    # ds = concat_arctic(start_date,end_date,data_dir)
    if var == 'temperature':
        long_name="Sea Surface Temperature (degC)"

    dates = dates_range(start_date, end_date)    
    times = [dd.strftime("%m/%d/%y") for dd in dates]

    x1,y1,connect_tri1,depth1 = fixed_connectivity_tri(data_dir1)
    lonmin1,lonmax1=x1.min(),x1.max()
    latmin1,latmax1=y1.min(),y1.max()
    triangulation1 = tri.Triangulation(x=x1, y=y1, triangles=connect_tri1)

    x2,y2,connect_tri2,depth2 = fixed_connectivity_tri(data_dir2)
    # lonmin2,lonmax2=x2.min(),x2.max()
    # latmin2,latmax2=y2.min(),y2.max()
    triangulation2 = tri.Triangulation(x=x2, y=y2, triangles=connect_tri2)

    # sat_date = [str(i).split("T")[0].replace('-', '') for i in np.array(ds['time'])]
    sat_date = [dd.strftime("%Y%m%d") for dd in dates]
    ds_sat = concat_LEO(sat_date, path_sat)
    x_sat=np.array(ds_sat['lon'][0])
    y_sat=np.array(ds_sat['lat'][0])
    z_sat=np.array(ds_sat['sst'])
    x_sat, y_sat = np.meshgrid(x_sat, y_sat)

    # ds = np.array(ds.variables[var][:,:,-1])

    for n, date in enumerate(dates):
        ds1 = open_schism(date, n, data_dir1)
        ds1 = np.array(ds1.variables[var][11,:,-1])
        z1 = ds1

        ds2 = open_schism(date, n, data_dir2)
        ds2 = np.array(ds2.variables[var][11,:,-1])
        z2 = ds2
   
        print("creating plot ",n," of ", len(dates))
        time_str=times[n]
        sat_time_n=sat_date[n]
        z_sat_n = z_sat[n]
        if var == 'temperature':
            plot_arctic(triangulation1,
                        triangulation2,
                        z1,
                        z2,
                        x_sat,
                        y_sat, 
                        z_sat_n,
                        depth1=depth1,
                        depth2=depth2,
                        isobaths=isobaths,
                        output_dir=output_dir,
                        n=n,
                        long_name=long_name,
                        time_str=time_str,
                        sat_time=sat_time_n,
                        latmin1=latmin1,
                        latmax1=latmax1,
                        lonmin1=lonmin1,
                        lonmax1=lonmax1,
                        vmin=13,
                        vmax=33,
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
    start_date='20210701'
    end_date='20211001'
    isobaths=[0,200,2000]#None

    #data_dir=r"/work2/noaa/nosofs/felicioc/STOFS-3D-Atl-Example-Setup/3D_setup/STOFS3D-v8/R08a/outputs/"
    data_dir1="/work2/noaa/nos-surge/felicioc/OCSMesh_Paper/step_07_Run/R12/outputs/"
    data_dir2="/work2/noaa/nos-surge/felicioc/OCSMesh_Paper/step_07_Run/R11a/outputs/"
    output_dir=r"/work2/noaa/nos-surge/felicioc/OCSMesh_Paper/step_07_Run/P_11a_12/sst_STOFS_OCSMESH/"
    path_sat=r"/work2/noaa/nosofs/felicioc/STOFS-3D-Atl-Example-Setup/3D_setup/STOFS3D-v8/P08/script_LEOL3/subset/"
    
    main(var,start_date,end_date,output_dir,path_sat,isobaths,data_dir1,data_dir2)
