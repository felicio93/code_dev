#!/usr/bin/python3

import os
import fsspec
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
import matplotlib

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

def concat_stofs_pac(date, n, averaged=True):
    """
    This function creates climatology (average) files for
    STOFS Pacific data based on a start and end dates, e.g.:
    clim_stofs_pac("20240801", "20240803")
    """
    # dates = dates_range(start_date, end_date)

    n=n+1
    fs = fsspec.filesystem("s3", anon=True)
    try:
        ds = xr.open_dataset(
            fs.open(
                f"s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac-shadow-VIMS/hindcasts/2018/RUN29c/outputs/temperature_{n}.nc"
            ),
            chunks={},
            engine='h5netcdf',
            drop_variables=['vvel4.5','uvel4.5','vvel_bottom','uvel_bottom','vvel_surface','uvel_surface','salt_bottom','temp_bottom',
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

    #     # if averaged==True:
    #     #     ds = ds.mean(dim='time')
    #     # ds = ds.isel(nSCHISM_vgrid_layers=-1)
        print("Model data found for: ", date)
    except:
        print("No model data found for: ", date)
        pass

    # ds = xr.concat(ds_all,dim="time",data_vars="all")
    # if averaged==True:
    #     ds = ds.mean(dim='time')

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

def fixed_connectivity_tri():
    fs = fsspec.filesystem("s3", anon=True)
    ds_tri = xr.open_dataset(
        fs.open(
            "s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac-shadow-VIMS/hindcasts/2018/RUN29c/outputs/out2d_10.nc"#this is tri only
        ),
        chunks={},
        engine='h5netcdf',
        drop_variables=['dryFlagNode','evaporationRate','precipitationRate','windSpeedX','windSpeedY','windStressX','windStressY','dryFlagElement','dryFlagSide']
    )
    
    connect=np.array(ds_tri['SCHISM_hgrid_face_nodes'][:])-1
    connect_tri=split_quads(np.array(connect))
    x, y = np.array(ds_tri['SCHISM_hgrid_node_x']), np.array(ds_tri['SCHISM_hgrid_node_y'])
    depth=np.array(ds_tri.variables['depth'])

    return x,y,connect_tri,depth

def plot_stofs_pac(
               triangulation,
               z,
               depth=None,
               isobaths=None,
               output_dir="",
               n=0,
               long_name="name of the var",
               time_str="time str",
               vmin1=None,
               vmax1=None,
               latmin=None,
               latmax=None,
               lonmin=None,
               lonmax=None,
               interv=None):
    

    fig = plt.figure(figsize=(7,10))
    ax = fig.add_subplot(111)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    tp = ax.tripcolor(triangulation, z,shading='gouraud',cmap='jet', vmin=vmin1, vmax=vmax1)#'gouraud' = linearly interp. 'flat'=averaged to tri
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin1,vmax1,0.25),
                      extend='both',
                    #   orientation='horizontal',
                    #   location='bottom',
                      ticks=np.linspace(vmin1,vmax1,len([i for i in np.arange(vmin1,vmax1,1)])+1))#,interv))
    cb.set_label(long_name)
    ax.set_title(time_str)
    ax.set_facecolor('grey')
    
    if depth is not None and isobaths is not None:
        depth[np.isnan(depth)] = -9999
        ax.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")

    fig.tight_layout()


    fig.savefig(output_dir+'{:04d}.jpeg'.format(n))

    # fig.savefig(output_dir+'{:04d}.jpeg'.format(n))
    plt.clf()
    matplotlib.pyplot.close()

def main(var,start_date,end_date,averaged,output_dir,timestep,isobaths):

    if var == 'temperature':
        long_name="Sea Surface Temperature (degC)"
    dates = dates_range(start_date, end_date)    
    times = [dd.strftime("%m/%d/%y") for dd in dates]

    x,y,connect_tri,depth = fixed_connectivity_tri()
    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

    for n, date in enumerate(dates):

        ds = concat_stofs_pac(date,n,averaged=averaged)
        ds = np.array(ds.variables[var][12,:,-1])
    
        time_str=times[n]
        if var == 'temperature':
            plot_stofs_pac(triangulation,
                        ds,
                        depth=depth,
                        isobaths=isobaths,
                        output_dir=output_dir,
                        n=n,
                        long_name=long_name,
                        time_str=time_str,
                        vmin1=10,
                        vmax1=20,
                        latmin=42,
                        latmax=47,
                        lonmin=233,
                        lonmax=238,
                        interv=0.5,
            )

    files = os.listdir(output_dir)
    images = []
    for file in files:
        images.append(imageio.imread(output_dir+file))
        # os.remove(output_dir/file)
    imageio.mimsave(output_dir+f'{start_date}_{end_date}_{var}.gif',images,loop=0)
    

if __name__ == '__main__':

    # Enter your inputs here:
    var = 'temperature'#'elev'#'salt_surface'#'temp_surface'
    start_date='20180101'
    end_date='20181231'
    isobaths=[200,2000]#None
    averaged=True #if True it calculates the climatology, if False it will be one file per timestep
    output_dir=r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\STOFS_postprocessing\figures\hindcast_Temp_Pac/"
    timestep=6 #every X hours
    main(var,start_date,end_date,averaged,output_dir,timestep,isobaths)
