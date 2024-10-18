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

def concat_stofs_pac(start_date, end_date, averaged=True):
    """
    This function creates climatology (average) files for
    STOFS Pacific data based on a start and end dates, e.g.:
    clim_stofs_pac("20240801", "20240803")
    """
    dates = dates_range(start_date, end_date)
    ds_all=[]
    for date in dates:
        print(date)
        try:
            fs = fsspec.filesystem("s3", anon=True)
            ds1 = xr.open_dataset(
                fs.open(
                    f"s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac/para/stofs_3d_pac.{date.strftime('%Y%m%d')}/stofs_3d_pac.t12z.field2d_n001_012.nc"#this is tri only
                ),
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
                                # 'depth',
                                ]
            )
            
            ds2 = xr.open_dataset(
                fs.open(
                    f"s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac/para/stofs_3d_pac.{date.strftime('%Y%m%d')}/stofs_3d_pac.t12z.field2d_n013_024.nc"#this is tri only
                ),
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
                                # 'depth',
                                ]
            )
        
            ds = xr.concat([ds1,ds2],dim="time",data_vars="all")

            if averaged==True:
                ds = ds.mean(dim='time')
            ds_all.append(ds)

        except:
            ds = xr.open_dataset(
                fs.open(
                    f"s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac/para/stofs_3d_pac.{date.strftime('%Y%m%d')}/stofs_3d_pac.t12z.n001_024.field2d.nc"#this is tri only
                ),
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
                                # 'depth'
                                ]
            )
            if averaged==True:
                ds = ds.mean(dim='time')
            ds_all.append(ds)

    ds = xr.concat(ds_all,dim="time",data_vars="all")
    if averaged==True:
        ds = ds.mean(dim='time')

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
            "s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac/para/stofs_3d_pac.20241001/stofs_3d_pac.t12z.fields.out2d_n001_012.nc"#this is tri only
        ),
        chunks={},
        engine='h5netcdf',
        drop_variables=['dryFlagNode','evaporationRate','precipitationRate','windSpeedX','windSpeedY','windStressX','windStressY','dryFlagElement','dryFlagSide']
    )
    
    connect=np.array(ds_tri['SCHISM_hgrid_face_nodes'][:])-1
    connect_tri=split_quads(np.array(connect))
    x, y = np.array(ds_tri['SCHISM_hgrid_node_x']), np.array(ds_tri['SCHISM_hgrid_node_y'])

    return x,y,connect_tri

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
               latmin2=None,
               latmax2=None,
               lonmin2=None,
               lonmax2=None,
               vmin2=None,
               vmax2=None,
               latmin3=None,
               latmax3=None,
               lonmin3=None,
               lonmax3=None,
               vmin3=None,
               vmax3=None,
               interv=None):
    

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(212)
    tp = ax.tripcolor(triangulation, z,shading='gouraud',cmap='jet', vmin=vmin1, vmax=vmax1)#'gouraud' = linearly interp. 'flat'=averaged to tri
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp, cax=cax, extend='both',boundaries=np.arange(vmin1,vmax1,interv))
    cb.set_label(long_name)
    rect1 = patches.Rectangle((lonmin2, latmin2), lonmax2-lonmin2, latmax2-latmin2, linewidth=1, edgecolor='k', facecolor='none')
    rect2 = patches.Rectangle((lonmin3, latmin3), lonmax3-lonmin3, latmax3-latmin3, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.set_title(time_str)
    ax.set_facecolor('grey')
    
    ax2 = fig.add_subplot(221)
    ax2.set_xlim(lonmin2,lonmax2)
    ax2.set_ylim(latmin2,latmax2)
    tp2 = ax2.tripcolor(triangulation, z,shading='gouraud',cmap='jet', vmin=vmin2, vmax=vmax2)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.2)
    cb2 = fig.colorbar(tp2, cax=cax2, extend='both',boundaries=np.arange(vmin2,vmax2,interv))
    ax2.set_facecolor('grey')
    
    ax3 = fig.add_subplot(222)
    ax3.set_xlim(lonmin3,lonmax3)
    ax3.set_ylim(latmin3,latmax3)
    tp3 = ax3.tripcolor(triangulation, z,shading='gouraud',cmap='jet', vmin=vmin3, vmax=vmax3)
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.2)
    cb3 = fig.colorbar(tp3, cax=cax3, extend='both',boundaries=np.arange(vmin3,vmax3,interv))
    ax3.set_facecolor('grey')

    if depth is not None and isobaths is not None:
        depth[np.isnan(depth)] = -9999
        ax2.tricontour(triangulation, depth, levels=isobaths, linewidths=.75, colors="k")
        ax3.tricontour(triangulation, depth, levels=isobaths, linewidths=.75, colors="k")

    fig.tight_layout()

    if isinstance(n, str):
        fig.savefig(output_dir+'{}.jpeg'.format(n))
    else:
        fig.savefig(output_dir+'{:04d}.jpeg'.format(n))

    # fig.savefig(output_dir+'{:04d}.jpeg'.format(n))
    plt.clf()

def main(var,start_date,end_date,averaged,output_dir,timestep,isobaths):

    ds = concat_stofs_pac(start_date,end_date,averaged=averaged)
    if var == 'temp_surface':
        long_name="Sea Surface Temperature (degC)"
    if var == 'salt_surface':
        long_name="Sea Surface Salinity (psu)"
    if var == 'elev':
        long_name="Sea Surface Height (m)"
    if averaged==False:
        times=[str(tt).split('.')[0] for tt in np.array(ds['time'][:])]
    else:
        times=f"average from {start_date} to {end_date}"

    if isobaths is not None and averaged == False:
        depth=np.array(ds.variables['depth'][0])
    if isobaths is not None and averaged == True:
        depth=np.array(ds.variables['depth'])
    else:
        depth=None

    ds = np.array(ds.variables[var])
    print("here is depth:",depth)
    x,y,connect_tri = fixed_connectivity_tri()
    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)
    if averaged==False:
        for n in range(0, len(ds), timestep):
            print(n)
            z = ds[n]
            time_str=times[n]
            if var == 'temp_surface':
                plot_stofs_pac(triangulation,
                            z,
                            depth=depth,
                            isobaths=isobaths,
                            output_dir=output_dir,
                            n=n,
                            long_name=long_name,
                            time_str=time_str,
                            vmin1=0,
                            vmax1=32,
                            latmin2=45,
                            latmax2=67,
                            lonmin2=170,
                            lonmax2=200,
                            vmin2=5,
                            vmax2=15,
                            latmin3=42,
                            latmax3=47,
                            lonmin3=233,
                            lonmax3=238,
                            vmin3=10,
                            vmax3=20,
                            interv=0.5,
                )
            if var == 'salt_surface':
                plot_stofs_pac(triangulation,
                            z,
                            depth=depth,
                            isobaths=isobaths,
                            output_dir=output_dir,
                            n=n,
                            long_name=long_name,
                            time_str=time_str,
                            vmin1=30,
                            vmax1=37,
                            latmin2=45,
                            latmax2=67,
                            lonmin2=170,
                            lonmax2=200,
                            vmin2=28,
                            vmax2=34.5,
                            latmin3=42,
                            latmax3=47,
                            lonmin3=233,
                            lonmax3=238,
                            vmin3=27,
                            vmax3=33.5,
                            interv=0.5,
                )
            if var == 'elev':
                plot_stofs_pac(triangulation,
                            z,
                            depth=depth,
                            isobaths=isobaths,
                            output_dir=output_dir,
                            n=n,
                            long_name=long_name,
                            time_str=time_str,
                            vmin1=-2.0,
                            vmax1=2.0,
                            latmin2=45,
                            latmax2=67,
                            lonmin2=170,
                            lonmax2=200,
                            vmin2=-1,#5,
                            vmax2=1,#15,
                            latmin3=42,
                            latmax3=47,
                            lonmin3=233,
                            lonmax3=238,
                            vmin3=-1.5,
                            vmax3=1.5,
                            interv=0.1,
                )
        files = os.listdir(output_dir)
        images = []
        for file in files:
            images.append(imageio.imread(output_dir+file))
            # os.remove(output_dir/file)
        imageio.mimsave(output_dir+f'{start_date}_{end_date}_{var}.gif',images)
        
    else:
        z=ds
        if var == 'temp_surface':
            plot_stofs_pac(triangulation,
                        z,
                        depth=depth,
                        isobaths=isobaths,
                        output_dir=output_dir,
                        n="average_temp_surface",
                        long_name=long_name,
                        time_str=times,
                        vmin1=0,
                        vmax1=32,
                        latmin2=45,
                        latmax2=67,
                        lonmin2=170,
                        lonmax2=200,
                        vmin2=5,
                        vmax2=15,
                        latmin3=42,
                        latmax3=47,
                        lonmin3=233,
                        lonmax3=238,
                        vmin3=10,
                        vmax3=20,
                        interv=0.5,
            )
        if var == 'salt_surface':
            plot_stofs_pac(triangulation,
                        z,
                        depth=depth,
                        isobaths=isobaths,
                        output_dir=output_dir,
                        n="average_salt_surface",
                        long_name=long_name,
                        time_str=times,
                        vmin1=30,
                        vmax1=37,
                        latmin2=45,
                        latmax2=67,
                        lonmin2=170,
                        lonmax2=200,
                        vmin2=28,
                        vmax2=34.5,
                        latmin3=42,
                        latmax3=47,
                        lonmin3=233,
                        lonmax3=238,
                        vmin3=27,
                        vmax3=33.5,
                        interv=0.5,
            )
        if var == 'elev':
            plot_stofs_pac(triangulation,
                        z,
                        depth=depth,
                        isobaths=isobaths,
                        output_dir=output_dir,
                        n="average_height_surface",
                        long_name=long_name,
                        time_str=times,
                        vmin1=-2.0,
                        vmax1=2.0,
                        latmin2=45,
                        latmax2=67,
                        lonmin2=170,
                        lonmax2=200,
                        vmin2=-1,#5,
                        vmax2=1,#15,
                        latmin3=42,
                        latmax3=47,
                        lonmin3=233,
                        lonmax3=238,
                        vmin3=-1.5,
                        vmax3=1.5,
                        interv=0.1,
            )


if __name__ == '__main__':

    # Enter your inputs here:
    var = 'temp_surface'#'elev'#'salt_surface'#'temp_surface'
    start_date='20240801'
    end_date='20240801'
    isobaths=[200,2000]#None
    averaged=True #if True it calculates the climatology, if False it will be one file per timestep
    output_dir=r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\STOFS_postprocessing\figures\2024_08_sst/"
    timestep=6 #every X hours
    main(var,start_date,end_date,averaged,output_dir,timestep,isobaths)
