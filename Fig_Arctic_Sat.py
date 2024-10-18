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

def concat_arctic(start_date, end_date):
    """
    This function creates climatology (average) files for
    STOFS Pacific data based on a start and end dates, e.g.:
    clim_stofs_pac("20240801", "20240803")
    """
    dates = dates_range(start_date, end_date)
    ds_all=[]
    for n, date in enumerate(dates):
        n=n+1
        fs = fsspec.filesystem("s3", anon=True)
        try:
            ds1 = xr.open_dataset(
                fs.open(
                    f"s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac-shadow-VIMS/hindcasts/BeringSea/RUN05s/outputs/temperature_{n}.nc"
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
            ds_all.append(ds1)
            print("Model data found for: ", date)
        except:
            print("No model data found for: ", date)
    ds = xr.concat(ds_all,dim="time",data_vars="all")
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
            "s3://noaa-nos-stofs3d-pds/STOFS-3D-Pac-shadow-VIMS/hindcasts/BeringSea/RUN05s/outputs/out2d_30.nc"
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
               z,
               x_sat,
               y_sat, 
               z_sat_n,
               depth=None,
               isobaths=None,
               output_dir="",
               n=0,
               long_name="name of the var",
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
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(211)
    tp = ax.tripcolor(triangulation,z,shading='flat',cmap=cmap,norm=norm)
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
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)


    ax2 = fig.add_subplot(212)
    tp2 = ax2.pcolor(x_sat,y_sat,z_sat_n,cmap=cmap,norm=norm)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.2)
    cb2 = fig.colorbar(tp2,
                       cax=cax2,
                       boundaries=np.arange(vmin,vmax,interv),
                       cmap=cmap,
                       norm=norm,
                       ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,1)])+1))
    cb2.set_label(long_name)

    ax2.set_title(sat_time)
    # ax.set_facecolor('grey')
    ax2.set_xlim(lonmin,lonmax)
    ax2.set_ylim(latmin,latmax)
    ax2.set_xlim(lonmin,lonmax)
    ax2.set_ylim(latmin,latmax)

    if depth is not None and isobaths is not None:
        depth[np.isnan(depth)] = -9999
        ax.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")
        ax2.tricontour(triangulation, depth, levels=isobaths, linewidths=.5, colors="k")

    fig.tight_layout()
    fig.savefig(output_dir+'{:04d}.jpeg'.format(n))
    plt.clf()

def main(var,start_date,end_date,output_dir,path_sat,isobaths):

    ds = concat_arctic(start_date,end_date)
    if var == 'temperature':
        long_name="Sea Surface Temperature (degC)"

    times=[str(tt).split('.')[0] for tt in np.array(ds['time'][:])]
    x,y,connect_tri,depth = fixed_connectivity_tri()
    lonmin,lonmax=x.min(),x.max()
    latmin,latmax=y.min(),y.max()

    triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)

    sat_date = [str(i).split("T")[0].replace('-', '') for i in np.array(ds['time'])]

    ds_sat = concat_LEO(sat_date, path_sat)
    sat_time=[str(i).split(".")[0] for i in np.array(ds_sat['time'])]
    x_sat=np.array(ds_sat['lon'][0])
    y_sat=np.array(ds_sat['lat'][0])
    z_sat=np.array(ds_sat['sst'])
    x_sat, y_sat = np.meshgrid(x_sat, y_sat)

    ds = np.array(ds.variables[var][:,:,-1])

    for n in range(0, len(ds)):
        print("creating plot ",n," of ", len(ds))
        z = ds[n]
        time_str=times[n]
        sat_time_n=sat_time[n]
        z_sat_n = z_sat[n]
        if var == 'temperature':
            plot_arctic(triangulation,
                        z,
                        x_sat,
                        y_sat, 
                        z_sat_n,
                        depth=depth,
                        isobaths=isobaths,
                        output_dir=output_dir,
                        n=n,
                        long_name=long_name,
                        time_str=time_str,
                        sat_time=sat_time_n,
                        latmin=latmin,
                        latmax=latmax,
                        lonmin=lonmin,
                        lonmax=lonmax,
                        vmin=5,
                        vmax=15,
                        interv=.5)

    files = os.listdir(output_dir)
    images = []
    for file in files:
        images.append(imageio.imread(output_dir+file))
        # os.remove(output_dir/file)
    imageio.mimsave(output_dir+f'{start_date}_{end_date}_{var}.gif',images, fps = 1/(len(images)/60))

if __name__ == '__main__':

    # Enter your inputs here:
    var = 'temperature'
    start_date='20240702'
    end_date='20240930'
    isobaths=[200,2000]#None

    output_dir=r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\STOFS_postprocessing\figures\arctic_sat_sst/"
    path_sat=r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\STOFS_postprocessing\LEO/"
    main(var,start_date,end_date,output_dir,path_sat,isobaths)
