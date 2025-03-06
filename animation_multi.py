import matplotlib.pyplot as plt
import matplotlib.tri as tri
import xarray as xr
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import imageio.v2 as imageio
import os
import multiprocessing
from itertools import islice
from pathlib import Path
from argparse import ArgumentParser
import sys
import argparse

'''
Example run:
python animation_multi.py -p PathToWW3OutputFile\ww3.202209.nc -v hs -m True -t 3

where:
        -p = path to .nc file,
        -o = dir path for the .jpeg and .gif outputs. If None an animation/ dir will be created under nc_path
        -v = ww3 variable, e.g. hs
        -s = initial model timestep, default = 0
        -e = final model timestep, default = -1
        -t = frequency of timestep, default = 1 (i.e. every timestep). if -t 2, then every other timestep
        -min =minimum variable (e.g. 'hs') for legend, default = var.min()
        -max =maximum variable (e.g. 'hs') for legend, default = var.max()/2
        -m = will use basemap for the background?, default = False
        -y = space between parallels (lat), default = 5deg
        -x = space between meridians (lon), default = 10deg
        -b = plot buffer, default = 1deg
        -n = name for legend, default=netcdf file long_name
'''

def ww3_gif_pre(nc_path,
            var='hs',
            time_start=0,
            time_end=-1,
            vmin=None,
            vmax=None,
            b_map=False,
            par_space=5,
            mer_space=10,
            mapping_buffer=1,
            long_name=None):
    '''
    pre-process ww3 outputs to create figure
    '''
    
    ds = xr.open_dataset(nc_path)
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    var_data = ds.variables[var][time_start:time_end].data

    # nv needs to be transposed if it's not in the (N, 3) shape
    nv = ds.variables['tri'][:] - 1  # Adjust for 0-based indexing if necessary
    if long_name is None:
        long_name = ds.variables[var].attrs['long_name'] if 'long_name' in ds.variables[var].attrs else var
    units = ds.variables[var].attrs['units'] if 'units' in ds.variables[var].attrs else 'no units'    
    extents = np.array((np.array(lon).min(),
                        np.array(lon).max(),
                        np.array(lat).min(),
                        np.array(lat).max()))

    if b_map is not None:
        m = Basemap(llcrnrlon=extents[0]-mapping_buffer,
                    llcrnrlat=extents[2]-mapping_buffer,
                    urcrnrlon=extents[1]+mapping_buffer,
                    urcrnrlat=extents[3]+mapping_buffer,
                    rsphere=(6378137.00, 6356752.3142),
                    resolution='l',
                    projection='cyl',
                    lat_0=extents[-2:].mean(),
                    lon_0=extents[:2].mean(),
                    lat_ts=extents[2:].mean(),
                    epsg=4326)
    else:
        m=None

    parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), par_space)
    meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), mer_space)

    if vmin is None:
        vmin = var_data[~np.isnan(var_data)].min()
    if vmax is None:
        vmax = var_data[~np.isnan(var_data)].max()/2

    triangulation = tri.Triangulation(lon, lat, triangles=nv)

    return triangulation,long_name,m,parallels,meridians,ds,vmin,vmax


def static_plot(triangulation,long_name,m,parallels,meridians,ds,vara,n,vmin,vmax,output_dir):
    fig = plt.figure(figsize=(4.5,3.5))  # units are inches for the size
    ax = fig.add_subplot(111)

    if m is not None:
        m.drawcoastlines()
        m.fillcontinents(color='#B0B0B0',alpha=0.65)
        m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=1)
        m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=1)
        m.arcgisimage(service='World_Imagery', xpixels = 500, verbose= False)
    
    tp = ax.tripcolor(triangulation, ds.variables[vara][n].data, shading='flat', cmap='jet', vmin=vmin, vmax=vmax)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp, cax=cax, extend='both',boundaries=np.arange(0,vmax))
    cb.set_label(long_name)
    ax.set_title(str(np.array(ds['time'][n])).split('.')[0])
    fig.tight_layout()

    fig.savefig(output_dir/'{:04d}.jpeg'.format(n))
    plt.close(fig)


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser(
        prog="animation_multi.py",
        usage="%(prog)s",
        description="Creates .gif from ww3.nc output",
    )
    parser_arg.add_argument(
        "-p",
        "--nc_path",
        required=True,
        help="path to .nc file",
    )
    parser_arg.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="dir path for the .jpeg and .gif outputs. If None an animation/ dir will be created under nc_path",
    )
    parser_arg.add_argument(
        "-v",
        "--ww3_variable",
        required=True,
        help="ww3 variable, e.g. 'hs'",
    )
    parser_arg.add_argument(
        "-s",
        "--time_start",
        required=False,
        help="initial model timestep, default = 0",
    )
    parser_arg.add_argument(
        "-e",
        "--time_end",
        required=False,
        help="final model timestep, default = -1",
    )
    parser_arg.add_argument(
        "-t",
        "--timestep",
        required=False,
        help="frequency of timestep, default = 1 (i.e. every timestep). if -t 2, then every other timestep",
    )
    parser_arg.add_argument(
        "-min",
        "--vmin",
        required=False,
        help="minimum variable (e.g. 'hs') for legend, default = var.min()",
    )
    parser_arg.add_argument(
        "-max",
        "--vmax",
        required=False,
        help="maximum variable (e.g. 'hs') for legend, default = var.max()/2",
    )
    parser_arg.add_argument(
        "-m",
        "--b_map",
        required=False,
        help="will use basemap for the background?, default = False",
    )
    parser_arg.add_argument(
        "-y",
        "--par_space",
        required=False,
        help="space between parallels (lat), default = 5deg",
    )
    parser_arg.add_argument(
        "-x",
        "--mer_space",
        required=False,
        help="space between meridians (lon), default = 10deg",
    )
    parser_arg.add_argument(
        "-b",
        "--mapping_buffer",
        required=False,
        help="plot buffer, default = 1deg",
    )
    parser_arg.add_argument(
        "-n",
        "--long_name",
        required=False,
        help="name for legend, default=netcdf file long_name",
    )    
    args = parser_arg.parse_args()

    nc_path=Path(args.nc_path)
    if args.output_dir is None:
        output_dir=nc_path.parent/"animation"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.time_start is None:
        args.time_start=0
    if args.time_end is None:
        args.time_end=-1
    if args.timestep is None:
        args.timestep=1
    if args.par_space is None:
        args.par_space=5
    if args.mer_space is None:
        args.mer_space=10
    if args.mapping_buffer is None:
        args.mapping_buffer=1

    gif_inp = ww3_gif_pre(nc_path,
                          var=args.ww3_variable,
                          time_start=args.time_start,
                          time_end=args.time_end,
                          vmin=args.vmin,
                          vmax=args.vmax,
                          b_map=args.b_map,
                          par_space=args.par_space,
                          mer_space=args.mer_space,
                          mapping_buffer=args.mapping_buffer,
                          long_name=args.long_name)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for n in range(int(args.time_start), len(gif_inp[5]['time'][int(args.time_start):int(args.time_end)]), int(args.timestep)):
        p = multiprocessing.Process(target=static_plot, args=(gif_inp[0],
                                                              gif_inp[1],
                                                              gif_inp[2],
                                                              gif_inp[3],
                                                              gif_inp[4],
                                                              gif_inp[5],
                                                              args.ww3_variable,
                                                              n,
                                                              gif_inp[6],
                                                              gif_inp[7],
                                                              output_dir,))
        p.start()
    p.join()

    files = os.listdir(output_dir)
    images = []
    for file in files:
        images.append(imageio.imread(output_dir/file))
        os.remove(output_dir/file)
    imageio.mimsave(output_dir/'animation.gif', images)
    
