
from __future__ import annotations
import xarray as xr
import numpy as np

import ocsmesh


import pathlib

import matplotlib.pyplot
import netCDF4
import pyfes
import pandas as pd

import utide
import numpy.typing as npt
import typing as T

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.tri as tri
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        engine='netcdf4',
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



rundir = r"C:\Users\Felicio.Cassalho\Work\Modeling\AK_Project\model_runs\2019_07_01_12+120\R07"
ds = xr.open_dataset(f'{rundir}/elev_combine.nc',
                        engine='netcdf4',
                        )
time = np.array(pd.date_range('2019-07-01T13:00:00','2019-10-31T12:00:00',freq='H').astype('datetime64[us]'))

h = np.array(ds['z'][:,:])
x,y,connect_tri,depth = fixed_connectivity_tri(f'{rundir}/run/outputs/')
amp,phs = [],[]
for i in range(len(h[0,:])):
    print(i,":",len(h[0,:]))
    coef = utide.solve(
        time,
        h[:,i],
        lat=y[i],
        constit=["M2", "S2", "K1", "O1", "N2", "K2", "P1", "Q1", "S1"],
        method="ols",
        conf_int='linear',
        order_constit=["M2", "S2", "K1", "O1", "N2", "K2", "P1", "Q1", "S1"],
        verbose=False,
                      )
    amp.append(coef['A'])
    phs.append(coef['g'])
# .append(coef['aux']['frq'])

df_amp = pd.DataFrame(amp, columns=["M2", "S2", "K1", "O1", "N2", "K2", "P1", "Q1", "S1"])
df_phs = pd.DataFrame(phs, columns=["M2", "S2", "K1", "O1", "N2", "K2", "P1", "Q1", "S1"])

df_amp.to_csv(f'{rundir}/df_amp.csv', index=False)
df_phs.to_csv(f'{rundir}/df_phs.csv', index=False)


### Plot: ###
df_amp = pd.read_csv(f'{rundir}/df_amp.csv')
df_phs = pd.read_csv(f'{rundir}/df_phs.csv')

x,y,connect_tri,depth = fixed_connectivity_tri(f'{rundir}/run/outputs/')
triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)
lonmin,lonmax=x.min(),x.max()
latmin,latmax=y.min(),y.max()

vmin, vmax, interv = 0,1,0.05
phases = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]

for column_name in df_amp.columns:

    print(column_name)
    cmap = plb.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    tp = ax.tripcolor(triangulation,df_amp[column_name],shading='flat',cmap=cmap,norm=norm)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(tp,
                      cax=cax,
                      boundaries=np.arange(vmin,vmax,interv),
                      cmap=cmap,norm=norm,
                      ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,.5)])+1))
    cb.set_label("Tidal Amplitude (m)")
    
    ax.set_title(column_name)
    ax.set_facecolor('grey')
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)
    
    tricontour = ax.tricontour(triangulation, df_phs[column_name], levels=phases, linewidths=.5, colors="white")
    ax.clabel(tricontour, inline=True, fontsize=7., colors="k")
    
    fig.tight_layout()
    # fig.show()
    fig.savefig(f'{rundir}/tidal_maps/{column_name}.jpeg',dpi=300)
    plt.clf()
