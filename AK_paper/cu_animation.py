
import math
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.tri as tri
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import numpy.typing as npt
import typing as T


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



rundir=r"/work2/noaa/nosofs/felicioc/BeringSea/R07/"

x,y,connect_tri,depth = fixed_connectivity_tri(f'{rundir}/outputs/')
triangulation = tri.Triangulation(x=x, y=y, triangles=connect_tri)


interv=0.05
vmin, vmax = -1,1

for dt in range(1,121):

    #fields
    ds = xr.open_dataset(f'{rundir}/outputs/out2d_{dt}.nc',
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
                                            ])
    u = np.array(xr.open_dataset(f'{rundir}/outputs/horizontalVelX_{dt}.nc',
                        engine='h5netcdf',
                ).horizontalVelX[:,:,-1])
    v = np.array(xr.open_dataset(f'{rundir}/outputs/horizontalVelY_{dt}.nc',
                        engine='h5netcdf',
                ).horizontalVelY[:,:,-1])
    zeta = np.array(ds.elevation[:,:])

    #timeseries:
    zts = np.array(ds.elevation[:,490809-1])
    uts = np.array(xr.open_dataset(f'{rundir}/outputs/horizontalVelX_{dt}.nc',
                        engine='h5netcdf',
                ).horizontalVelX[:,490809-1,-1])
    vts = np.array(xr.open_dataset(f'{rundir}/outputs/horizontalVelY_{dt}.nc',
                        engine='h5netcdf',
                ).horizontalVelY[:,490809-1,-1])
    ts = np.array(ds.time)

    vel = np.array(uts**2 + vts**2) ** 0.5
    d = [math.atan2(vts,uts) / math.pi * 180 % 360.0 for uts, vts in zip(uts, vts)]
    d = [deg * np.pi / 180 for deg in d]

    i_ts,i_zts,i_vel,i_d = [],[],[],[]
    for hr_idx,t in enumerate(ts):

        z_hr_idx = zeta[hr_idx]
        u_hr_idx = u[hr_idx]
        v_hr_idx = v[hr_idx]
        ts_hr_idx = ts[hr_idx]
        zts_hr_idx = zts[hr_idx]
        vel_hr_idx = vel[hr_idx]
        i_d_hr_idx = d[hr_idx]
        
        i_ts.append(ts_hr_idx)
        i_zts.append(zts_hr_idx)
        i_vel.append(vel_hr_idx)
        i_d.append(i_d_hr_idx)

        
        cmap = plb.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(vmin, vmax, len([i for i in np.arange(vmin, vmax, interv)])+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create fields plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(211,projection=ccrs.PlateCarree(central_longitude=-180))
        
        # Create fields plot: SSH
        tp = ax.tripcolor(triangulation,z_hr_idx,transform=ccrs.PlateCarree(),shading='flat',cmap=cmap,norm=norm)
        cb = fig.colorbar(tp,
                        # cax=cax,
                        boundaries=np.arange(vmin,vmax,interv),
                        cmap=cmap,norm=norm,
                        ticks=np.linspace(vmin,vmax,len([i for i in np.arange(vmin,vmax,.5)])+1),
                        orientation='horizontal',
                        location='bottom',
                        aspect=50,
                        pad=0.01,
                        )
        cb.set_label("Zeta (m)")
        
        # Create fields plot: Cu
        ax.quiver(x, y, u_hr_idx, v_hr_idx, scale=50, transform=ccrs.PlateCarree(), width=0.0005, pivot='middle')

        #Zoom in
        ax.set_xlim(4.6,7.1)
        ax.set_ylim(51.95,52.45)
        #place of node:
        ax.plot(185.931, 52.119, 'rx', transform=ccrs.Geodetic())   
        # ax.add_feature(cfeature.COASTLINE)
        # ax.add_feature(cfeature.BORDERS)

        #Cu timeseries plot:
        ax2 = fig.add_subplot(212,)
        q = ax2.quiver(i_ts, i_zts, 
                i_vel * np.cos(i_d), 
                i_vel * np.sin(i_d), 
                color='k',
                scale=50,
                width=0.001,
                alpha=0.5,
                    )
        qk = ax2.quiverkey(q, 0.9, 0.1, 1, '1 m/s', labelpos='E', coordinates='figure')
        # ax2.set_yticks([])
        
        #Zeta timeseries plot
        ax3 = ax2.twinx()    
        ax3.plot(i_ts, i_zts, color='cyan',linewidth=0.5)
        ax2.set_ylabel('Zeta (m)', color='cyan')
        ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5,linewidth=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5,linewidth=0.5)
        ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.5,linewidth=0.5)
        ax3.set_xlim(pd.Timestamp('2019-07-01 00:00:00'), pd.Timestamp('2019-10-1 00:00:00'))
        ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.set_yticks([])
        
        #Cu Arrow timeseries plot (current timestep):
        ax4 = ax3.twinx()
        q2 = ax4.quiver(ts_hr_idx, zts_hr_idx, 
                vel_hr_idx * np.cos(i_d_hr_idx), 
                vel_hr_idx * np.sin(i_d_hr_idx), 
                color='r',
                scale=50,
                    width=0.0015)

        ax2.set_ylim(-2, 2)
        ax3.set_ylim(-2, 2)
        ax4.set_ylim(-2, 2)
        # Show the plot
        fig.tight_layout()
        fig.savefig(f"/work2/noaa/nosofs/felicioc/BeringSea/P07/cu_ak/cu_fig/{dt:03}_{hr_idx:03}.jpeg",dpi=300)
        plt.clf()
