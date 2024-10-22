import pyproj
from pylib import schism_grid as read_hgrid
import numpy as np
pyproj.network.set_network_enabled(active=True)

def navd88_to_xgeoid20b(lat,lon,z,t):
    nad832011_navd88geoid18_to_itrf14_2010_xgeoid20b =r"""+proj=pipeline
  +step +proj=axisswap +order=2,1
  +step +proj=unitconvert +xy_in=deg +xy_out=rad
  +step +proj=vgridshift +grids=us_noaa_g2018u0.tif +multiplier=1
  +step +proj=cart +ellps=GRS80
  +step +inv +proj=helmert +x=1.0053 +y=-1.9092 +z=-0.5416 +rx=0.0267814
        +ry=-0.0004203 +rz=0.0109321 +s=0.00037 +dx=0.0008 +dy=-0.0006
        +dz=-0.0014 +drx=6.67e-05 +dry=-0.0007574 +drz=-5.13e-05 +ds=-7e-05
        +t_epoch=2010 +convention=coordinate_frame
  +step +inv +proj=cart +ellps=GRS80
  +step +proj=vgridshift +grids=C:/Users/Felicio.Cassalho/Downloads/xGEOID20B.tif +multiplier=-1
  +step +proj=unitconvert +xy_in=rad +xy_out=deg
  +step +proj=axisswap +order=2,1"""

    t_nad832011_navd88geoid18_to_itrf14_2010_xgeoid20b = pyproj.Transformer.from_pipeline(nad832011_navd88geoid18_to_itrf14_2010_xgeoid20b).transform
    out = t_nad832011_navd88geoid18_to_itrf14_2010_xgeoid20b(lat,lon,z,t)

    return out

def grid_vdatum_update(hgrid,transform="navd88_to_xgeoid20b",epoch=2010):
    gd = read_hgrid(hgrid)
    x,y,z=gd.x,gd.y,gd.z
    if transform == "navd88_to_xgeoid20b":
       conv = navd88_to_xgeoid20b(y,x,z,[epoch for i in z])
       y_vd,x_vd,z_vd = conv[0],conv[1],conv[2]

    x_vd[np.isinf(x_vd)] = x[np.isinf(x_vd)]
    y_vd[np.isinf(y_vd)] = y[np.isinf(y_vd)]
    z_vd[np.isinf(z_vd)] = z[np.isinf(z_vd)]
    # gd.x=x_vd
    # gd.y=y_vd
    gd.dp=z_vd
    
    return gd

if __name__ == '__main__':
    wdir = r'C:\Users\Felicio.Cassalho\Work\Modeling\OCSMesh_Paper\STOFS_DEM'
    gd = grid_vdatum_update(f'{wdir}/hgrid.ll.dem_loaded.mpi.gr3')
    gd.write_hgrid(f'{wdir}/hgrid.ll.dem_loaded.mpi.XGEOID20B.gr3')
