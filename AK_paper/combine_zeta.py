import netCDF4 as nc
import xarray as xr
import numpy as np

### Change accordingly: ###
rundir = r"/work2/noaa/nosofs/felicioc/BeringSea/R07"
first_file,last_file = 1,123
###########################

z,t=[],[]
for i in range (first_file,last_file):
    print(i)
    ds = xr.open_dataset(f'{rundir}/outputs/out2d_{i}.nc',
                            engine='netcdf4',
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
    z.append(np.array(ds.elevation[:,:]))
    t.append(np.array(ds.time))

z = np.array(z)
t = np.array(t)

z,t = z.reshape(-1, z.shape[-1]), t.ravel()

dataset = nc.Dataset('elev_combine.nc', 'w', format='NETCDF4')

time_dim = dataset.createDimension('t', len(t)) 
data_dim = dataset.createDimension('z', z.shape[1])
time_var = dataset.createVariable('t', 'f8', ('t',))
data_var = dataset.createVariable('z', 'f4', ('t', 'z'))
time_var[:] = t
data_var[:] = z
time_var.units = 'datetime'
data_var.units = 'meters'

dataset.close()
print("elev_combine.nc created!")

