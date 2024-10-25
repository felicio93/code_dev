import fsspec
import xarray as xr
import numpy as np
import pandas as pd

nodes = [2855435,2841202,2834742]
fs = fsspec.filesystem("s3", anon=True)

day,avg=[],[]
time,data=[],[]
for n in range(1,365+1):
    print(n)
    fs = fsspec.filesystem("s3", anon=True)
    
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
    
    dd = np.array(ds['temperature'][:,[nodes[0]-1,nodes[1]-1,nodes[-1]-1],-1])
    dt = np.array(ds['time'])
    data.append(dd)
    time.append(dt)

    day.append(np.array(str(dt[0]).split('T')[0]))
    avg.append(dd.mean(axis=0))


all_data = np.vstack(data)
all_time = np.vstack(time).ravel()
df_all = pd.DataFrame(data=all_data,    # values
             index=all_time,    # 1st column as index
             columns=nodes
            )
df_all.to_csv(r'/contrib/Felicio.Cassalho/work/schism_pac/timeseries/temp_hour_timeseries.csv', index=True) 

all_avg = np.vstack(avg)
all_day = np.vstack(day)
df_avg = pd.DataFrame(data=all_avg,    # values
             index=[d[0] for d in all_day],    # 1st column as index
             columns=nodes
            )
df_avg.to_csv(r'/contrib/Felicio.Cassalho/work/schism_pac/timeseries/temo_davg_timeseries.csv', index=True)
