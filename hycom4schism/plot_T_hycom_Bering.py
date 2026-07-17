import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import imageio.v2 as imageio

# 1. Open the dataset lazily
file_path = 'FINAL_TS_20180701_20181130.nc'
ds = xr.open_dataset(file_path)

# 2. Select roughly one day per month (every 30th time index)
ds_subset = ds.isel(time=slice(0, None, 30))

# 3. Define the Bering Sea bounds to subset spatially
ds_bering = ds_subset.sel(xlon=slice(150, 225), ylat=slice(45, 78))

# Set consistent color scale limits tailored for the colder Bering Sea
vmin_temp, vmax_temp = -2, 16

# Create an empty list to store the filenames for our GIF
image_filenames = []

# Loop through our monthly subset
for i in range(len(ds_bering.time)):
    current_time = ds_bering.time[i].values
    # Format time to a readable string (YYYY-MM-DD)
    date_str = str(current_time)[:10]
    print(f"Processing and plotting {date_str}...")

    # Extract the 3D temperature block for this single timestep
    temp_3d = ds_bering.temperature.isel(time=i).compute()

    # --- Extract Surface Layer ---
    temp_surface = temp_3d.isel(depth=0)

    # --- Extract Bottom Layer ---
    temp_bottom = temp_3d.ffill(dim='depth').isel(depth=-1)

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 6))

    # Using central_longitude=180 prevents the map from tearing at the antimeridian
    proj = ccrs.PlateCarree(central_longitude=180)
    data_proj = ccrs.PlateCarree()

    # Surface Plot
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    # Set the spatial extent bounds corresponding to the 150 to 225 box
    ax1.set_extent([150, 225, 45, 78], crs=data_proj)
    ax1.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='lightgray')
    ax1.add_feature(cfeature.COASTLINE, zorder=101)

    im1 = ax1.pcolormesh(temp_surface.xlon, temp_surface.ylat, temp_surface,
                         transform=data_proj, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    ax1.set_title(f'Surface Temperature ({date_str})', fontsize=14)

    # Bottom Plot
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.set_extent([150, 225, 45, 78], crs=data_proj)
    ax2.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='lightgray')
    ax2.add_feature(cfeature.COASTLINE, zorder=101)

    im2 = ax2.pcolormesh(temp_bottom.xlon, temp_bottom.ylat, temp_bottom,
                         transform=data_proj, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    ax2.set_title(f'Bottom Temperature ({date_str})', fontsize=14)

    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Potential Temperature (degC)')

    plt.subplots_adjust(right=0.9)

    # Save the figure
    filename = f"HYCOM_Temp_Surf_Bot_{date_str}.jpeg"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # Add the filename to our list
    image_filenames.append(filename)
    print(f"Saved {filename}")

print("Daily plotting complete. Compiling GIF...")

# --- GIF CREATION ---
gif_filename = "HYCOM_Temperature_Animation.gif"

images = []
for filename in image_filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(gif_filename, images, duration=0.2, loop=0)

print(f"GIF successfully saved as: {gif_filename}")
print("Processing complete.")
