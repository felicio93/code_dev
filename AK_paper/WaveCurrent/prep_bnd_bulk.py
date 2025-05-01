import os
import xarray as xr
import numpy as np
#import ocsmesh

def convert_longitude(lon,mode):
    """
    Convert longitudes between common geographic conventions.

    Args:
        lon: array-like of longitudes
        mode: conversion mode:
            - 1: Convert from [-180, 180] to [0, 360] (Greenwich at 0°)
            - 2: Convert from [0, 360] to [-180, 180] (Greenwich at 0°)
            - 3: Convert from [-180, 180] to [0, 360] (Greenwich at 180°)

    Returns:
        np.ndarray of converted longitudes
    """

    lon = np.asarray(lon)
    if mode == 1:
        return lon % 360
    elif mode == 2:
        return np.where(lon > 180, lon - 360, lon)
    elif mode == 3:
        return lon + 180
    return lon

def crop_by_box(dataset: xr.Dataset,
              lat_min: float,
              lat_max: float,
              lon_min: float,
              lon_max: float) -> xr.Dataset:
    """
    Crops xarray data based on lats and lons

    Args:
        lat_min: float/int of mininum latitude
        lat_max: float/int of maximum latitude
        lon_min: float/int of mininum longitude
        lon_max: float/int of maximum latitude
    Returns:
        xarray object of the cropped data
    Note:
        Satellite data uses the -180 to 180 standard
        If you want to cross the meridian, then pass a lon_min > lon_max

    """
    # Check if latitude and longitude coordinates are in the dataset
    if 'latitude' not in dataset or 'longitude' not in dataset:
        raise ValueError("Dataset does not contain lat or lon dimensions")

    if lon_min < lon_max:
        lon_mask = (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max)
    else:
        lon_mask = (dataset.longitude >= lon_min) | (dataset.longitude <= lon_max)

    lat_mask = (dataset.latitude >= lat_min) & (dataset.latitude <= lat_max)
    cropped = dataset.where(lat_mask & lon_mask, drop=True)

    return cropped


def find_and_sort_files(directory, search_string):
    """
    Finds all files in a directory containing a specific string in their names and returns a sorted list of their full paths.

    Args:
        directory (str): The path to the directory to search in.
        search_string (str): The string to search for in the file names.

    Returns:
        list: A sorted list of full paths to the files containing the search string.
    """
    matching_files = []
    for filename in os.listdir(directory):
        if search_string in filename:
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                matching_files.append(full_path)
    return sorted(matching_files)


def main(path,xmin,xmax,ymin,ymax,variables):


    for var in variables:
        print(f"Preparing {var} data...")
        files = find_and_sort_files(path, var)
        dd=[]
        for ff in files:
            print(f"Processing file {ff}")
            ds = xr.open_dataset(ff)
            ds = ds.assign_coords({'longitude': convert_longitude(ds.longitude,1)})
            ds = ds.sortby('longitude')
            ds = crop_by_box(ds,ymin,ymax,xmin,xmax)
            dd.append(ds)
        print(f"...Concat and save final {var} data")
        dataset = xr.concat(dd,dim='time')

        #fix the encoding problem:
        encoding = {vv: {"_FillValue":None} for vv in ["longitude", "latitude", "time","MAPSTA", var]}
        dataset.to_netcdf(f"{path}/{var}.nc", encoding=encoding)



if __name__ == "__main__":
    
    path = r"./"
    #mesh = ocsmesh.Mesh.open(path + 'hgrid.gr3', crs=4326)
    #xmin,xmax = mesh.vert2['coord'][:, 0].min()-0.5,mesh.vert2['coord'][:, 0].max()+0.5
    #ymin,ymax = mesh.vert2['coord'][:, 1].min()-0.5,mesh.vert2['coord'][:, 1].max()+0.5
    variables = ["fp", "dir", "hs", "t02", "spr"]
    ymin,ymax=48.,69.
    xmin,xmax=154.,206.

    main(path,xmin,xmax,ymin,ymax,variables)

