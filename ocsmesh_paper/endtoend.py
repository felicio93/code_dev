#!/usr/bin/env python3
'''
End to end floodplain mesh gen.

Needs to prepare the following folders:
    path/
    |-- endtoend.py
    |-- inputs/
    |   |-- hgrid.gr3
    |   |-- ocean_mesh.2dm
    |   |-- rivers_v49.shp
    |   |-- dems/
    |       |-- gebco_file.tif  
    |-- outputs/

- path: defined on line 103

- hgrid.gr3: STOFS3D mesh, used for defining the floodplain domain.
    Please download it using: wget XX

- ocean_mesh.2dm: Ocean mesh, merged to the river+floodplina mesh at the end.
    Please download it using: wget XX

- rivers_v49.shp: RiverMapper polygons, used to create the river mesh.
    Please download it using: wget XX
    For details on how to create this file, please refer to: https://github.com/schism-dev/RiverMeshTools

- gebco_file.tif: should be placed inside dems/, should cover the entire model domain.
    Please download it from: https://download.gebco.net/
    Using the coordinates: -98.5 -51.0, 3, 53

- outputs/: where outputs (final mesh) will be saved.
'''


import os
import time
from copy import deepcopy
import numpy as np
import geopandas as gpd
from shapely.geometry import ( # type: ignore[import]
        Polygon, MultiPolygon, mapping)
import pandas as pd
import ocsmesh

def remove_interiors(poly):
    """
    Close polygon holes by limitation to the exterior ring.

    Arguments
    ---------
    poly: shapely.geometry.Polygon
        Input shapely Polygon

    Returns
    ---------
    Polygon without any interior holes
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly
def pop_largest(gs):
    """
    Pop the largest polygon off of a GeoSeries

    Arguments
    ---------
    gs: geopandas.GeoSeries
        Geoseries of Polygon or MultiPolygon objects

    Returns
    ---------
    Largest Polygon in a Geoseries
    """
    geoms = [g.area for g in gs]
    return geoms.pop(geoms.index(max(geoms)))
def close_holes(geom):
    """
    Remove holes in a polygon geometry

    Arguments
    ---------
    gseries: geopandas.GeoSeries
        Geoseries of Polygon or MultiPolygon objects

    Returns
    ---------
    Largest Polygon in a Geoseries
    """
    if isinstance(geom, MultiPolygon):
        ser = gpd.GeoSeries([remove_interiors(g) for g in geom])
        big = pop_largest(ser)
        outers = ser.loc[~ser.within(big)].tolist()
        if outers:
            return MultiPolygon([big] + outers)
        return Polygon(big)
    if isinstance(geom, Polygon):
        return remove_interiors(geom)



path = r"/work2/noaa/nosofs/felicioc/OCSMesh_Paper/step_01_MeshGen/"

print("Begin Deliniating the Floodplain Domain")
start_time = time.time()
##########################
### Floodplain Domain: ###
##########################
STOFS_mesh = ocsmesh.Mesh.open(path+"inputs/hgrid.gr3", crs=4326)
oc_mesh = ocsmesh.Mesh.open(path+"inputs/ocean_mesh.2dm", crs=4326)

poly_STOFS = ocsmesh.utils.get_mesh_polygons(STOFS_mesh.msh_t)
poly_oc = ocsmesh.utils.get_mesh_polygons(oc_mesh.msh_t)

fp_c = poly_STOFS.difference(poly_oc) #clipped floodplain
gdf = gpd.GeoDataFrame(geometry = gpd.GeoSeries(fp_c),crs=4326).dissolve().explode()
gdf.geometry=gdf.geometry.apply(lambda p: close_holes(p)) #closing all holes in the polygons
gdf= gdf[gdf.geometry.area >= 1e-3] #removing slivers based on area

#creating a buffer around the true FP and dissolving it so we have 1 continuos FP:
#gdf_0 = deepcopy(gdf)
#gdf_0 = gdf_0.dissolve()
#gdf_0['geometry'] = gdf_0.geometry.buffer(0.02)
#gdf_0 = gdf_0.dissolve().explode()
#gdf_0.geometry=gdf_0.geometry.apply(lambda p: close_holes(p))
#gdf_0.to_file(path+"outputs/gdf_0.shp")

gdf['geometry'] = gdf.geometry.buffer(0.01)
gdf = gdf.dissolve().explode()
gdf.geometry=gdf.geometry.apply(lambda p: close_holes(p))
gdf.to_file(path+"outputs/fp_domain.shp")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for Floodplain deliniation: {elapsed_time} seconds")



print("Begin River Mesh Gen")
start_time = time.time()
###################
### River Mesh: ###
###################
rm_poly = gpd.read_file(path+"inputs/rivers_v49.shp")
river_tr = ocsmesh.utils.triangulate_rivermapper_poly(rm_poly)
river_tr = ocsmesh.utils.clip_mesh_by_shape(river_tr, gdf.union_all(), adjacent_layers=8)
ocsmesh.Mesh(river_tr).write(path+"outputs/river_tr_v49.2dm", format='2dm', overwrite=True)
del rm_poly, river_tr

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for RiverMesh: {elapsed_time} seconds")


print("Begin Floodplain Mesh Gen")
start_time = time.time()
########################
### Floodplain Mesh: ###
########################
dem_paths = [f"{path}inputs/dems/{dem}" for dem in os.listdir(path+"inputs/dems/")]
geom_rast_list = [ocsmesh.Raster(f) for f in dem_paths if f[-4:] == '.tif']
hfun_rast_list = [ocsmesh.Raster(f) for f in dem_paths if f[-4:] == '.tif']

#Mesh gen:
geom = ocsmesh.Geom(
    geom_rast_list,
    base_shape=gdf.union_all(),
    base_shape_crs=gdf.crs,
    # zmax=10
    )
hfun = ocsmesh.Hfun(
    hfun_rast_list,
    base_shape=gdf.union_all(),
    base_shape_crs=geom.crs,
    hmin=500, hmax=10000,
    method='fast')
#hfun.add_constant_value(3000, lower_bound=-99999, upper_bound=-20)
hfun.add_constant_value(1200, lower_bound=-999990, upper_bound=-5)
hfun.add_constant_value(500, lower_bound=-5, upper_bound=99999)
#hfun.add_constant_value(1200, lower_bound=10, upper_bound=99999)
driver = ocsmesh.JigsawDriver(geom, hfun, crs=4326)
fp_mesh = driver.run()

hfun_mesh = ocsmesh.mesh.EuclideanMesh2D(hfun.msh_t())
ocsmesh.utils.reproject(mesh=hfun_mesh.msh_t,dst_crs=4326)
fp_mesh = ocsmesh.utils.fix_small_el(fp_mesh, hfun_mesh, u_limit = 1e-7)
ocsmesh.utils.cleanup_isolates(fp_mesh)
ocsmesh.utils.cleanup_duplicates(fp_mesh)
ocsmesh.utils.cleanup_isolates(fp_mesh)
ocsmesh.utils.put_id_tags(fp_mesh)

ocsmesh.Mesh(fp_mesh).write(path+"outputs/fp_mesh.2dm", format="2dm", overwrite=True)
del geom, hfun, geom_rast_list, hfun_rast_list, driver

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for FloodplainMesh: {elapsed_time} seconds")


print("Begin River+Floodplain Mesh Merge")
start_time = time.time()
###################
### Merge Mesh: ###
###################
###Merge River into the Floodplain
fp_mesh = ocsmesh.Mesh.open(path+"outputs/fp_mesh.2dm", crs=4326)
river_mesh = ocsmesh.Mesh.open(path+"outputs/river_tr_v49.2dm", crs=4326)

fp_r = ocsmesh.utils.merge_overlapping_meshes([fp_mesh.msh_t,river_mesh.msh_t])
ocsmesh.Mesh(fp_r).write(path+"outputs/fp_r.2dm", format='2dm', overwrite=True)

del fp_mesh, river_mesh
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for River+Floodplain Mesh Merge: {elapsed_time} seconds")


print("Begin River+Floodplain Mesh Cleanup")
start_time = time.time()
###Remove islands in the floodplain+river (cleanup for source_sink placement)
fp_r_poly = ocsmesh.utils.get_mesh_polygons(fp_r)

fp_r_gdf_noholes = gpd.GeoDataFrame(geometry = gpd.GeoSeries(fp_r_poly),crs=4326).dissolve().explode()
fp_r_gdf_noholes.geometry = fp_r_gdf_noholes.geometry.apply(lambda p: close_holes(p)) #closing all holes in the polygons
fp_r_noholes_poly = fp_r_gdf_noholes.union_all()

fp_islands = fp_r_noholes_poly.difference(fp_r_poly)
fp_islands = gpd.GeoDataFrame(geometry = gpd.GeoSeries(fp_islands),crs=4326).dissolve().explode()
area_threshold = 1.0e-15 #to remove slivers
fp_islands['area'] = fp_islands.geometry.area
fp_islands = fp_islands[fp_islands['area'] >= area_threshold]
#fp_islands = fp_islands.reset_index(drop=True)

msht_island = ocsmesh.utils.triangulate_polygon_s(fp_islands)
fp_r = ocsmesh.utils.merge_neighboring_meshes(fp_r,msht_island)

fp_r = ocsmesh.utils.cleanup_folded_bound_el(fp_r)
fp_r_poly = ocsmesh.utils.get_mesh_polygons(fp_r)
fp_r_gdf = gpd.GeoDataFrame(geometry = gpd.GeoSeries(fp_r_poly),crs=4326).dissolve().explode()
fp_r_gdf = fp_r_gdf.reset_index(drop=True)

fp_r_gdf['area'] = fp_r_gdf.geometry.area
largest_poly = fp_r_gdf.loc[fp_r_gdf['area'].idxmax()]

fp_r = ocsmesh.utils.clip_mesh_by_shape(fp_r, largest_poly.geometry, adjacent_layers=12)

ocsmesh.utils.cleanup_duplicates(fp_r)
ocsmesh.utils.cleanup_isolates(fp_r)
ocsmesh.utils.put_id_tags(fp_r)

ocsmesh.Mesh(fp_r).write(path+"outputs/fp_r_islands.2dm", format='2dm', overwrite=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for River+Floodplain Mesh Clean up: {elapsed_time} seconds")


print("Begin RiverFloodplain+Ocean Mesh Merge")
start_time = time.time()
###Merge Floodplain+River with the Ocean
ocean_mesh = ocsmesh.Mesh.open(path+"inputs/ocean_mesh.2dm", crs=4326)
fp_r_o = ocsmesh.utils.merge_overlapping_meshes([fp_r, ocean_mesh.msh_t])
ocsmesh.Mesh(fp_r_o).write(path+"outputs/fp_r_o.2dm", format='2dm', overwrite=True)
ocsmesh.Mesh(fp_r_o).write(path+"outputs/hgrid.ll", format='grd', overwrite=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for RiverFloodplain+Ocean Mesh Merge: {elapsed_time} seconds")


