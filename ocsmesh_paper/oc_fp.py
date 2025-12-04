import gmsh
import geopandas as gpd
import numpy as np
import sys
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt, binary_dilation
import pandas as pd

def get_mesh_quality_stats():
    try:
        elementTags, nodeTags = gmsh.model.mesh.getElementsByType(2)
    except:
        return 0.0, 0.0
    if len(elementTags) == 0: return 0.0, 0.0
    qualities = gmsh.model.mesh.getElementQualities(elementTags, "minSICN")
    return np.min(qualities), np.mean(qualities)

def calculate_sizes(depths_1d, width, height, transform, rules_list, aux_geoms=None):
    """
    Calculates mesh sizes based on depth AND geometry proximity rules.
    """
    sizes = np.full_like(depths_1d, fill_value=np.nan, dtype=float)
    Z_2d = depths_1d.reshape((height, width))
    
    # Sort rules by priority (Low number runs first, High number overwrites)
    sorted_rules = sorted(rules_list, key=lambda x: x.get('priority', 0))
    
    print("\n--- RULE EXECUTION REPORT ---")
    
    for i, rule in enumerate(sorted_rules):
        prio = rule.get('priority', 0)
        mode = rule.get('mode', 'depth')
        desc = ""
        count = 0
        
        # --- TYPE A: DEPTH / DISTANCE RULES (Existing Logic) ---
        if mode in ['depth', 'distance']:
            z_min = rule.get('z_min', -np.inf)
            z_max = rule.get('z_max', np.inf)
            k = rule.get('exponent', 1.0)
            
            mask_1d = (depths_1d >= z_min) & (depths_1d < z_max)
            count = np.sum(mask_1d)
            
            if count == 0: continue

            current_sizes = np.zeros(count, dtype=float)
            
            if 'res' in rule:
                current_sizes[:] = rule['res']
                desc = f"Constant {rule['res']}"
                
            elif 'res_min' in rule and 'res_max' in rule:
                res_min = rule['res_min']
                res_max = rule['res_max']
                
                if mode == 'depth':
                    vals = depths_1d[mask_1d]
                    z_clamped = np.clip(vals, z_min, z_max)
                    if z_max != z_min:
                        ratio = (z_clamped - z_min) / (z_max - z_min)
                    else:
                        ratio = np.zeros_like(vals)
                    desc = f"Depth Ramp (Exp {k})"

                elif mode == 'distance':
                    # Distance logic (unchanged)
                    mask_2d = mask_1d.reshape((height, width))
                    mask_deep_side = Z_2d < z_min
                    mask_shallow_side = Z_2d >= z_max
                    
                    if not np.any(mask_deep_side) and not np.any(mask_shallow_side):
                        ratio = 0.5 
                    else:
                        if np.any(mask_deep_side): d_deep = distance_transform_edt(~mask_deep_side)
                        else: d_deep = np.inf 
                        if np.any(mask_shallow_side): d_shallow = distance_transform_edt(~mask_shallow_side)
                        else: d_shallow = np.inf

                        d_d_vals = d_deep[mask_2d]
                        d_s_vals = d_shallow[mask_2d]
                        total_dist = d_d_vals + d_s_vals
                        total_dist[total_dist == 0] = 1.0
                        ratio = d_d_vals / total_dist
                    desc = f"Dist Ramp (Exp {k})"

                if k != 1.0: ratio = np.power(ratio, k)
                current_sizes = res_min + ratio * (res_max - res_min)

            sizes[mask_1d] = current_sizes
            print(f"Rule {i+1} (Prio {prio}): {desc} | [{z_min}, {z_max}] | {count} pixels updated")

        # --- TYPE B: PROXIMITY RULES (New River Logic) ---
        elif mode == 'proximity':
            source_key = rule.get('source')
            if aux_geoms is None or source_key not in aux_geoms:
                print(f"Rule {i+1}: Skipped (Source '{source_key}' not found)")
                continue
                
            gdf_source = aux_geoms[source_key]
            res_target = rule.get('res')
            res_max = rule.get('res_max')
            ramp_width = rule.get('ramp_width', 0.1) # degrees
            
            # 1. Rasterize the source geometry
            mask_source = features.rasterize(
                gdf_source.geometry,
                out_shape=(height, width),
                transform=transform,
                default_value=1,
                all_touched=True
            ).astype(bool)
            
            # 2. Calculate Distance Field (inverted mask)
            dist_pixels = distance_transform_edt(~mask_source)
            dist_degrees = dist_pixels * transform[0] # Approx pixel width
            
            # 3. Calculate Size Ramp
            # size = target + (dist / ramp) * (max - target)
            calculated_field = res_target + (dist_degrees / ramp_width) * (res_max - res_target)
            
            # 4. Apply Logic
            # We apply this rule wherever the calculated ramp is less than the 'res_max'
            # effectively creating a "buffer zone" of influence.
            # Inside the river, dist is 0, so size is res_target.
            mask_apply = calculated_field <= res_max
            
            # Flatten for assignment
            mask_apply_1d = mask_apply.flatten()
            sizes[mask_apply_1d] = calculated_field.flatten()[mask_apply_1d]
            
            desc = f"Proximity Ramp ({source_key})"
            count = np.sum(mask_apply_1d)
            print(f"Rule {i+1} (Prio {prio}): {desc} | {res_target}->{res_max} over {ramp_width} deg | {count} pixels updated")

    mask_nan = np.isnan(sizes)
    if np.any(mask_nan):
        print(f"Warning: {np.sum(mask_nan)} pixels not covered. Filling with 0.05")
        sizes[mask_nan] = 0.05
        
    print("-----------------------------\n")
    return sizes

def create_background_field_from_dem(dem_path, rules, aux_geoms=None, subsample=1):
    print(f"Reading DEM: {dem_path}...")
    try:
        with rasterio.open(dem_path) as src:
            h = src.height // subsample
            w = src.width // subsample
            data = src.read(1, out_shape=(h, w))
            
            # Transform adjustment
            transform = src.transform * src.transform.scale(subsample, subsample)
            
            cols, rows = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            
            xs = np.array(xs).flatten()
            ys = np.array(ys).flatten()
            zs = data.flatten()
            
            valid_mask = zs != src.nodata
            if src.nodata is None: valid_mask = zs > -30000 
            
            zs_filled = zs.astype(float)
            zs_filled[~valid_mask] = -99999.0 
            
            # --- MAIN CALCULATION ---
            # Now passing transform and aux_geoms
            target_sizes = calculate_sizes(zs_filled, w, h, transform, rules, aux_geoms)
            
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            target_sizes = target_sizes[valid_mask]

            if len(xs) == 0: return None, None

            view_tag = gmsh.view.add("Background Bathymetry")
            combined_data = np.column_stack((xs, ys, np.zeros_like(xs), target_sizes))
            list_data = combined_data.flatten().tolist()
            gmsh.view.addListData(view_tag, "SP", len(xs), list_data)
            
            dem_bounds = (np.min(xs), np.min(ys), np.max(xs), np.max(ys))
            return view_tag, dem_bounds

    except Exception as e:
        print(f"Error processing DEM: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_mesh_from_shapefile(shapefile_path, dem_path, output_msh, mesh_rules, 
                                 river_shp_path=None, island_res=None, simplification_tol=None):
    
    # 1. Load Ocean
    print(f"Reading Ocean: {shapefile_path}...")
    gdf = gpd.read_file(shapefile_path)
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty]

    # 2. Load Aux Geometries (Rivers)
    aux_geoms = {}
    if river_shp_path:
        print(f"Reading Rivers: {river_shp_path}")
        try:
            gdf_river = gpd.read_file(river_shp_path)
            # Clip for speed
            minx, miny, maxx, maxy = gdf.total_bounds
            gdf_river = gdf_river.cx[minx:maxx, miny:maxy]
            aux_geoms['rivers'] = gdf_river
        except Exception as e:
            print(f"Warning: Could not load rivers: {e}")
    
    # Also add Islands to aux_geoms if you want to use them in a rule too!
    # aux_geoms['islands'] = gdf 

    gmsh.initialize()
    gmsh.model.add("OceanMesh")

    # 3. Create Field
    view_tag, dem_bounds = create_background_field_from_dem(
        dem_path, mesh_rules, aux_geoms=aux_geoms, subsample=1
    )
    
    if view_tag is not None:
        field_tag = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        gmsh.view.write(view_tag, "background_field.pos")
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-6)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e+22)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    else:
        fallback = 0.05
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", fallback)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", fallback)

    if simplification_tol is not None:
        gdf['geometry'] = gdf.geometry.simplify(tolerance=simplification_tol, preserve_topology=True)

    def add_loop(coords, explicit_size=0.0):
        point_tags = []
        for x, y in coords:
            p = gmsh.model.occ.addPoint(x, y, 0, explicit_size)
            point_tags.append(p)
        line_tags = []
        for i in range(len(point_tags)):
            p1 = point_tags[i]
            p2 = point_tags[(i + 1) % len(point_tags)]
            l = gmsh.model.occ.addLine(p1, p2)
            line_tags.append(l)
        return gmsh.model.occ.addCurveLoop(line_tags)

    print("Constructing geometry...")
    surface_tags = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon': polygons = [geom]
        elif geom.geom_type == 'MultiPolygon': polygons = geom.geoms
        else: continue

        for poly in polygons:
            ext_coords = list(poly.exterior.coords)[:-1] 
            if len(ext_coords) < 3: continue 
            try:
                ext_loop = add_loop(ext_coords, 0.0) 
                hole_loops = []
                for interior in poly.interiors:
                    if island_res is not None:
                        interior = interior.simplify(tolerance=island_res, preserve_topology=True)
                    int_coords = list(interior.coords)[:-1]
                    if len(int_coords) >= 3: 
                        hole_loops.append(add_loop(int_coords, 0.0))
                s = gmsh.model.occ.addPlaneSurface([ext_loop] + hole_loops)
                surface_tags.append(s)
            except: pass

    print("Cleaning topology...")
    gmsh.option.setNumber("Geometry.Tolerance", 1e-7)
    object_dim_tags = [(2, s) for s in surface_tags]
    out_dim_tags, _ = gmsh.model.occ.fragment(object_dim_tags, [])
    gmsh.model.occ.synchronize()

    print("Enforcing boundary vertices...")
    all_curves = gmsh.model.getEntities(1)
    for c in all_curves:
        gmsh.model.mesh.setTransfiniteCurve(c[1], 2)

    valid_surfaces = [tag for dim, tag in out_dim_tags if dim == 2]
    if valid_surfaces:
        ps = gmsh.model.addPhysicalGroup(2, valid_surfaces)
        gmsh.model.setPhysicalName(2, ps, "Ocean_Domain")
        pc = gmsh.model.addPhysicalGroup(1, [tag for dim, tag in all_curves])
        gmsh.model.setPhysicalName(1, pc, "Coastline")
    else:
        return

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    print("Generating mesh...")
    gmsh.model.mesh.generate(2)
    gmsh.write(output_msh)
    gmsh.finalize()
    print("Done.")

if __name__ == "__main__":
    input_shp = r"C:\Users\Felicio.Cassalho\Work\Python_Development\Mesh\AutoOceanMesh\Meshing\inputs\stofs3.shp"
    input_river_shp = r"../inputs/clipped_river.shp"
    input_dem = r"C:\Users\Felicio.Cassalho\Work\Python_Development\Mesh\AutoOceanMesh\RegimeIdentification\inputs\gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif"
    output_file = r"./ocean_stofs_integrated_river.msh"

    # --- UPDATED MESH RULES ---
    mesh_rules = [
        # 1. Global Background
        {'priority': 1, 'z_min': -100000.0, 'z_max': 100000.0, 'res': 0.01, 'mode': 'depth'},
        
        # 2. Shallow Water
        {'priority': 10, 'z_min': -50.0, 'z_max': 0.0, 'res_min': 0.045, 'res_max': 0.01, 'mode': 'distance', 'exponent': 1.0},
        
        # 3. RIVER RULE (Priority 12)
        # Higher priority than Global(1) and Shallow(10), so it overwrites them.
        # It fades from 0.005 (at river) to 0.05 (at 0.1 deg distance).
        {'priority': 12, 'mode': 'proximity', 'source': 'rivers', 'res': 0.005, 'res_max': 0.01, 'ramp_width': 0.05},
        
        # 4. Deep Water (Priority 15 & 20)
        # These are strictly higher priority. If the river rule put a 0.005 pixel 
        # in the middle of a -400m trench, this rule will overwrite it with 0.07.
        {'priority': 15, 'z_min': -400.0, 'z_max': -50.0, 'res_min': 0.07, 'res_max': 0.045, 'mode': 'depth', 'exponent': 1.0},
        {'priority': 20, 'z_max': -400.0, 'res': 0.07, 'mode': 'depth'},
    ]
    
    generate_mesh_from_shapefile(input_shp,
                                 input_dem,
                                 output_file,
                                 mesh_rules, 
                                 river_shp_path=input_river_shp, 
                                 island_res=None)