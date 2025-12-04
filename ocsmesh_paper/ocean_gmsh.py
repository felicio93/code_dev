import gmsh
import geopandas as gpd
import numpy as np
import sys
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt, binary_dilation

def get_mesh_quality_stats():
    try:
        elementTags, nodeTags = gmsh.model.mesh.getElementsByType(2)
    except:
        return 0.0, 0.0
    if len(elementTags) == 0: return 0.0, 0.0
    qualities = gmsh.model.mesh.getElementQualities(elementTags, "minSICN")
    return np.min(qualities), np.mean(qualities)

def calculate_sizes(depths_1d, width, height, rules_list):
    sizes = np.full_like(depths_1d, fill_value=np.nan, dtype=float)
    Z_2d = depths_1d.reshape((height, width))
    sorted_rules = sorted(rules_list, key=lambda x: x.get('priority', 0))
    
    print("\n--- RULE EXECUTION REPORT ---")
    
    for i, rule in enumerate(sorted_rules):
        z_min = rule.get('z_min', -np.inf)
        z_max = rule.get('z_max', np.inf)
        prio = rule.get('priority', 0)
        mode = rule.get('mode', 'depth') 
        k = rule.get('exponent', 1.0)
        
        mask_1d = (depths_1d >= z_min) & (depths_1d < z_max)
        count = np.sum(mask_1d)
        
        if count == 0:
            continue

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
                mask_2d = mask_1d.reshape((height, width))
                mask_deep_side = Z_2d < z_min
                mask_shallow_side = Z_2d >= z_max
                
                if not np.any(mask_deep_side) and not np.any(mask_shallow_side):
                    ratio = 0.5 
                else:
                    if np.any(mask_deep_side):
                        d_deep = distance_transform_edt(~mask_deep_side)
                    else:
                        d_deep = np.inf 
                    if np.any(mask_shallow_side):
                        d_shallow = distance_transform_edt(~mask_shallow_side)
                    else:
                        d_shallow = np.inf

                    d_d_vals = d_deep[mask_2d]
                    d_s_vals = d_shallow[mask_2d]
                    
                    total_dist = d_d_vals + d_s_vals
                    total_dist[total_dist == 0] = 1.0
                    ratio = d_d_vals / total_dist
                desc = f"Dist Ramp (Exp {k})"

            if k != 1.0:
                ratio = np.power(ratio, k)
            current_sizes = res_min + ratio * (res_max - res_min)

        sizes[mask_1d] = current_sizes
        print(f"Rule {i+1} (Prio {prio}): {desc} | [{z_min}, {z_max}] | {count} pixels | Mode: {mode}")

    mask_nan = np.isnan(sizes)
    if np.any(mask_nan):
        print(f"Warning: {np.sum(mask_nan)} pixels not covered. Filling with 0.05")
        sizes[mask_nan] = 0.05
        
    print("-----------------------------\n")
    return sizes

def extract_islands_as_geometries(gdf):
    island_lines = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = geom.geoms
        else:
            continue
            
        for p in polys:
            for interior in p.interiors:
                island_lines.append(interior)
    return island_lines

def create_background_field_from_dem(dem_path, rules, gdf=None, island_res=None, subsample=5):
    print(f"Reading DEM: {dem_path}...")
    try:
        with rasterio.open(dem_path) as src:
            h = src.height // subsample
            w = src.width // subsample
            data = src.read(1, out_shape=(h, w))
            
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
            
            print(f"DEM Stats: Min Z = {zs_filled.min():.2f}, Max Z = {zs_filled.max():.2f}")
            
            target_sizes = calculate_sizes(zs_filled, w, h, rules)
            
            if gdf is not None and island_res is not None:
                print(f"Burning in Island Resolution ({island_res}) into Background Field...")
                island_lines = extract_islands_as_geometries(gdf)
                
                if island_lines:
                    island_mask = features.rasterize(
                        island_lines,
                        out_shape=(h, w),
                        transform=transform,
                        default_value=1,
                        all_touched=True
                    ).astype(bool)
                    
                    dilated_mask = binary_dilation(island_mask, iterations=2)
                    target_sizes_2d = target_sizes.reshape((h, w))
                    
                    count_before = np.sum(target_sizes_2d[dilated_mask] != island_res)
                    target_sizes_2d[dilated_mask] = island_res
                    target_sizes = target_sizes_2d.flatten()
                    print(f"Modified {count_before} pixels near islands to force resolution {island_res}.")
            
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

def generate_mesh_from_shapefile(shapefile_path, dem_path, output_msh, mesh_rules, island_res=None, simplification_tol=None):
    print(f"Reading {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path)
        
        print("Validating and repairing input geometry...")
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty]
        
        shp_bounds = gdf.total_bounds 
        print(f"Shapefile Bounds: {shp_bounds}")
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return

    gmsh.initialize()
    gmsh.model.add("OceanMesh")

    view_tag, dem_bounds = create_background_field_from_dem(dem_path, mesh_rules, 
                                                            gdf=gdf, island_res=island_res, 
                                                            subsample=5)
    
    if view_tag is not None:
        field_tag = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        gmsh.view.write(view_tag, "background_field.pos")
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-6)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e+22)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    else:
        print("Warning: DEM load failed.")
        fallback = mesh_rules[0].get('res', 0.05)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", fallback)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", fallback)

    if simplification_tol is not None:
        print(f"Simplifying Main Geometry (Tolerance: {simplification_tol})...")
        gdf['geometry'] = gdf.geometry.simplify(tolerance=simplification_tol, preserve_topology=True)
    else:
        print("Using exact polygon vertices.")

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

    print("Constructing geometry (OCC Kernel)...")
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
            except Exception as e: pass

    print("Cleaning topology...")
    
    merge_tol = 1e-7 
    print(f"Setting Geometry Tolerance to {merge_tol} to fix gaps...")
    gmsh.option.setNumber("Geometry.Tolerance", merge_tol)
    
    object_dim_tags = [(2, s) for s in surface_tags]
    out_dim_tags, _ = gmsh.model.occ.fragment(object_dim_tags, [])
    gmsh.model.occ.synchronize()

    # --- FIX: STRICTLY RESPECT VERTICES (NO SPLITTING) ---
    # Iterate over ALL curves in the model (coastlines & islands)
    # Set Transfinite = 2. This forces the curve to be exactly 1 element edge.
    # This overrides the background field on the boundary, preventing over-refinement.
    print("Enforcing strict vertex respect on boundaries (Transfinite=2)...")
    all_curves = gmsh.model.getEntities(1)
    for c in all_curves:
        gmsh.model.mesh.setTransfiniteCurve(c[1], 2)

    valid_surfaces = [tag for dim, tag in out_dim_tags if dim == 2]
    
    if valid_surfaces:
        ps = gmsh.model.addPhysicalGroup(2, valid_surfaces)
        gmsh.model.setPhysicalName(2, ps, "Ocean_Domain")
        
        boundary_curves_tags = [tag for dim, tag in all_curves]
        pc = gmsh.model.addPhysicalGroup(1, boundary_curves_tags)
        gmsh.model.setPhysicalName(1, pc, "Coastline")
    else:
        print("Error: No valid surfaces found after fragmentation.")
        gmsh.finalize()
        return

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    print("Generating mesh...")
    gmsh.model.mesh.generate(2)
    
    print("\nRunning Optimizer (Netgen)...")
    gmsh.model.mesh.optimize("Netgen")

    print(f"\nSaving to {output_msh}...")
    gmsh.write(output_msh)
    gmsh.finalize()
    print("Done.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_shp = r"C:\Users\Felicio.Cassalho\Work\Python_Development\Mesh\AutoOceanMesh\Meshing\inputs\stofs3.shp"
    input_dem = r"C:\Users\Felicio.Cassalho\Work\Python_Development\Mesh\AutoOceanMesh\RegimeIdentification\inputs\gebco_2024_n56.0_s5.0_w-100.0_e-50.0.tif"
    output_file = r"./outputs/ocean_stofs.msh"

    mesh_rules = [
        {'priority': 1, 'z_min': -100000.0, 'z_max': 100000.0, 'res': 0.01, 'mode': 'depth'},
        {'priority': 10, 'z_min': -50.0, 'z_max': 0.0, 'res_min': 0.045, 'res_max': 0.01, 'mode': 'distance', 'exponent': 1.0},
        {'priority': 15, 'z_min': -400.0, 'z_max': -50.0, 'res_min': 0.07, 'res_max': 0.045, 'mode': 'depth', 'exponent': 1.0},
        {'priority': 20, 'z_max': -400.0, 'res': 0.07, 'mode': 'depth'},
    ]
    
    # Ensure island_res and simplification_tol are None to respect input vertices exactly
    generate_mesh_from_shapefile(input_shp, input_dem, output_file, mesh_rules, 
                                 island_res=None, 
                                 simplification_tol=None)
    
