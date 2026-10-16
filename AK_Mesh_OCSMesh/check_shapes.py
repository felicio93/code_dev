#!/usr/bin/env python
"""Validate the Alaska shapefiles BEFORE spending hours in OCSMesh.

    python check_shapes.py [DATA_DIR] [--sizes 1500 2500 3500]

Per shapefile it reports geometry types, validity (raw, healed, and after
projection to EPSG:3338), holes/parts, and tiny slivers.

For the domain it then rehearses the step that crashed the first job: it
resamples the boundary with OCSMesh's own ``resample_geom_by_hfun`` using
synthetic size fields (constant hmin, constant hmax, and a gradient), then
applies ``make_valid`` + ``union_all`` exactly like ``GmshEngine.generate`` and
reports whether the result is a clean Polygon/MultiPolygon or a
GeometryCollection. Takes a few minutes, not hours.

Exit code is non-zero if a hard problem is found.
"""
import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity, make_valid

ap = argparse.ArgumentParser()
ap.add_argument("data_dir", nargs="?", type=Path, default=Path(
    "/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI/"
    "AK_LargeDomain_Mesh/2026_10_15"))
ap.add_argument("--sizes", type=float, nargs="+", default=[1500.0, 2500.0, 3500.0],
                help="constant sizing values (m) used in the resampling rehearsal")
ap.add_argument("--skip-resample", action="store_true")
args = ap.parse_args()

SHAPES = {
    "domain": "AKdomain_20261014_360.shp",
    "shelf_150": "150m_contour_360.shp",
    "shelf_200": "200m_contour_360.shp",
    "bering": "beringsea_360.shp",
    "center": "coarse_ocean_center_360.shp",
    "south": "coarse_ocean_south_360.shp",
    "north": "coarse_ocean_north_360.shp",
}
METRIC = CRS.from_epsg(3338)
SLIVER_M2 = 1.0e4          # polygons smaller than 0.01 km^2 are flagged

problems, warnings_ = 0, 0


def fail(msg):
    global problems
    problems += 1
    print(f"   [FAIL] {msg}")


def warn(msg):
    global warnings_
    warnings_ += 1
    print(f"   [WARN] {msg}")


def ok(msg):
    print(f"   [ OK ] {msg}")


def polys_of(g):
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        return [p for s in g.geoms for p in polys_of(s)]
    return []


def describe(g):
    ps = polys_of(g)
    holes = sum(len(p.interiors) for p in ps)
    npts = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors)
               for p in ps)
    return f"{g.geom_type}: {len(ps)} polygon(s), {holes} hole(s), {npts:,} vertices"


def n_nonpoly(g):
    if hasattr(g, "geoms") and not isinstance(g, MultiPolygon):
        return sum(not isinstance(s, (Polygon, MultiPolygon)) for s in g.geoms)
    return 0


healed, healed_m = {}, {}
for name, fn in SHAPES.items():
    path = args.data_dir / fn
    print(f"\n== {name}  ({fn})")
    if not path.is_file():
        fail("file missing")
        continue
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        warn("no CRS in file; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    print(f"   {len(gdf)} feature(s), types={sorted(gdf.geometry.geom_type.unique())}, "
          f"bounds={np.round(gdf.total_bounds, 3).tolist()}")

    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        warn(f"{int(gdf.geometry.isna().sum() + gdf.geometry.is_empty.sum())} "
             "null/empty geometries (ignored)")
    gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)]
    non_poly = ~gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if non_poly.any():
        warn(f"{int(non_poly.sum())} non-polygon feature(s) will be dropped")
    gdf = gdf[~non_poly]
    if gdf.empty:
        fail("no polygon features")
        continue

    bad = gdf[~gdf.geometry.is_valid]
    if len(bad):
        warn(f"{len(bad)} invalid raw feature(s); e.g. "
             f"{explain_validity(bad.geometry.iloc[0])}")
    else:
        ok("all raw features valid")

    h = gdf.geometry.make_valid().buffer(0).union_all().buffer(0)
    if n_nonpoly(h):
        warn(f"healed result has {n_nonpoly(h)} non-polygon part(s) "
             "(load_shape() filters these)")
        h = gpd.GeoSeries(polys_of(h)).union_all()
    if not h.is_valid:
        fail(f"still invalid after healing: {explain_validity(h)}")
    else:
        ok(f"healed: {describe(h)}")
    if h.is_empty:
        fail("empty after healing")
        continue
    healed[name] = (h, gdf.crs)

    # projected validity (what the metric calcs actually use)
    hm = gpd.GeoSeries([h], crs=gdf.crs).to_crs(METRIC).iloc[0]
    if not np.isfinite(hm.bounds).all():
        fail("non-finite coordinates after projection to EPSG:3338")
        continue
    if not hm.is_valid:
        warn(f"invalid after projection to 3338: {explain_validity(hm)}")
        hm = make_valid(hm)
    healed_m[name] = hm
    x0, y0, x1, y1 = hm.bounds
    ok(f"3338 bounds x[{x0/1e3:.0f},{x1/1e3:.0f}] y[{y0/1e3:.0f},{y1/1e3:.0f}] km")

    areas = np.array([p.area for p in polys_of(hm)])
    small = int((areas < SLIVER_M2).sum())
    if small:
        warn(f"{small} polygon(s) < {SLIVER_M2/1e6:.2f} km2 (slivers/islets)")
    ok(f"area {areas.sum()/1e6:,.0f} km2 over {len(areas)} polygon(s)")

# ---------------------------------------------------------------------------
# Rehearsal of the failing step: boundary resampling + make_valid + union_all
# ---------------------------------------------------------------------------
if "domain" in healed_m and not args.skip_resample:
    print("\n== Boundary resampling rehearsal (what crashed the first job)")
    from ocsmesh import utils
    from ocsmesh.internal import MeshData
    try:
        from ocsmesh.engines.gmsh import _polygonal_only
        have_fix = True
    except ImportError:
        have_fix = False
    print(f"   gmsh.py _polygonal_only fix present: {have_fix}")

    dom = healed_m["domain"]
    x0, y0, x1, y1 = dom.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    tests = [(f"const {s:.0f} m", np.full(len(coords), s)) for s in args.sizes]
    lo, hi = min(args.sizes), max(args.sizes)
    tests.append((f"gradient {lo:.0f}->{hi:.0f} m (by y)",
                  lo + (coords[:, 1] - y0) / (y1 - y0) * (hi - lo)))

    for label, vals in tests:
        t0 = time.time()
        sizing = MeshData(coords, values=vals, crs=METRIC)
        series = gpd.GeoSeries([dom], crs=METRIC)
        res = utils.resample_geom_by_hfun(series, sizing)
        res = res.apply(lambda g: make_valid(g) if not g.is_valid else g)
        comb = res.union_all()
        extra = n_nonpoly(comb)
        line = (f"{label:<28} -> {describe(comb)}  [{time.time()-t0:.0f}s]")
        if comb.is_empty:
            fail(line + "  EMPTY")
        elif comb.geom_type == "GeometryCollection":
            msg = line + f"  GeometryCollection with {extra} non-polygon part(s)"
            if have_fix:
                fixed = _polygonal_only(comb)
                warn(msg + f"; fix handles it -> {describe(fixed)}")
            else:
                fail(msg + "  (this is what crashed Gmsh; apply the patch)")
        else:
            ok(line)

print(f"\n{problems} problem(s), {warnings_} warning(s)")
sys.exit(1 if problems else 0)
