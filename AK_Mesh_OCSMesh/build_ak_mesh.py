#!/usr/bin/env python
"""Build the Alaska large-domain mesh with OCSMesh (dateline-safe, 0-360 lon).

All inputs (domain/refinement shapefiles and the GEBCO raster) must already be
in the continuous 0-360 longitude convention (``*_360.shp`` / ``*_360.tif``).
The output mesh is written in EPSG:4326 with longitudes in [0, 360):
0 = Greenwich, 180 = Bering Sea (dateline), 360 = Greenwich again.

How the dateline is handled
---------------------------
OCSMesh's ``utils.estimate_bounds_utm`` returns ``None`` for any longitude
> 180, which silently makes the size-function distance calculations run in
degrees and breaks CRS selection in ``MeshDriver``. We therefore replace it
with a function that always returns one metric CRS (EPSG:3338, Alaska Albers),
which PROJ evaluates correctly for 0-360 input. After meshing, the mesh is
reprojected to EPSG:4326 (-180..180) and wrapped with ``lon % 360``. Wrapping
does not alter connectivity.

Usage (normally launched through slurm_ak_mesh.sh)
    python build_ak_mesh.py --data-dir DIR --out-dir DIR [--nprocs N]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ocsmesh import Geom, Hfun, MeshDriver, Raster, utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("AK_Mesh")

# --------------------------------------------------------------------------
# Inputs (file names inside --data-dir)
# --------------------------------------------------------------------------
DOMAIN_SHP = "AKdomain_20261014_360.shp"
SHELF_150_SHP = "150m_contour_360.shp"
SHELF_200_SHP = "200m_contour_360.shp"
BERING_SHP = "beringsea_360.shp"
COARSE_CENTER_SHP = "coarse_ocean_center_360.shp"
COARSE_SOUTH_SHP = "coarse_ocean_south_360.shp"
COARSE_NORTH_SHP = "coarse_ocean_north_360.shp"
DEM_GEBCO = "GEBCO_2026_360.tif"

# --------------------------------------------------------------------------
# Mesh sizing recipe (meters)
# --------------------------------------------------------------------------
HMIN = 1500.0
HMAX = 3500.0
SHELF_TARGET = 1500.0
SHELF_EXPANSION = 0.0001
# (shape key, value, rate or None)
REGION_CONSTRAINTS = [
    ("shelf_200", 2500.0, None),
    ("bering", 2500.0, 0.001),
    ("center", 3500.0, 0.005),
    ("south", 3500.0, 0.0025),
    ("north", 3500.0, 0.0025),
]

METRIC_CRS = CRS.from_epsg(3338)   # Alaska Albers (meters)
OUTPUT_CRS = "EPSG:4326"


# --------------------------------------------------------------------------
# Dateline handling
# --------------------------------------------------------------------------
def install_dateline_safe_metric_crs(metric_crs: CRS = METRIC_CRS) -> None:
    """Make every OCSMesh metric-CRS lookup return ``metric_crs``.

    Every internal caller (Geom, Hfun, Raster, MeshDriver) goes through
    ``ocsmesh.utils.estimate_bounds_utm``, so replacing it once is enough.
    """

    def _metric(bounds, crs="EPSG:4326"):  # noqa: ARG001
        return metric_crs if CRS.from_user_input(crs).is_geographic else None

    utils.estimate_bounds_utm = _metric
    _logger.info("Metric CRS for all calculations forced to %s", metric_crs.to_string())


def wrap_lon_360(mesh) -> None:
    """Wrap node longitudes of a geographic Mesh to [0, 360) in place."""
    md = mesh.meshdata
    if md.crs is None or not CRS.from_user_input(md.crs).is_geographic:
        raise ValueError(f"Expected a geographic CRS to wrap, got {md.crs}")
    coords = md.coords.copy()
    coords[:, 0] = np.mod(coords[:, 0], 360.0)
    md.coords = coords        # values/connectivity untouched (same node count)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_shape(shp_path: Path):
    """Read a 0-360 shapefile, heal it and return (Polygon|MultiPolygon, crs)."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise ValueError(f"No polygon geometry in {shp_path}")
    gdf["geometry"] = gdf.geometry.make_valid().buffer(0)
    shape = gdf.union_all().buffer(0)
    if shape.geom_type == "GeometryCollection":
        shape = unary_union(
            [g for g in shape.geoms if isinstance(g, (Polygon, MultiPolygon))]
        )
    return shape, gdf.crs


def check_0_360(name: str, shape) -> None:
    x0, _, x1, _ = shape.bounds
    _logger.info("  %-14s lon range %.3f .. %.3f", name, x0, x1)
    if x0 < 0 or x1 > 360:
        raise ValueError(
            f"{name} is not in the 0-360 convention (lon {x0:.2f}..{x1:.2f}). "
            "Use the *_360 inputs."
        )


def mesh_summary(mesh) -> dict:
    md = mesh.meshdata
    lon, lat = md.coords[:, 0], md.coords[:, 1]
    tri = md.tria
    # Elements whose lon span is huge would indicate a wrap/connectivity bug.
    span = np.ptp(lon[tri], axis=1) if len(tri) else np.array([0.0])
    return {
        "nodes": int(md.num_nodes),
        "triangles": int(len(tri)),
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "max_element_lon_span_deg": float(span.max()),
        "crs": str(md.crs),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory with the *_360 shapefiles and GEBCO tif")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="AK_LARGEDOMAIN")
    ap.add_argument("--nprocs", type=int, default=1,
                    help="Worker processes for size-function evaluation")
    ap.add_argument("--method", choices=["exact", "fast"], default="exact")
    ap.add_argument("--engine", choices=["gmsh", "triangle"], default="gmsh")
    ap.add_argument("--hmin", type=float, default=HMIN)
    ap.add_argument("--hmax", type=float, default=HMAX)
    args = ap.parse_args()

    data, out = args.data_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timings = {}

    install_dateline_safe_metric_crs()

    # 1. Shapes ---------------------------------------------------------
    _logger.info("Step 1: loading shapefiles from %s", data)
    t = time.time()
    domain, domain_crs = load_shape(data / DOMAIN_SHP)
    shapes = {
        "shelf_150": load_shape(data / SHELF_150_SHP)[0],
        "shelf_200": load_shape(data / SHELF_200_SHP)[0],
        "bering": load_shape(data / BERING_SHP)[0],
        "center": load_shape(data / COARSE_CENTER_SHP)[0],
        "south": load_shape(data / COARSE_SOUTH_SHP)[0],
        "north": load_shape(data / COARSE_NORTH_SHP)[0],
    }
    check_0_360("domain", domain)
    for k, v in shapes.items():
        check_0_360(k, v)
    timings["load_shapes_s"] = time.time() - t

    # 2. Geometry -------------------------------------------------------
    # Empty raster list: avoids raster-boundary extraction across the dateline;
    # the domain polygon fully defines the geometry.
    _logger.info("Step 2: building Geom")
    t = time.time()
    geom = Geom(
        [],
        base_shape=domain,
        base_shape_crs=domain_crs,
        zmin=-10000.0,
        zmax=99999.0,
    )
    timings["geom_s"] = time.time() - t

    # 3. Size function --------------------------------------------------
    _logger.info("Step 3: building Hfun (method=%s, nprocs=%d)",
                 args.method, args.nprocs)
    t = time.time()
    rast = Raster(data / DEM_GEBCO)
    _logger.info("  GEBCO bbox: %s  crs: %s", rast.get_bbox().bounds, rast.crs)
    hfun = Hfun(
        [rast],
        hmin=args.hmin,
        hmax=args.hmax,
        nprocs=args.nprocs,
        base_shape=domain,
        base_shape_crs=domain_crs,
        method=args.method,
    )
    if args.nprocs > 1:
        hfun.execution_mode = "parallel"

    hfun.add_patch(
        shape=shapes["shelf_150"],
        expansion_rate=SHELF_EXPANSION,
        target_size=SHELF_TARGET,
    )
    for key, value, rate in REGION_CONSTRAINTS:
        kwargs = dict(value=value, shape=shapes[key], crs=domain_crs,
                      value_type="min")
        if rate is not None:
            kwargs["rate"] = rate
        hfun.add_region_constraint(**kwargs)
    timings["hfun_setup_s"] = time.time() - t

    # 4. Mesh -----------------------------------------------------------
    _logger.info("Step 4: running MeshDriver (engine=%s)", args.engine)
    t = time.time()
    driver = MeshDriver(
        geom, hfun,
        crs=OUTPUT_CRS,
        engine_name=args.engine,
        bnd_representation="adapt",
    )
    mesh = driver.run()
    timings["mesh_s"] = time.time() - t

    # 5. Wrap to 0-360 and write ---------------------------------------
    _logger.info("Step 5: wrapping longitudes to 0-360 and writing")
    wrap_lon_360(mesh)
    summary = mesh_summary(mesh)
    _logger.info("Mesh summary: %s", summary)
    if summary["lon_min"] < 0 or summary["lon_max"] >= 360:
        raise RuntimeError("Output longitudes are not within [0, 360)")
    if summary["max_element_lon_span_deg"] > 90:
        raise RuntimeError(
            "Element spanning >90 deg of longitude found; mesh wraps incorrectly"
        )

    mesh.write(out / f"{args.name}.2dm", format="2dm", overwrite=True)
    mesh.write(out / f"{args.name}.gr3", format="grd", overwrite=True)

    timings["total_s"] = time.time() - t_start
    with open(out / f"{args.name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()},
                   "timings": timings, "mesh": summary}, fh, indent=2)
    _logger.info("Done in %.1f min. Output: %s", timings["total_s"] / 60, out)


if __name__ == "__main__":
    main()
