#!/usr/bin/env python
"""Build only the size function for one case and plot it (no meshing).

Meant for quick local checks on a coarsened DEM, e.g. on a laptop:

    python local_hfun_check.py --data-dir ~/ak_data_coarse --case m1_shelf150

It uses the same recipe as build_ak_mesh.py (apply_refinements), calls
``hfun.meshdata()`` and scatter-plots the size values against longitude
(0-360) so west-vs-east of the dateline can be compared at a glance.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ak_mesh as B  # noqa: E402
from ocsmesh import Hfun, Raster  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", type=Path, required=True)
ap.add_argument("--case", choices=list(B.CASES), default="m1_shelf150")
ap.add_argument("--method", choices=["fast", "exact"], default="fast")
ap.add_argument("--nprocs", type=int, default=1)
ap.add_argument("--hmin", type=float, default=B.HMIN)
ap.add_argument("--hmax", type=float, default=B.HMAX)
ap.add_argument("--png", default=None)
args = ap.parse_args()

B.install_dateline_safe_metric_crs()
data = args.data_dir.resolve()
refine = B.CASES[args.case]
print(f"case {args.case}: {refine}")

domain, dcrs = B.load_shape(data / B.DOMAIN_SHP)
shapes = {k: B.load_shape(data / B.REFINEMENTS[k]["shp"])[0] for k in refine}

t0 = time.time()
hfun = Hfun([Raster(data / B.DEM_GEBCO)], hmin=args.hmin, hmax=args.hmax,
            nprocs=args.nprocs, base_shape=domain, base_shape_crs=dcrs,
            method=args.method)
if args.nprocs > 1:
    hfun.execution_mode = "parallel"
B.apply_refinements(hfun, refine, shapes, dcrs)
md = hfun.meshdata()
print(f"meshdata: {md}   [{time.time() - t0:.0f}s]")

xy, vals = md.coords, np.asarray(md.values).ravel()
if md.crs is not None and not md.crs.is_geographic:
    lon, lat = Transformer.from_crs(md.crs, 4326, always_xy=True).transform(
        xy[:, 0], xy[:, 1])
else:
    lon, lat = xy[:, 0], xy[:, 1]
lon = np.mod(np.asarray(lon), 360.0)
lat = np.asarray(lat)

west, east = lon <= 180.0, lon > 180.0
for name, m in (("west (<=180)", west), ("east (>180)", east)):
    v = vals[m]
    print(f"  {name:13s} n={m.sum():8d}  size min/mean/max = "
          f"{v.min():7.0f} / {v.mean():7.0f} / {v.max():7.0f} m")

png = args.png or f"hfun_{args.case}_{args.method}.png"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(14, 8))
sc = ax.scatter(lon, lat, c=vals, s=1, cmap="viridis_r",
                vmin=args.hmin, vmax=args.hmax, rasterized=True)
ax.axvline(180, color="r", lw=0.8)
ax.set_xlabel("longitude (0-360)")
ax.set_ylabel("latitude")
ax.set_title(f"{args.case} size function ({args.method}), red = 180")
fig.colorbar(sc, label="size (m)")
fig.tight_layout()
fig.savefig(png, dpi=110)
print("wrote", png)
