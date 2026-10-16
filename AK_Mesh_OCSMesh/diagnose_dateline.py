#!/usr/bin/env python
"""Find where refinements break east of the dateline (lon > 180).

Runs the individual steps OCSMesh's size function performs, on a coarse 10 km
EPSG:3338 grid, and compares west (lon <= 180) against east (lon > 180):

  T1  PROJ forward/inverse longitude behaviour
  T2  rasterio.warp.reproject of GEBCO_2026_360.tif (does the east half survive?)
  T3  patch mask: shelf polygon (4326, 0-360) -> 3338 -> rasterized
  T4  RegionConstraint.apply on the Bering polygon (point-in-polygon in 4326)

    python diagnose_dateline.py [DATA_DIR] [--res 10000] [--png diag.png]

Takes about a minute. Writes diag_dateline.png (if matplotlib is available).
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ak_mesh as B  # noqa: E402
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shp_transform

ap = argparse.ArgumentParser()
ap.add_argument("data_dir", nargs="?", type=Path, default=Path(
    "/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI/"
    "AK_LargeDomain_Mesh/2026_10_15"))
ap.add_argument("--res", type=float, default=10000.0, help="grid size (m)")
ap.add_argument("--png", default="diag_dateline.png")
args = ap.parse_args()

D = args.data_dir
C4326, C3338 = CRS.from_epsg(4326), CRS.from_epsg(3338)
fwd = Transformer.from_crs(C4326, C3338, always_xy=True)
inv = Transformer.from_crs(C3338, C4326, always_xy=True)
bad = 0


def verdict(ok, msg):
    global bad
    print(("   [ OK ] " if ok else "   [FAIL] ") + msg)
    bad += (not ok)


def load(fn):
    shape, crs = B.load_shape(B.resolve_shp(D, fn))
    return (shape if shape.geom_type == "MultiPolygon" else MultiPolygon([shape])), crs


# ---------------------------------------------------------------- T1
print("\nT1  PROJ longitude behaviour (4326 -> 3338 -> 4326)")
for lon in (170.0, 179.9, 180.1, 190.0, 210.0):
    x, y = fwd.transform(lon, 60.0)
    lon_back, _ = inv.transform(x, y)
    print(f"      lon {lon:6.1f} -> x={x/1e3:9.1f} km -> back lon {lon_back:8.2f}")
x_a, _ = fwd.transform(179.0, 60.0)
x_b, _ = fwd.transform(181.0, 60.0)
verdict(0 < x_b - x_a < 200e3, f"forward transform is continuous across 180 "
                               f"(dx for 2 deg = {(x_b - x_a)/1e3:.0f} km)")
lon_back, _ = inv.transform(*fwd.transform(190.0, 60.0))
print(f"      NOTE: inverse returns {lon_back:.1f} for lon 190 -> anything that "
      "converts back to 4326 loses the 0-360 convention")

# ---------------------------------------------------------------- grid
domain, _ = load(B.DOMAIN_SHP)
dom_m = shp_transform(fwd.transform, domain)
x0, y0, x1, y1 = dom_m.bounds
res = args.res
W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
tfm = from_origin(x0, y1, res, res)
cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
gx, gy = x0 + cols * res, y1 - rows * res
glon, glat = inv.transform(gx, gy)
glon360 = np.mod(glon, 360.0)
east = glon360 > 180.0
west = ~east
print(f"\nGrid {W}x{H} @ {res/1e3:.0f} km; west cells {int(west.sum())}, "
      f"east cells {int(east.sum())}")

# ---------------------------------------------------------------- T2
print("\nT2  rasterio.warp.reproject of GEBCO_2026_360.tif -> 3338 grid")
NODATA = -99999.0
warped = np.full((H, W), NODATA, dtype=np.float32)
with rasterio.open(B.ensure_gebco_360(D)) as src:
    print(f"      source: crs={src.crs} bounds={tuple(round(b, 2) for b in src.bounds)}")
    reproject(source=rasterio.band(src, 1), destination=warped,
              dst_transform=tfm, dst_crs=C3338, resampling=Resampling.nearest,
              init_dest_nodata=False)
    # reference: sample the 360 file directly at the same lon/lat (0-360)
    ref = np.full((H, W), np.nan, dtype=np.float32)
    inside = ((glon360 >= src.bounds.left) & (glon360 <= src.bounds.right)
              & (glat >= src.bounds.bottom) & (glat <= src.bounds.top))
    pts = list(zip(glon360[inside], glat[inside]))
    ref[inside] = np.array([v[0] for v in src.sample(pts)], dtype=np.float32)

valid_ref = ~np.isnan(ref)
for name, m in (("west", west), ("east", east)):
    sel = m & valid_ref
    nod = float((warped[sel] == NODATA).mean()) if sel.any() else float("nan")
    err = np.abs(warped[sel] - ref[sel])
    ok_frac = float((err < 100).mean()) if sel.any() else float("nan")
    print(f"      {name}: cells={int(sel.sum()):7d}  unfilled(nodata)={nod:6.1%}  "
          f"match-direct-sample(<100 m)={ok_frac:6.1%}")
    verdict(nod < 0.02 and ok_frac > 0.90,
            f"warp of GEBCO is correct in the {name} half")

# ---------------------------------------------------------------- T3
print("\nT3  patch mask (150 m shelf polygon, as add_patch rasterizes it)")
shelf, _ = load(B.SHELF_150_SHP)
shelf_m = shp_transform(fwd.transform, shelf)
mask_patch = features.geometry_mask(shelf_m.geoms, out_shape=(H, W),
                                    transform=tfm, all_touched=True, invert=True)
# truth: point-in-polygon in the polygon's own (0-360) frame
from shapely import contains_xy  # noqa: E402
truth = contains_xy(shelf, glon360.ravel(), glat.ravel()).reshape(H, W)
for name, m in (("west", west), ("east", east)):
    n_mask, n_true = int((mask_patch & m).sum()), int((truth & m).sum())
    both = int((mask_patch & truth & m).sum())
    print(f"      {name}: mask={n_mask:7d}  truth={n_true:7d}  overlap={both:7d}")
    verdict(n_true == 0 or both / max(n_true, 1) > 0.9,
            f"patch mask agrees with true polygon in the {name} half")

# ---------------------------------------------------------------- T4
print("\nT4  RegionConstraint.apply on the Bering polygon")
try:
    from ocsmesh.features.constraint import RegionConstraint
    bering, bcrs = load(B.BERING_SHP)
    rc = RegionConstraint(value=2500.0, shape=bering, crs=bcrs,
                          value_type="min", rate=None)
    old = np.full((H, W), 1500.0)
    locs = np.column_stack([gx.ravel(), gy.ravel()])
    out = rc.apply(old.copy(), old.copy(), locations=locs, crs=C3338)
    applied = (out.reshape(H, W) >= 2500.0)
    btruth = contains_xy(bering, glon360.ravel(), glat.ravel()).reshape(H, W)
    for name, m in (("west", west), ("east", east)):
        n_app, n_true = int((applied & m).sum()), int((btruth & m).sum())
        print(f"      {name}: applied={n_app:7d}  expected={n_true:7d}")
        verdict(n_true == 0 or n_app / max(n_true, 1) > 0.9,
                f"constraint applied in the {name} half")
except Exception as exc:  # noqa: BLE001
    verdict(False, f"T4 could not run: {exc!r}")

# ---------------------------------------------------------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    ext = (x0, x0 + W * res, y1 - H * res, y1)
    w = np.where(warped == NODATA, np.nan, warped)
    ax[0].imshow(w, extent=ext, cmap="terrain", vmin=-6000, vmax=2000)
    ax[0].set_title("T2 GEBCO warped to 3338 (blank = unfilled)")
    ax[1].imshow(mask_patch.astype(int) + 2 * truth.astype(int), extent=ext,
                 cmap="viridis")
    ax[1].set_title("T3 patch mask: 1=raster mask 2=truth 3=both")
    try:
        ax[2].imshow(applied.astype(int) + 2 * btruth.astype(int), extent=ext,
                     cmap="viridis")
        ax[2].set_title("T4 Bering constraint: 1=applied 2=expected 3=both")
    except NameError:
        pass
    x180, _ = fwd.transform(180.0, 60.0)
    for a in ax:
        a.axvline(fwd.transform(180.0, 52.0)[0], color="r", lw=0.8)
        a.axvline(fwd.transform(180.0, 70.0)[0], color="r", lw=0.8)
    fig.tight_layout()
    fig.savefig(args.png, dpi=110)
    print(f"\nFigure written: {args.png} (red lines ~ lon 180)")
except Exception as exc:  # noqa: BLE001
    print(f"\n(no figure: {exc})")

print(f"\n{bad} FAIL(s)")
sys.exit(1 if bad else 0)
