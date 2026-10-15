#!/usr/bin/env python
"""Pre-flight checks for the Alaska 0-360 mesh job. Run before sbatch.

    python preflight_check.py [DATA_DIR] [ORIGINAL_GEBCO_TIF]

Exit code is non-zero if any check FAILS.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI/"
            "AK_LargeDomain_Mesh/2026_10_15")
ORIG = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "GEBCO_2026.tif"
SHAPES = ["AKdomain_20261014_360", "150m_contour_360", "200m_contour_360",
          "beringsea_360", "coarse_ocean_center_360",
          "coarse_ocean_south_360", "coarse_ocean_north_360"]
RASTER = DATA / "GEBCO_2026_360.tif"

fails = 0


def report(ok, msg):
    global fails
    print(("[ OK ] " if ok else "[FAIL] ") + msg)
    fails += (not ok)


# 1. Python packages / OCSMesh version ---------------------------------------
try:
    import ocsmesh
    pkg = Path(ocsmesh.__file__).resolve().parent
    report(hasattr(ocsmesh, "MeshDriver"), f"ocsmesh at {pkg} has MeshDriver")
    repo = pkg.parent
    if (repo / ".git").exists():
        br = subprocess.run(["git", "-C", str(repo), "rev-parse",
                             "--abbrev-ref", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
        sha = subprocess.run(["git", "-C", str(repo), "log", "-1",
                              "--format=%h %s"], capture_output=True,
                             text=True).stdout.strip()
        report(br == "dev", f"OCSMesh git branch = {br!r} (want 'dev'); {sha}")
    else:
        print("[WARN] ocsmesh is not a git checkout; cannot verify branch")
    from ocsmesh import utils
    report(hasattr(utils, "estimate_bounds_utm"), "utils.estimate_bounds_utm exists")
except Exception as exc:  # noqa: BLE001
    report(False, f"import ocsmesh failed: {exc}")

try:
    import geopandas as gpd
    report(hasattr(gpd.GeoSeries, "union_all"),
           f"geopandas {gpd.__version__} has union_all (needs >= 1.0)")
except Exception as exc:  # noqa: BLE001
    report(False, f"geopandas: {exc}")
try:
    import gmsh
    report(True, f"gmsh {gmsh.__version__}")
except Exception as exc:  # noqa: BLE001
    report(False, f"gmsh import: {exc}")

# 2. Input files exist --------------------------------------------------------
for stem in SHAPES:
    for ext in (".shp", ".shx", ".dbf", ".prj"):
        p = DATA / (stem + ext)
        report(p.is_file(), f"exists {p.name}")
report(RASTER.is_file(), f"exists {RASTER.name}")

# 3. Shapefiles are 0-360 and WGS84 geographic --------------------------------
try:
    for stem in SHAPES:
        gdf = gpd.read_file(DATA / (stem + ".shp"))
        x0, y0, x1, y1 = gdf.total_bounds
        geographic = gdf.crs is not None and gdf.crs.is_geographic
        ok = geographic and x0 >= 0 and x1 <= 360 and -90 <= y0 and y1 <= 90
        report(ok, f"{stem}: lon {x0:.2f}..{x1:.2f} lat {y0:.2f}..{y1:.2f} "
                   f"crs={gdf.crs.to_string() if gdf.crs else None}")
        if stem == SHAPES[0]:
            report(x1 > 180 and x0 < 180, "domain straddles 180 (crosses dateline)")
except Exception as exc:  # noqa: BLE001
    report(False, f"shapefile checks: {exc}")

# 4. Raster is 0-360 AND data was rolled (not just re-georeferenced) ----------
try:
    import numpy as np
    import rasterio
    with rasterio.open(RASTER) as r360:
        b = r360.bounds
        report(r360.crs is not None and r360.crs.is_geographic,
               f"raster crs={r360.crs}")
        report(b.left >= -1e-6 and b.right <= 360 + 1e-6 and b.right > 180,
               f"raster lon {b.left:.3f}..{b.right:.3f} lat {b.bottom:.2f}..{b.top:.2f}")
        if ORIG.is_file():
            pts = [(170.5, 62.0), (175.0, 60.0), (179.5, 55.0),
                   (181.0, 55.0), (185.0, 58.0), (190.0, 60.0), (200.0, 55.0),
                   (210.0, 58.0), (190.0, 65.0)]
            with rasterio.open(ORIG) as ro:
                ob = ro.bounds
                n_cmp, bad = 0, 0
                for lon, lat in pts:
                    lon_o = lon - 360 if lon > 180 else lon
                    if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                        continue
                    if not (ob.left <= lon_o <= ob.right and ob.bottom <= lat <= ob.top):
                        continue
                    v360 = next(r360.sample([(lon, lat)]))[0]
                    vo = next(ro.sample([(lon_o, lat)]))[0]
                    n_cmp += 1
                    good = np.isclose(v360, vo, atol=1.0)
                    bad += (not good)
                    print(f"        lon {lon:6.1f} (orig {lon_o:7.1f}) lat {lat:5.1f}: "
                          f"360={v360:8.1f}  orig={vo:8.1f}  {'ok' if good else 'MISMATCH'}")
                report(n_cmp > 0 and bad == 0,
                       f"raster values match original at {n_cmp} sample points")
        else:
            print(f"[WARN] original raster {ORIG} not found; skipped value check")
except Exception as exc:  # noqa: BLE001
    report(False, f"raster checks: {exc}")

# 5. Scratch space -------------------------------------------------------------
tmp = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
try:
    with tempfile.NamedTemporaryFile(dir=tmp):
        pass
    free_gb = shutil.disk_usage(tmp).free / 1e9
    report(free_gb > 20, f"TMPDIR {tmp} writable, {free_gb:.0f} GB free")
except Exception as exc:  # noqa: BLE001
    report(False, f"TMPDIR {tmp}: {exc}")

print("\nALL CHECKS PASSED" if not fails else f"\n{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
