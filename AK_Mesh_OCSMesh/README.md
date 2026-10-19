# AK_Mesh_OCSMesh

OCSMesh workflow for the Alaska large-domain mesh, which crosses the
dateline. Output longitudes are **0-360** (0 = Greenwich, 180 = Bering Sea,
360 = Greenwich again).

## Files
| File | Purpose |
|---|---|
| `build_ak_mesh.py` | Builds Geom + Hfun, runs `MeshDriver`, wraps lon to 0-360, writes `.2dm`/`.gr3` + summary JSON |
| `slurm_ak_mesh.sh` | Gaea (`c6`) batch script |

## Inputs
In `DATA_DIR` (default
`/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI/AK_LargeDomain_Mesh/2026_10_15`),
the **original** files, no manual shifting needed:
`AKdomain_20261014.shp`, `150m_contour.shp`, `below_150m_contour.shp`
(any single `below_150*.shp` is accepted), `beringsea.shp`,
`coarse_ocean_{center,south,north}.shp`, `GEBCO_2026.tif`.

`build_ak_mesh.py` shifts them to 0-360 on every run: shapefile vertices with
lon < 0 get +360 in memory (already-0-360 files pass through unchanged), so you
can edit a polygon and resubmit. The shifted shapes are saved to
`<OUT_DIR>/shapes_360/` to open in QGIS. The GEBCO raster is rolled once into
`GEBCO_2026_360.tif` next to the original (a pure column permutation, no
resampling) and reused while newer than `GEBCO_2026.tif`; if the original is
already 0-360 it is used as is. `--force-raster` rebuilds it. An existing
`GEBCO_2026_360.tif` that is newer than the original is trusted as is.

## Run
```bash
cd AK_Mesh_OCSMesh
sbatch slurm_ak_mesh.sh
# overrides
sbatch --export=ALL,METHOD=fast,NPROCS=16 slurm_ak_mesh.sh
squeue --me -M c6
```
Outputs go to `OUT_DIR` (default `.../AK_LargeDomain_Mesh/results/<jobid>/`):
`AK_LARGEDOMAIN.2dm`, `AK_LARGEDOMAIN.gr3`, `AK_LARGEDOMAIN_summary.json`.

## Mesh cases (one mesh per refinement step)
| Case | Refinements |
|---|---|
| `m1_shelf150` | 1500 m patch on the 150 m contour polygon, rate 1e-6 |
| `m2_plus_below150` | m1 + "no coarser than 2500 m" (max) constraint in the `below_150` polygon |
| `m3_plus_bering` | m2 + 2500 m **patch** on the Bering Sea polygon, rate 5e-6 (patches take the min, so it cannot coarsen the 1500 m shelf patch) |
| `m4_plus_coarse` | m3 + "no finer than 3500 m" in the center, south and north regions (added after `below_150`, so they win where they overlap) |

Rates: 1e-6 for the shelf patch, 1e-5 for all constraints, 5e-6 for the Bering patch.

```bash
./submit_cases.sh                       # all four, METHOD=fast (~30 min each)
METHOD=exact ./submit_cases.sh m4_plus_coarse
sbatch --export=ALL,CASE=m1_shelf150,METHOD=fast slurm_ak_mesh.sh   # a single case
```
Outputs: `results/<case>_<method>_<jobid>/AK_LARGEDOMAIN_<case>.{2dm,gr3}`.

Rate units (1/m): patch `size(d) = T*(1 + r*d)`; region `min` constraint
`bound(d) = V*(1 - r*d)`. Tune `SHELF_RATE` (1e-6) / `RATE` (1e-5) / `BERING_RATE` (5e-6) in `build_ak_mesh.py`.

## OCSMesh library patches (apply to the OCSMesh `dev` checkout on Gaea)
```bash
cd <OCSMesh checkout>
for p in gmsh_polygonal_only constraint_lon360 hfun_lon360; do
  git apply --check <path>/AK_Mesh_OCSMesh/$p.patch && git apply <path>/AK_Mesh_OCSMesh/$p.patch
done
```
- `gmsh_polygonal_only.patch`: drop non-polygon parts (GeometryCollection) before Gmsh.
- `constraint_lon360.patch`: `RegionConstraint` wraps points to 0-360 (constraints
  were ignored east of 180).
- `hfun_lon360.patch`: `fast` size function no longer clips away everything east of 180.

## Local testing
`diagnose_dateline.py` (seconds) and `local_hfun_check.py` (size function only, plot)
run on a laptop with a conda env from OCSMesh's `environment.yml` and a coarsened DEM.

## How the dateline is handled
OCSMesh's `utils.estimate_bounds_utm` returns `None` for any longitude
> 180, which makes size-function distances run in degrees and breaks CRS
selection in `MeshDriver`. `build_ak_mesh.py` replaces it so all metric
calculations use EPSG:3338 (Alaska Albers), reprojects the finished mesh to
EPSG:4326 and wraps `lon % 360`. The script aborts if any output longitude is
outside [0, 360) or any element spans more than 90 degrees of longitude.

Requires the OCSMesh `dev` branch with `MeshDriver`, and the conda env
`ocsmesh_mpi_test`.
