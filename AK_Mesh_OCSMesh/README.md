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
all in 0-360 longitude:
`AKdomain_20261014_360.shp`, `150m_contour_360.shp`,
`below_200m_contour_360.shp`, `beringsea_360.shp`, `coarse_ocean_{center,south,north}_360.shp`,
`GEBCO_2026_360.tif`.

Before the first run, check the 360 raster really was rolled (not just
re-georeferenced): sample the same Bering Sea point in the original at
170 W and in the 360 file at 190 E; depths must match.

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
| `m1_shelf150` | 1500 m patch on the 150 m contour polygon, rate 1e-5 |
| `m2_plus_below200` | m1 + "no finer than 2500 m" constraint in `below_200m_contour` |
| `m3_plus_bering` | m2 + 2500 m **patch** on the Bering Sea polygon, rate 5e-6 (patches take the min, so it cannot coarsen the 1500 m shelf patch) |
| `m4_plus_coarse` | m3 + "no finer than 3500 m" in the center, south and north regions (these win over `below_200m` where they overlap) |

Rates: 1e-5 for the shelf patch and all constraints, 5e-6 for the Bering patch.

```bash
./submit_cases.sh                       # all four, METHOD=fast (~30 min each)
METHOD=exact ./submit_cases.sh m4_plus_coarse
sbatch --export=ALL,CASE=m1_shelf150,METHOD=fast slurm_ak_mesh.sh   # a single case
```
Outputs: `results/<case>_<method>_<jobid>/AK_LARGEDOMAIN_<case>.{2dm,gr3}`.

Rate units (1/m): patch `size(d) = T*(1 + r*d)`; region `min` constraint
`bound(d) = V*(1 - r*d)`. Tune `RATE` (1e-5) / `BERING_RATE` (5e-6) in `build_ak_mesh.py`.

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
