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
`AKdomain_20261014_360.shp`, `150m_contour_360.shp`, `200m_contour_360.shp`,
`beringsea_360.shp`, `coarse_ocean_{center,south,north}_360.shp`,
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

## How the dateline is handled
OCSMesh's `utils.estimate_bounds_utm` returns `None` for any longitude
> 180, which makes size-function distances run in degrees and breaks CRS
selection in `MeshDriver`. `build_ak_mesh.py` replaces it so all metric
calculations use EPSG:3338 (Alaska Albers), reprojects the finished mesh to
EPSG:4326 and wraps `lon % 360`. The script aborts if any output longitude is
outside [0, 360) or any element spans more than 90 degrees of longitude.

Requires the OCSMesh `dev` branch with `MeshDriver`, and the conda env
`ocsmesh_mpi_test`.
