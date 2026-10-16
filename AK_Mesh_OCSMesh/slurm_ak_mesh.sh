#!/bin/bash
# =============================================================================
# Alaska large-domain mesh (0-360 lon, crosses the dateline)  |  NOAA Gaea (c6)
# Submit one mesh case (see CASES in build_ak_mesh.py):
#   sbatch --export=ALL,CASE=m1_shelf150 slurm_ak_mesh.sh
# or all four at once:  ./submit_cases.sh
# Other overrides: METHOD=fast|exact  ENGINE=gmsh|triangle  NPROCS=n
# =============================================================================
#SBATCH --job-name=ak_mesh
#SBATCH --account=nos-surge
#SBATCH --partition=batch
#SBATCH --clusters=c6
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=logs/ak_mesh_%j.out
#SBATCH --error=logs/ak_mesh_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felicio.cassalho@noaa.gov

set -euo pipefail

# ── Configuration (edit or override via --export) ────────────────────────────
DATA_DIR="${DATA_DIR:-/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI/AK_LargeDomain_Mesh/2026_10_15}"
CASE="${CASE:?Set CASE, e.g. sbatch --export=ALL,CASE=m1_shelf150 slurm_ak_mesh.sh}"
METHOD="${METHOD:-exact}"          # exact | fast
OUT_DIR="${OUT_DIR:-/gpfs/f6/nos-surge/scratch/Felicio.Cassalho/OCSMesh_MPI/AK_LargeDomain_Mesh/results/${CASE}_${METHOD}_${SLURM_JOB_ID:-manual}}"
ENGINE="${ENGINE:-gmsh}"           # gmsh | triangle
NPROCS="${NPROCS:-${SLURM_CPUS_PER_TASK:-1}}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# ── Environment ──────────────────────────────────────────────────────────────
module load PrgEnv-intel
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
source "/ncrc/proj/nos-surge/Felicio.Cassalho/miniconda3/etc/profile.d/conda.sh"
conda activate "ocsmesh_mpi_test"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Temporary files on the parallel filesystem (node-local /tmp is too small)
mkdir -p "${OUT_DIR}/tmp" "${SCRIPT_DIR}/logs"
export TMPDIR="${OUT_DIR}/tmp"
export OCSMESH_SHARED_TMPDIR="${OUT_DIR}/tmp"

cd "${SCRIPT_DIR}"

echo "============================================================="
echo "  Alaska 0-360 mesh   method=${METHOD}  engine=${ENGINE}  nprocs=${NPROCS}"
echo "  Case   : ${CASE}"
echo "  Job    : ${SLURM_JOB_ID:-n/a}"
echo "  Node   : ${SLURM_NODELIST:-n/a}"
echo "  Data   : ${DATA_DIR}"
echo "  Output : ${OUT_DIR}"
echo "  Date   : $(date)"
echo "============================================================="

# Fail early if an input is missing
for f in AKdomain_20261014_360.shp 150m_contour_360.shp below_200m_contour_360.shp \
         beringsea_360.shp coarse_ocean_center_360.shp coarse_ocean_south_360.shp \
         coarse_ocean_north_360.shp GEBCO_2026_360.tif; do
    [ -f "${DATA_DIR}/${f}" ] || { echo "ERROR: missing ${DATA_DIR}/${f}"; exit 1; }
done

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    python build_ak_mesh.py \
        --data-dir "${DATA_DIR}" \
        --out-dir  "${OUT_DIR}" \
        --method   "${METHOD}" \
        --engine   "${ENGINE}" \
        --nprocs   "${NPROCS}" \
        --case     "${CASE}"

echo "=== DONE $(date) ==="
