#!/bin/bash
# Submit one Gaea job per mesh case.
#   ./submit_cases.sh                 # all four cases, METHOD=fast
#   METHOD=exact ./submit_cases.sh    # production quality (hours per case)
#   ./submit_cases.sh m1_shelf150 m3_plus_bering   # a subset
set -euo pipefail
cd "$(dirname "$0")"
METHOD="${METHOD:-fast}"
CASES=("$@")
[ ${#CASES[@]} -gt 0 ] || CASES=(m1_shelf150 m2_plus_below200 m3_plus_bering m4_plus_coarse)
WALL=$([ "${METHOD}" = exact ] && echo 16:00:00 || echo 03:00:00)
for c in "${CASES[@]}"; do
    sbatch --job-name="ak_${c%%_*}_${METHOD}" --time="${WALL}" \
           --export=ALL,CASE="${c}",METHOD="${METHOD}" slurm_ak_mesh.sh
done
