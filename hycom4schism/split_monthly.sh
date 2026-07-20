#!/bin/bash

# Split FINAL merged HYCOM files into monthly chunks.
# Each output file covers from the 1st of a month to the 1st of the next month
# (inclusive on both ends), e.g. 20180701-20180801, 20180801-20180901, etc.
#
# Usage: bash split_monthly.sh <data_dir>
#   data_dir: directory containing FINAL_TS_*.nc, FINAL_SSH_*.nc, FINAL_UV_*.nc
#             (defaults to current directory if not provided)
#
# Outputs are written to the same directory as the input files.
# Example output names: TS_20180701_20180801.nc, SSH_20180801_20180901.nc, etc.
#
# NOTE: Requires NCO (ncks) with DAP/CF time support.
# Run from the DTN node (hercules-dtn) or any node with the hycom_env conda environment.

DATA_DIR="${1:-.}"

# Verify directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Directory '$DATA_DIR' not found."
    exit 1
fi

# Compute hours since 2000-01-01 UTC for a given date string (YYYY-MM-DD)
hours_since_2000() {
    echo $(( ( $(TZ=UTC date -d "$1" +%s) - $(TZ=UTC date -d "2000-01-01" +%s) ) / 3600 ))
}

# Process one variable
# Args: label (TS/SSH/UV), input_file
split_variable() {
    local LABEL="$1"
    local INFILE="$2"

    if [ ! -f "$INFILE" ]; then
        echo "WARNING: $INFILE not found, skipping $LABEL."
        return
    fi

    echo "=== Splitting $LABEL: $INFILE ==="

    # Determine the time range from the filename (FINAL_<LABEL>_YYYYMMDD_YYYYMMDD.nc)
    local BASENAME
    BASENAME=$(basename "$INFILE" .nc)
    # Extract start and end dates embedded in filename
    local DATES
    DATES=$(echo "$BASENAME" | grep -oE '[0-9]{8}_[0-9]{8}')
    local FILE_START FILE_END
    FILE_START=$(echo "$DATES" | cut -d_ -f1)
    FILE_END=$(echo "$DATES"   | cut -d_ -f2)

    # Convert to YYYY-MM-DD
    local START_DATE END_DATE
    START_DATE="${FILE_START:0:4}-${FILE_START:4:2}-${FILE_START:6:2}"
    END_DATE="${FILE_END:0:4}-${FILE_END:4:2}-${FILE_END:6:2}"

    echo "  File covers $START_DATE to $END_DATE"

    # Build list of month-start dates within the file range
    # We generate dates for the 1st of each month from the file start month
    # through the file end month + 1 (to get the closing boundary).
    local FIRST_MONTH LAST_MONTH
    FIRST_MONTH=$(TZ=UTC date -d "$START_DATE" +%Y-%m-01)
    # One month after the end date's month
    LAST_MONTH=$(TZ=UTC date -d "$(TZ=UTC date -d "$END_DATE" +%Y-%m-01) + 1 month" +%Y-%m-%d)

    # Collect all month boundaries
    local BOUNDARIES=()
    local cur
    cur=$(TZ=UTC date -d "$FIRST_MONTH" +%Y-%m-%d)
    while [ "$(TZ=UTC date -d "$cur" +%Y%m%d)" -le "$(TZ=UTC date -d "$LAST_MONTH" +%Y%m%d)" ]; do
        BOUNDARIES+=("$cur")
        cur=$(TZ=UTC date -d "$cur + 1 month" +%Y-%m-%d)
    done

    # Extract each monthly chunk: [BOUNDARIES[i], BOUNDARIES[i+1]] inclusive
    local i
    for (( i=0; i<${#BOUNDARIES[@]}-1; i++ )); do
        local T0="${BOUNDARIES[$i]}"
        local T1="${BOUNDARIES[$((i+1))]}"

        local H0 H1
        H0=$(hours_since_2000 "$T0")
        H1=$(hours_since_2000 "$T1")

        local FLAT0 FLAT1
        FLAT0=$(TZ=UTC date -d "$T0" +%Y%m%d)
        FLAT1=$(TZ=UTC date -d "$T1" +%Y%m%d)

        local OUTFILE="${DATA_DIR}/${LABEL}_${FLAT0}_${FLAT1}.nc"

        echo "  Extracting $T0 to $T1 -> $(basename "$OUTFILE")"
        ncks -O -d time,${H0}.,${H1}. "$INFILE" "$OUTFILE"

        if [ $? -ne 0 ]; then
            echo "  ERROR: ncks failed for $T0 to $T1"
        fi
    done

    echo "  Done with $LABEL."
}

# Find and process each FINAL file
TS_FILE=$(ls "${DATA_DIR}"/FINAL_TS_*.nc 2>/dev/null | head -1)
SSH_FILE=$(ls "${DATA_DIR}"/FINAL_SSH_*.nc 2>/dev/null | head -1)
UV_FILE=$(ls "${DATA_DIR}"/FINAL_UV_*.nc 2>/dev/null | head -1)

split_variable "TS"  "$TS_FILE"
split_variable "SSH" "$SSH_FILE"
split_variable "UV"  "$UV_FILE"

echo ""
echo "All done. Monthly files written to: $DATA_DIR"
