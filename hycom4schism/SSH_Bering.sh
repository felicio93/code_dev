#!/bin/bash

# 1. Define the output directory and create it
OUT_DIR="/work2/noaa/nos-surge/felicioc/BeringSea/hotstart_yunfang/hycom_data/hycom_20180701_20181130"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit

# 2. Define the start and end dates
start_date="2018-07-01"
end_date="2018-11-30"

current_date=$(date -d "$start_date" +%Y%m%d)
end_date_sec=$(date -d "$end_date" +%Y%m%d)

echo "Starting daily downloads and processing for SSH in $OUT_DIR..."

while [ "$current_date" -le "$end_date_sec" ]; do
    
    date_hyphen=$(date -d "$current_date" +%Y-%m-%d)
    date_flat=$current_date
    echo "Processing $date_hyphen..."
    
    # 3. Date-Based Epoch Mapping (GOFS 3.1 experiment sequence)
    # Sequence: expt_57.2 -> expt_92.8 -> expt_57.7 -> expt_92.9 -> expt_93.0
    # Source: https://www.hycom.org/dataserver/gofs-3pt1/analysis
    #
    # GLBv0.08 epochs (pre-2018): single aggregated file, no variable sub-paths.
    # GLBv0.08/expt_93.0: 2018-01-01 to 2018-12-03, variable sub-paths (/ssh, etc.)
    # GLBy0.08/expt_93.0: 2018-12-04 to present (finer lat resolution), variable sub-paths
    #
    # IMPORTANT: Do NOT use yearly sub-paths (e.g. /ssh/2018) -- they contain only
    # partial date ranges and the time filter will silently return wrong data.
    # Always use the full variable-level aggregation (e.g. /ssh).
    #
    # NOTE: Run these scripts from the DTN node (hercules-dtn) which has external internet access.
    if [ "$date_flat" -le 20170131 ]; then
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2"
    elif [ "$date_flat" -le 20170531 ]; then
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8"
    elif [ "$date_flat" -le 20170930 ]; then
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7"
    elif [ "$date_flat" -le 20171231 ]; then
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9"
    elif [ "$date_flat" -le 20181203 ]; then
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ssh"
    else
        URL="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh"
    fi
    BASE_URL="${URL}"
    SUCCESS=0

    # ATTEMPT A: Try Modern Positive Longitudes (0 to 360 grid)
    ncks -O -d lon,150.,225. -d lat,45.,78. -d time,"$date_hyphen" \
         -v surf_el "$URL" ssh_${date_flat}.nc 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "  -> Found data in $URL (0-360 grid)"
        SUCCESS=1
    fi

    if [ $SUCCESS -eq 0 ]; then
        # ATTEMPT B: Try Legacy Negative Longitudes (-180 to 180 grid)
        ncks -O -d lat,45.,78. -d time,"$date_hyphen" \
             -v surf_el "$URL" temp_global_ssh.nc 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (-180 to 180 grid). Rectifying longitudes..."
            ncap2 -O -s 'where(lon<0) lon=lon+360' temp_global_ssh.nc temp_pos_ssh.nc
            ncks -O -d lon,150.,225. temp_pos_ssh.nc ssh_${date_flat}.nc
            SUCCESS=1
            rm temp_global_ssh.nc temp_pos_ssh.nc
        fi
    fi
    
    if [ $SUCCESS -eq 0 ]; then
        echo "ERROR: Could not find data for $date_hyphen in epoch $BASE_URL!"
        exit 1
    fi

    # Step 2: Unpack the data
    ncpdq -O -U ssh_${date_flat}.nc ssh_test1.nc

    # Step 3: Fix time axis corrupted by ncpdq unpacking, then cast data variables to float32
    # ncpdq -U unpacks the packed time coordinate using the aggregation's scale/offset,
    # yielding the last timestamp in the source dataset rather than the requested day.
    # We recompute the correct value (hours since 2000-01-01) from the date string.
    HOURS_SINCE=$(( ( $(TZ=UTC date -d "$date_hyphen" +%s) - $(TZ=UTC date -d "2000-01-01" +%s) ) / 3600 ))
    ncap2 -O -s "time(:)=${HOURS_SINCE}.0" ssh_test1.nc ssh_test1.nc
    ncap2 -O -s 'lat=float(lat); lon=float(lon); surf_el=float(surf_el);' ssh_test1.nc ssh_test2.nc

    # Step 4: Rename lat/lon to ylat/xlon
    ncrename -O -d lon,xlon -d lat,ylat -v lon,xlon -v lat,ylat ssh_test2.nc

    # Step 5: Make time a record dimension for concatenation
    ncks -O --mk_rec_dmn time ssh_test2.nc SSH_${date_flat}.nc

    # Clean up intermediate files for this day
    rm ssh_${date_flat}.nc ssh_test1.nc ssh_test2.nc

    # Advance the date by 1 day
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Daily processing complete. Concatenating files..."

# Step 6: Concatenate all daily files into the final file
ncrcat -O SSH_*.nc FINAL_SSH_20180701_20181130.nc

echo "Done! Final file saved to $OUT_DIR/FINAL_SSH_20180701_20181130.nc"
