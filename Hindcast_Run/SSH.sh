#!/bin/bash

# 1. Define the output directory and create it
OUT_DIR="/work2/noaa/nosofs/felicioc/OCSMesh_Paper/step_06_ManualPrePro/hycom_20161101_20180110"
mkdir -p "$OUT_DIR"

# Move into the directory so all temporary files stay there
cd "$OUT_DIR" || exit

# 2. Define the start and end dates
start_date="2016-11-01"
end_date="2018-01-10"

current_date=$(date -d "$start_date" +%Y%m%d)
end_date_sec=$(date -d "$end_date" +%Y%m%d)

# 3. List of HYCOM experiments (Included fallbacks without /ssh just in case)
HYCOM_URLS=(
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2/ssh"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8/ssh"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7/ssh"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9/ssh"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9"
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh"
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
)

echo "Starting daily downloads and processing for SSH in $OUT_DIR..."

while [ "$current_date" -le "$end_date_sec" ]; do
    
    date_hyphen=$(date -d "$current_date" +%Y-%m-%d)
    date_flat=$current_date
    
    echo "Processing $date_hyphen..."
    
    # 4. Try each URL and both Longitude formats until one works
    SUCCESS=0
    for URL in "${HYCOM_URLS[@]}"; do
        
        # ATTEMPT A: Try Modern Positive Longitudes (0 to 360)
        ncks -O -d lon,260.,307.5 -d lat,7.0,53. -d time,"$date_hyphen" \
             -v surf_el "$URL" ssh_${date_flat}.nc 2>/dev/null
             
        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (0-360 grid)"
            SUCCESS=1
            break
        fi

        # ATTEMPT B: Try Legacy Negative Longitudes (-180 to 180)
        ncks -O -d lon,-100.,-52.5 -d lat,7.0,53. -d time,"$date_hyphen" \
             -v surf_el "$URL" ssh_${date_flat}.nc 2>/dev/null
             
        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (-180 to 180 grid)"
            
            # Convert the negative longitudes to positive so the final file matches
            ncap2 -O -s 'where(lon<0) lon=lon+360' ssh_${date_flat}.nc ssh_${date_flat}.nc
            
            SUCCESS=1
            break
        fi
    done
    
    if [ $SUCCESS -eq 0 ]; then
        echo "ERROR: Could not find data for $date_hyphen in any known HYCOM experiment!"
        exit 1
    fi

    # Step 2: Unpack the data
    ncpdq -O -U ssh_${date_flat}.nc test1.nc

    # Step 3: Cast all variables to 32-bit floats (Replaces cvtZ.nco)
    ncap2 -O -s 'time=float(time); lat=float(lat); lon=float(lon); surf_el=float(surf_el);' test1.nc test2.nc

    # Step 4: Rename lat/lon to ylat/xlon
    ncrename -O -d lon,xlon -d lat,ylat -v lon,xlon -v lat,ylat test2.nc

    # Step 5: Make time a record dimension for concatenation
    ncks -O --mk_rec_dmn time test2.nc SSH_${date_flat}.nc

    # Clean up intermediate files for this day
    rm ssh_${date_flat}.nc test1.nc test2.nc

    # Advance the date by 1 day
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Daily processing complete. Concatenating files..."

# Step 6: Concatenate all daily files into the final file
ncrcat -O SSH_*.nc FINAL_SSH_20161101_20180110.nc

echo "Done! Final file saved to $OUT_DIR/FINAL_SSH_20161101_20180110.nc"

# Optional: Uncomment to clean up the hundreds of daily files once merged
rm SSH_201*.nc
