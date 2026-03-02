#!/bin/bash

# 1. Define output directory
OUT_DIR="/work2/noaa/nosofs/felicioc/OCSMesh_Paper/step_06_ManualPrePro/hycom_20161101_20180110"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit

# 2. Define dates
start_date="2016-11-01"
end_date="2018-01-10"

current_date=$(date -d "$start_date" +%Y%m%d)
end_date_sec=$(date -d "$end_date" +%Y%m%d)

# 3. HYCOM URL Fallbacks
HYCOM_URLS=(
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2/uv3z"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8/uv3z"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7/uv3z"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9/uv3z"
    "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9"
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
)

echo "Starting daily downloads and processing for UV in $OUT_DIR..."

while [ "$current_date" -le "$end_date_sec" ]; do
    
    date_hyphen=$(date -d "$current_date" +%Y-%m-%d)
    date_flat=$current_date
    echo "Processing $date_hyphen..."
    
    SUCCESS=0
    for URL in "${HYCOM_URLS[@]}"; do
        
        # ATTEMPT A: Try Modern Positive Longitudes (0 to 360)
        ncks -O -d lon,260.,307.5 -d lat,7.0,53. -d time,"$date_hyphen" \
             -v water_u,water_v "$URL" uv3z_${date_flat}.nc 2>/dev/null
             
        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (0-360 grid)"
            SUCCESS=1
            break
        fi

        # ATTEMPT B: Try Legacy Negative Longitudes (-180 to 180)
        ncks -O -d lon,-100.,-52.5 -d lat,7.0,53. -d time,"$date_hyphen" \
             -v water_u,water_v "$URL" uv3z_${date_flat}.nc 2>/dev/null
             
        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (-180 to 180 grid)"
            # Convert negative longitudes to positive
            ncap2 -O -s 'where(lon<0) lon=lon+360' uv3z_${date_flat}.nc uv3z_${date_flat}.nc
            SUCCESS=1
            break
        fi
    done
    
    if [ $SUCCESS -eq 0 ]; then
        echo "ERROR: Could not find data for $date_hyphen!"
        exit 1
    fi

    # Unpack the data
    ncpdq -O -U uv3z_${date_flat}.nc test1.nc

    # Cast ALL variables (including depth) to 32-bit floats
    ncap2 -O -s 'time=float(time); depth=float(depth); lat=float(lat); lon=float(lon); water_u=float(water_u); water_v=float(water_v);' test1.nc test2.nc

    # Rename lat/lon to ylat/xlon
    ncrename -O -d lon,xlon -d lat,ylat -v lon,xlon -v lat,ylat test2.nc

    # Make time a record dimension for concatenation
    ncks -O --mk_rec_dmn time test2.nc UV_${date_flat}.nc

    # Clean up
    rm uv3z_${date_flat}.nc test1.nc test2.nc
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Concatenating UV files..."
ncrcat -O UV_*.nc FINAL_UV_20161101_20180110.nc
echo "Done! Final file saved to $OUT_DIR/FINAL_UV_20161101_20180110.nc"
rm UV_201*
