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

echo "Starting daily downloads and processing for UV in $OUT_DIR..."

while [ "$current_date" -le "$end_date_sec" ]; do
    
    date_hyphen=$(date -d "$current_date" +%Y-%m-%d)
    date_flat=$current_date
    echo "Processing $date_hyphen..."
    
    # 3. Strict Date-Based Epoch Mapping
    if [ "$date_flat" -le 20170131 ]; then
        BASE_URL="http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.2"
    elif [ "$date_flat" -le 20170531 ]; then
        BASE_URL="http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.8"
    elif [ "$date_flat" -le 20170930 ]; then
        BASE_URL="http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_57.7"
    elif [ "$date_flat" -le 20171231 ]; then
        BASE_URL="http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_92.9"
    else
        BASE_URL="http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
    fi
    
    # Check both the sub-folder and the base URL for this specific epoch
    EPOCH_URLS=("${BASE_URL}/uv3z" "${BASE_URL}")
    SUCCESS=0
    
    for URL in "${EPOCH_URLS[@]}"; do
        # ATTEMPT A: Try Modern Positive Longitudes
        ncks -O -d lon,260.,307.5 -d lat,7.0,53. -d time,"$date_hyphen" \
             -v water_u,water_v "$URL" uv3z_${date_flat}.nc
             
        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (0-360 grid)"
            SUCCESS=1
            break
        fi

        # ATTEMPT B: Try Legacy Negative Longitudes
        ncks -O -d lon,-100.,-52.5 -d lat,7.0,53. -d time,"$date_hyphen" \
             -v water_u,water_v "$URL" uv3z_${date_flat}.nc
             
        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (-180 to 180 grid)"
            # Convert negative longitudes to positive
            ncap2 -O -s 'where(lon<0) lon=lon+360' uv3z_${date_flat}.nc uv3z_${date_flat}.nc
            SUCCESS=1
            break
        fi
    done
    
    if [ $SUCCESS -eq 0 ]; then
        echo "ERROR: Could not find data for $date_hyphen in epoch $BASE_URL!"
        exit 1
    fi

    # Unpack the data
    ncpdq -O -U uv3z_${date_flat}.nc uv_test1.nc

    # Cast ALL variables (including depth) to 32-bit floats
    ncap2 -O -s 'time=float(time); depth=float(depth); lat=float(lat); lon=float(lon); water_u=float(water_u); water_v=float(water_v);' uv_test1.nc uv_test2.nc

    # Rename lat/lon to ylat/xlon
    ncrename -O -d lon,xlon -d lat,ylat -v lon,xlon -v lat,ylat uv_test2.nc

    # Make time a record dimension for concatenation
    ncks -O --mk_rec_dmn time uv_test2.nc UV_${date_flat}.nc

    # Clean up
    rm uv3z_${date_flat}.nc uv_test1.nc uv_test2.nc
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Concatenating UV files..."
ncrcat -O UV_*.nc FINAL_UV_20161101_20180110.nc
echo "Done! Final file saved to $OUT_DIR/FINAL_UV_20161101_20180110.nc"