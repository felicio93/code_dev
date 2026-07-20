#!/bin/bash

# 1. Define output directory
OUT_DIR="/work2/noaa/nos-surge/felicioc/BeringSea/hotstart_yunfang/hycom_data/hycom_20180701_20181130"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit

# 2. Define dates
start_date="2018-07-01"
end_date="2018-11-30"

current_date=$(date -d "$start_date" +%Y%m%d)
end_date_sec=$(date -d "$end_date" +%Y%m%d)

echo "Starting daily downloads and processing for UV in $OUT_DIR..."

while [ "$current_date" -le "$end_date_sec" ]; do
    
    date_hyphen=$(date -d "$current_date" +%Y-%m-%d)
    date_flat=$current_date
    echo "Processing $date_hyphen..."
    
    # 3. Date-Based Epoch Mapping (GOFS 3.1 experiment sequence)
    # Sequence: expt_57.2 -> expt_92.8 -> expt_57.7 -> expt_92.9 -> expt_93.0
    # Source: https://www.hycom.org/dataserver/gofs-3pt1/analysis
    #
    # GLBv0.08 epochs (pre-2018): single aggregated file, no variable sub-paths.
    # GLBv0.08/expt_93.0: 2018-01-01 to 2018-12-03, variable sub-paths (/uv3z, etc.)
    # GLBy0.08/expt_93.0: 2018-12-04 to present (finer lat resolution), variable sub-paths
    #
    # IMPORTANT: Do NOT use yearly sub-paths (e.g. /uv3z/2018) -- they contain only
    # partial date ranges and the time filter will silently return wrong data.
    # Always use the full variable-level aggregation (e.g. /uv3z).
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
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z"
    else
        URL="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"
    fi
    BASE_URL="${URL}"
    SUCCESS=0

    # ATTEMPT A: Try Modern Positive Longitudes (0 to 360 grid)
    ncks -O -d lon,150.,225. -d lat,45.,78. -d time,"$date_hyphen" \
         -v water_u,water_v "$URL" uv3z_${date_flat}.nc 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "  -> Found data in $URL (0-360 grid)"
        SUCCESS=1
    fi

    if [ $SUCCESS -eq 0 ]; then
        # ATTEMPT B: Try Legacy Negative Longitudes (-180 to 180 grid)
        ncks -O -d lat,45.,78. -d time,"$date_hyphen" \
             -v water_u,water_v "$URL" temp_global_uv.nc 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (-180 to 180 grid). Rectifying longitudes..."
            ncap2 -O -s 'where(lon<0) lon=lon+360' temp_global_uv.nc temp_pos_uv.nc
            ncks -O -d lon,150.,225. temp_pos_uv.nc uv3z_${date_flat}.nc
            SUCCESS=1
            rm temp_global_uv.nc temp_pos_uv.nc
        fi
    fi
    
    if [ $SUCCESS -eq 0 ]; then
        echo "ERROR: Could not find data for $date_hyphen in epoch $BASE_URL!"
        exit 1
    fi

    # Unpack the data
    ncpdq -O -U uv3z_${date_flat}.nc uv_test1.nc

    # Fix time axis corrupted by ncpdq unpacking, then cast data variables to float32
    # ncpdq -U unpacks the packed time coordinate using the aggregation's scale/offset,
    # yielding the last timestamp in the source dataset rather than the requested day.
    # We recompute the correct value (hours since 2000-01-01) from the date string.
    HOURS_SINCE=$(( ( $(TZ=UTC date -d "$date_hyphen" +%s) - $(TZ=UTC date -d "2000-01-01" +%s) ) / 3600 ))
    ncap2 -O -s "time(:)=${HOURS_SINCE}.0" uv_test1.nc uv_test1.nc
    ncap2 -O -s 'depth=float(depth); lat=float(lat); lon=float(lon); water_u=float(water_u); water_v=float(water_v);' uv_test1.nc uv_test2.nc

    # Rename lat/lon to ylat/xlon
    ncrename -O -d lon,xlon -d lat,ylat -v lon,xlon -v lat,ylat uv_test2.nc

    # Make time a record dimension for concatenation
    ncks -O --mk_rec_dmn time uv_test2.nc UV_${date_flat}.nc

    # Clean up
    rm uv3z_${date_flat}.nc uv_test1.nc uv_test2.nc
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Concatenating UV files..."
ncrcat -O UV_*.nc FINAL_UV_20180701_20181130.nc
echo "Done! Final file saved to $OUT_DIR/FINAL_UV_20180701_20181130.nc"
