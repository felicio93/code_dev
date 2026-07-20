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

echo "Starting daily downloads and processing for TS in $OUT_DIR..."

while [ "$current_date" -le "$end_date_sec" ]; do
    
    date_hyphen=$(date -d "$current_date" +%Y-%m-%d)
    date_flat=$current_date
    echo "Processing $date_hyphen..."
    
    # 3. Date-Based Epoch Mapping (GOFS 3.1 experiment sequence)
    # Sequence: expt_57.2 -> expt_92.8 -> expt_57.7 -> expt_92.9 -> expt_93.0
    # Source: https://www.hycom.org/dataserver/gofs-3pt1/analysis
    #
    # GLBv0.08 epochs (pre-2018): single aggregated file, no variable sub-paths.
    # GLBv0.08/expt_93.0: 2018-01-01 to 2018-12-03, variable sub-paths (/ts3z, etc.)
    # GLBy0.08/expt_93.0: 2018-12-04 to present (finer lat resolution), variable sub-paths
    #
    # IMPORTANT: Do NOT use yearly sub-paths (e.g. /ts3z/2018) -- they contain only
    # partial date ranges and the time filter will silently return wrong data.
    # Always use the full variable-level aggregation (e.g. /ts3z).
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
        URL="https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z"
    else
        URL="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z"
    fi
    BASE_URL="${URL}"
    SUCCESS=0

    # ATTEMPT A: Try Modern Positive Longitudes (0 to 360 grid)
    ncks -O -d lon,150.,225. -d lat,45.,78. -d time,"$date_hyphen" \
         -v water_temp,salinity "$URL" ts3z_${date_flat}.nc 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "  -> Found data in $URL (0-360 grid)"
        SUCCESS=1
    fi

    if [ $SUCCESS -eq 0 ]; then
        # ATTEMPT B: Try Legacy Negative Longitudes (-180 to 180 grid)
        ncks -O -d lat,45.,78. -d time,"$date_hyphen" \
             -v water_temp,salinity "$URL" temp_global_ts.nc 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "  -> Found data in $URL (-180 to 180 grid). Rectifying longitudes..."
            ncap2 -O -s 'where(lon<0) lon=lon+360' temp_global_ts.nc temp_pos_ts.nc
            ncks -O -d lon,150.,225. temp_pos_ts.nc ts3z_${date_flat}.nc
            SUCCESS=1
            rm temp_global_ts.nc temp_pos_ts.nc
        fi
    fi
    
    if [ $SUCCESS -eq 0 ]; then
        echo "ERROR: Could not find data for $date_hyphen in epoch $BASE_URL!"
        exit 1
    fi

    # Unpack the data safely before potential temperature math
    ncpdq -O -U ts3z_${date_flat}.nc ts3z_${date_flat}.nc

    # Calculate Potential Temperature using CDO
    # NOTE: cdo adipot resets the time axis to the last value in the source
    # aggregation. We correct this by computing the exact time value for this
    # day (hours since 2000-01-01) and overwriting the time variable directly.
    cdo adipot ts3z_${date_flat}.nc ts_test1.nc
    HOURS_SINCE=$(( ( $(TZ=UTC date -d "$date_hyphen" +%s) - $(TZ=UTC date -d "2000-01-01" +%s) ) / 3600 ))
    ncap2 -O -s "time(:)=${HOURS_SINCE}.0" ts_test1.nc ts_test1.nc

    # Cast data variables to 32-bit floats (keep time as double to preserve precision)
    ncap2 -O -s 'depth=float(depth); lat=float(lat); lon=float(lon); tho=float(tho); s=float(s);' ts_test1.nc ts_test2.nc

    # Rename dimensions and variables
    ncrename -O -d lon,xlon -d lat,ylat -v lon,xlon -v lat,ylat -v tho,temperature -v s,salinity ts_test2.nc

    # Update temperature units
    ncatted -O -a units,temperature,m,c,degC ts_test2.nc

    # Make time a record dimension for concatenation
    ncks -O --mk_rec_dmn time ts_test2.nc TS_${date_flat}.nc

    # Clean up
    rm ts3z_${date_flat}.nc ts_test1.nc ts_test2.nc
    current_date=$(date -d "$current_date + 1 day" +%Y%m%d)
done

echo "Concatenating TS files..."
ncrcat -O TS_*.nc FINAL_TS_20180701_20181130.nc
echo "Done! Final file saved to $OUT_DIR/FINAL_TS_20180701_20181130.nc"
