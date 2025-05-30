#!/bin/bash

indir="/work2/noaa/nos-surge/felicioc/BeringSea/WW3/P03/spec_files"
outdir="/work2/noaa/nos-surge/felicioc/BeringSea/WW3/P03"
tmpdir="$outdir/tmp"
mkdir -p "$tmpdir"

# extract station id from file name
station_ids=$(ls $indir/ww3.*_spec.nc | sed -E 's/.*ww3\.([0-9]+)_[0-9]{6}_spec\.nc/\1/' | sort -u)

echo "Processing ${#station_ids[@]} stations..."

for station in $station_ids; do
    # list files for the station sorted by date
    files=$(ls $indir/ww3.${station}_*_spec.nc | sort)
    # concat over time
    ncrcat $files "$tmpdir/${station}_time_concat.nc"
    # consistent station dim
    ncecat -O -u station "$tmpdir/${station}_time_concat.nc" "$tmpdir/${station}_station_dim.nc"
done

# concat over station
ncrcat "$tmpdir"/*_station_dim.nc "$outdir/ww3_all_station.nc"

rm -r "$tmpdir"