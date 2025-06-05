ncap2 -O -v -S subset.nco leosst_$1.nc test.nc
ncatted -a _FillValue,sst,c,f,-32768. test.nc 
ncks -O -v lon2,lat2,sst test.nc test2.nc
ncrename -v lon2,lon -v lat2,lat test2.nc
mv test2.nc subset/leosst_$1.nc
rm test.nc
