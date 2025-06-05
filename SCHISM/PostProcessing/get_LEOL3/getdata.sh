
#$1 is start date YYYYMMDD
#$2 is end date
#Usage:  ./getdata.sh 20180701 20180930

start=$1
end=$2

start=$(date -d $start +%Y%m%d)

end=$(date -d $end +%Y%m%d)



while [[ $start -le $end ]]
do

      iyr=$(date -d "$start" +%Y)
      iday=$(date -d "$start" +%j)

      if [ ! -f subset/leosst_${start}.nc ]
         then
         wget https://coastwatch.noaa.gov/pub/socd2/coastwatch/sst/ran/l3s/leo/daily/${iyr}/${iday}/${start}120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.81-v02.0-fv01.0.nc -O leosst_${start}.nc
         ./extract.sh ${start}
      fi
      #rm leosst*
   start=$(date -d"$start + 1 day" +"%Y%m%d")

done


