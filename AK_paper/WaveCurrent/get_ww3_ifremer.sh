 # input argument is yyyy mm                                                    
 YEAR=$1
 MNTH=$2
 vars=( "fp" "dir" "hs" "t02" "spr" )
 for i in "${vars[@]}"
 do
 echo "getting $i for ${YEAR} ${MNTH}"
 wget -q --progress=dot "ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/GLOBAL/${YEA\
R}_ECMWF/$i/WW3-GLOB-30M_${YEAR}${MNTH}_$i.nc" .
 echo "------------------------------------------------------------------------\
---------------------------------"
 done
