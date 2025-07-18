#!/bin/bash -xv
# funcion para sacar el numero de pasos temporales total. Necesita CDO
function tsteps() {
	fileval=$1
	if test ! -f ${fileval}; then
		echo "ERROR -- ERROR -- ERROR: file does not exist!"
		exit
	else
		cdo ntime ${fileval} >& out.out
		egrep -o "^[0-9]*" out.out
		rm out.out
	fi
}
# ACA SE DEFINE EL NOMBRE DE LA VARIABLE

# array de variables 
variables_array=('temperature' 'geopotential' 'relative_humidity')
variables_array=('specific_humidity')

# if all levels just pass 'all'
levels_array=(1 2 3 5 7 10 20 30 50 70 100 125 150 175 200 225 250 300 350 400 450 500 550 600 650 700 750 775 800 825 850 875 900 925 950 975 1000)
levels_array=('all')

init_year=2015
end_year=2023

months=(01 02 03 04 05 06 07 08 09 10 11 12)

days=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

# bounding box (For whole of Germany: 46-57 N 4-17 E) (For whole of Turkey: 33-45 N 24-47)
north=57
south=46
west=4
east=17

for variable in "${variables_array[@]}"
do

if [ -d "${variable}" ]; then
	cd ${variable}
else 
	mkdir ${variable}
	cd ${variable}
fi

for ((iy=$init_year;iy<=$end_year;iy++))
do

    for im in "${months[@]}"
    do

    for id in "${days[@]}"
    do

    for lvl in "${levels_array[@]}"
    do

        if [ ! -e ${variable}_${lvl}_${iy}-${im}-${id}.nc ]; then
            python ../ERA5_download_by_year_Juli.py ${iy} ${im} ${id} ${variable} ${lvl} ${north} ${south} ${west} ${east}
        fi
        if [ -e ${variable}_${lvl}_${iy}-${im}-${id}.nc ]; then
            ntime=$(tsteps ${variable}_${lvl}_${iy}-${im}-${id}.nc)
            if [ "${ntime}" -eq "8760" ] || [ "${ntime}" -eq "8784" ] || [ "${ntime}" -eq "12" ] || [ "${ntime}" -eq "744" ] || [ "${ntime}" -eq "720" ] || [ "${ntime}" -eq "696" ] || [ "${ntime}" -eq "672" ] || [ "${ntime}" -eq "24" ]; then
                echo "---------------------------------- ${iy}-${im}-${id} ok"


            else
                echo "ERROR -- ERROR -- ERROR: los archivos descargados no estan completos! verificar la cantidad de pasos temporales"
    #			exit
            fi
        else
            echo "ERROR -- ERROR -- ERROR: No se realizo la descarga del archivo a partir del script de Python"
    #		exit
        fi
    done

    done

done

done 

cd ..

done
