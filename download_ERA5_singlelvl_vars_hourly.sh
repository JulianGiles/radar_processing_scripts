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
variables_array=('2m_temperature' 'total_precipitation' 'boundary_layer_height' 'mean_evaporation_rate' 'mean_potential_evaporation_rate' 'mean_surface_downward_long_wave_radiation_flux' 'mean_surface_downward_short_wave_radiation_flux' 'mean_surface_latent_heat_flux' 'mean_surface_net_long_wave_radiation_flux' 'mean_surface_net_short_wave_radiation_flux' 'mean_surface_sensible_heat_flux' 'mean_vertically_integrated_moisture_divergence' 'vertical_integral_of_eastward_water_vapour_flux' 'vertical_integral_of_northward_water_vapour_flux' 'volumetric_soil_water_layer_1' 'volumetric_soil_water_layer_2' 'volumetric_soil_water_layer_3' 'volumetric_soil_water_layer_4' 'total_column_water_vapour' 'cloud_base_height' 'low_cloud_cover' 'total_cloud_cover' '10m_u_component_of_wind' '10m_v_component_of_wind' 'mean_sea_level_pressure' 'surface_pressure')


init_year=1979
end_year=2020

months=(01 02 03 04 05 06 07 08 09 10 11 12)

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
	
	if [ ! -e ${variable}_year_${iy}.nc ]; then
		python ../ERA5_download_singlelvl_by_year-month.py ${iy} ${im} ${variable} ${north} ${south} ${west} ${east}
	fi
	if [ -e ${variable}_year_${iy}.nc ]; then
		ntime=$(tsteps ${variable}_year_${iy}.nc)
		if [ "${ntime}" -eq "8760" ] || [ "${ntime}" -eq "8784" ] || [ "${ntime}" -eq "12" ] || [ "${ntime}" -eq "744" ] || [ "${ntime}" -eq "720" ] || [ "${ntime}" -eq "696" ] || [ "${ntime}" -eq "672" ]; then
			echo "---------------------------------- ${iy} ok"
			

		else
			echo "ERROR -- ERROR -- ERROR: los archivos descargados no estan completos! verificar la cantidad de pasos temporales"
#			exit
		fi
	else
		echo "ERROR -- ERROR -- ERROR: No se realizo la descarga del archivo a partir del script de Python"
#		exit
	fi
done

cd ..

done
