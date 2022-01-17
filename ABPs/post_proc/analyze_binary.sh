#!/bin/sh
#SBATCH -p general                          # partition to run on
#SBATCH -n 1                                # number of cores
#SBATCH -t 11-00:00                          # time (D-HH:MM)

# Command to increase memory allocated --mem=100g


#This is the path to hoomd
hoomd_path=$1
#This is the path to save the text files
txt_path=$2
#This is the path to save the pictures
pic_path=$3
#This is the path to save the videos
vid_path=$4
#This is the path to the analysis file
script_path=$5
#This is the file name to analyze
fname=$6
# This is the bin size (in simulation units)
bin=$7
# This is the step size (in number of time steps)
step=$8
# This is the analysis function to run
method=$9

echo $hoomd_path
echo $txt_path
echo $pic_path
echo $script_path
echo $fname
echo $bin
echo $step


if [ $hoomd_path == '/Users/nicklauersdorf/hoomd-blue/build' ]; then
    vars="$(python3 ${script_path}/get_parameters.py ${fname})"
fi

if [ $hoomd_path == "/nas/longleaf/home/njlauers/hoomd-blue/build" ]; then
    vars="$(python3.5 ${script_path}/get_parameters.py ${fname})"
fi

pass=()
for i in $vars
do
    # Put in array to unpack
    pass+=($i)
done

# This is activity (monodisperse)
pe=${pass[0]}
# This is A-type activity
pa=${pass[1]}
# This is B-type activity
pb=${pass[2]}
# This is fraction of A particles
xa=${pass[3]}
# This is epsilon
ep=${pass[4]}
# This is system density (phi)
phi=${pass[5]}
# This is if the system is initially a cluster (binary)
clust=${pass[6]}
# This is the timestep size (in Brownian time)
dtau=${pass[7]}
# This is the system size
pNum=${pass[8]}

echo 'fname'
echo $fname
echo 'pe'
echo $pe
echo 'pa'
echo $pa
echo 'pb'
echo $pb
echo 'xa'
echo $xa
echo 'ep'
echo $ep
echo 'phi'
echo $phi
echo 'dtau'
echo $dtau
echo $xa

if (( ${xa%.*} == 0 )); then
    result=$(echo "100*$xa" | bc )
    declare -i xa2=0
    xa2=${result%.*}
else
    xa2=${xa%.*}
fi

echo $xa2

declare -i pa2=0
if (( ${pe%%.*} > ${pa%%.*} )); then
    pa2=${pe%%.*}
else
    pa2=${pa%%.*}
fi

if [ "$method" = "interface_pressure_com" ]; then
    python3 $script_path/align_pressure_CoM.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path
elif [ "$method" = "interface_pressure_surface" ]; then
    python3 $script_path/align_pressure_updated.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path
elif [ "$method" = "bulk_pressure_total" ]; then
    python3 $script_path/interparticle_pressure_true2.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path
elif [ "$method" = "bulk_pressure_phases" ]; then
    python3 $script_path/interpart_press_updated.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path
elif [ "$method" = "lattice_spacing" ]; then
    python3 $script_path/lattice_spacing.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path
    
    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}

    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"lat_map_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"lat_map_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"lat_map_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"lat_histo_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"lat_histo_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"lat_histo_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"interface_acc_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"interface_acc_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"interface_acc_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "number_density" ]; then
    python3 $script_path/full_density_analysis_binary_updates.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}

    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"num_dens_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"num_dens_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"num_dens_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"num_densB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"num_densB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"num_densB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"num_densA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"num_densA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"num_densA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"num_densDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"num_densDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
      rm -rf "$pic_path"num_densDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"fa_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"fa_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"fa_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "alignment" ]; then
    python3 $script_path/full_alignment_analysis_binary_updates.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"align_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"align_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"align_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum100000_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"alignB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"alignB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"alignB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"alignA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"alignA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"alignA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"alignDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"alignDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"alignDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"aligngrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"aligngrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"aligngrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum100000_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"alignBgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"alignBgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"alignBgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"alignAgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"alignAgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"alignAgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"alignDifgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"alignDifgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"alignDifgrad_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*


elif [ "$method" = "simulation_frames" ]; then
    python3 $script_path/Sim_frame_image_com_updated.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"sim_frame_pa${pa2}_pb${pb}_xa${xa}_eps${eps}_phi${phi}_pNum${pNum}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"sim_frame_pa${pa2}_pb${pb}_xa${xa}_eps${eps}_phi${phi}_pNum${pNum}.mp4

    rm -rf "$pic_path"sim_frame_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "velocity" ]; then
    python3 $script_path/full_velocity_analysis_binary.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path
    
    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"velocity_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"velocity_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"velocity_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"velocityB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"velocityB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"velocityB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"velocityA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"velocityA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
   rm -rf "$pic_path"velocityA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"velocityDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"velocityDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"velocityDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
      
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"divergence_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"divergence_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"divergence_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"curl_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"curl_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"curl_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"divergenceA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"divergenceA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"divergenceA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"curlA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"curlA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"curlA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"divergenceB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"divergenceB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"divergenceB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"curlB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"curlB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"curlB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"divergenceDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"divergenceDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
     
    rm -rf "$pic_path"divergenceDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"curlDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"curlDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
      
    rm -rf "$pic_path"curlDif_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "neighbors" ]; then
    python3 $script_path/mesh_nearest_neighbors.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"defects_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"defects_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"defects_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"defectsB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"defectsB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"defectsB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"defectsA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"defectsA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"defectsA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "individual_velocities" ]; then
    python3 $script_path/single_particle_velocities.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"single_v_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"single_v_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"single_v_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"single_vA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"single_vA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"single_vA_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"single_vB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"single_vB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"single_vB_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "pressure_map" ]; then
    python3 $script_path/interparticle_pressure_map.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"interpart_press_cluster_pe${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"interpart_press_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4

    rm -rf "$pic_path"interpart_press_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "bulk_trace" ]; then
    python3 $script_path/bulk_trace.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}
    
    ffmpeg -start_number 1 -framerate 10 -i "$pic_path"trace_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"trace_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"trace_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
elif [ "$method" = "order_parameters" ]; then
    python3 $script_path/hexatic_order_parameter.py $fname $pa2 $pb $xa2 $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

    pe=${pe%%.*}
    pa=${pa%%.*}
    pb=${pb%.*}
    eps=${ep}
    phi=${phi%%.*}
    pNum=${pNum%.*}

    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"hexatic_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"hexatic_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"hexatic_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"translational_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"translational_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"translational_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
     
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"steinhardt_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"steinhardt_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"steinhardt_order_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*
    
    ffmpeg -start_number 0 -framerate 10 -i "$pic_path"relative_angle_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_%04d.png\
     -vcodec libx264 -s 1600x1200 -pix_fmt yuv420p -threads 1\
     "$vid_path"relative_angle_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}.mp4
    
    rm -rf "$pic_path"relative_angle_pa${pa2}_pb${pb}_xa${xa2}_eps${eps}_phi${phi}_pNum${pNum}_bin${bin}_time${step}_frame_*

else
    echo "specified analysis routine not found"
fi

exit 0
