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

echo poop
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

#python3.8 $script_path/
#python $script_path/nearest_neigh_small_array.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/nearest_neigh.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/heatmap.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/post_proc.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/pp_msd_perc_A.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/pp_msdten_perc_A.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/MCS.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/MCSten.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/voronoi.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/meshed_output.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/per_particle_output.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/gtar_pressure.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/phase_types.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/dense_CoM.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/number_densities.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/force_diff_sources.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/histogram_distance.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/histogram_output_txt.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/plotNumberDensities.py $pa $pb $xa
#python $script_path/pairCorrelationRelations.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/mesh_nearest_neighbor.py $pa $pb $xa $hoomd_path $gsd_path
#python3 $script_path/computeRDF.py $pe $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/heatmapType.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/extrinsic_txt.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/extrinsic_all_time.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/phi_extrinsic_all_time.py $pa $pb $xa $hoomd_path $gsd_path $ep $phi
#python $script_path/sim_to_txt_no_cutoff.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/convergence_analysis.py $pa $pb $xa $hoomd_path $gsd_path $ep $num
#python $script_path/extrinsic_all_time_back_compat.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/edge_detection.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/edge_detection_v2.py $pa $pb $xa $hoomd_path $gsd_path
#python $script_path/check_cluster_alg.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/diffHeatmapType.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/orientation_snapshots.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/binnedNetActivity.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/analyze_alpha.py $pa $pb $xa $hoomd_path $gsd_path $ep $al
#python $script_path/alpha_diameter_histogram.py $pa $pb $xa $hoomd_path $gsd_path $ep $al
#python /Users/kolbt/Desktop/compiled/whingdingdilly/art/voronoi_tessellation.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/mesh_nearest_neighbors.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/mesh_nearest_neighbors_periodic.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python3 $script_path/delta_spacial.py $pa $pb $xa $hoomd_path $gsd_path $ep $fname
#python $script_path/soft_nearest_neighbors.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python $script_path/compute_phase_area.py $pa $pb $xa $hoomd_path $gsd_path $ep
#python3 $script_path/computeMCS_threshold.py $fname $pe $pb $xa $ep $phi
#python3 $script_path/edge_distance.py $fname $pe $pb $xa $ep $phi
#python3 $script_path/edges_from_bins.py $fname $pe $pb $xa $ep $phi
#python3 $script_path/indiv_cluster_pressure.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/histogram-densities.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/interparticle_pressure.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/image_final_tstep.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/image_single_particle.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/sim_frames.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/sim_velocity.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/sim_orientation.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/full-video-analysis.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/radial_com_data.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/radial_density.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/alignment_analysis2.py $fname $pa $pb $xa $ep $phi $tau
#python3 $script_path/full_video_analysis_binary.py $fname $pa $pb $xa $ep $phi $tau
#python3 $script_path/bulk_trace.py $fname $pa $pb $xa $ep $phi $tau
#python3 $script_path/Sim_frame_image_com.py $fname $pa $pb $xa $ep $phi $tau
declare -i pa2=0
if (( ${pe%%.*} > ${pa%%.*} )); then
    pa2=${pe%%.*}
else
    pa2=${pa%%.*}
fi

python3 $script_path/interparticle_pressure_true2.py $fname $pa2 $pb $xa $ep $phi $dtau $bin $step $hoomd_path $txt_path $pic_path

#python3 $script_path/2activepartg_cluster.py $fname $pe $pb $xa $ep $phi $tau
#python3 $script_path/getRDF.py $pe $pb $xa $hoomd_path $gsd_path $ep
#python3 $script_path/untitled6.py $fname $pe $pb $xa $ep $phi $tau

##Nicks Video for Particle Motion
pe=${pe%%.*}
pa=${pa%%.*}
pb=${pb%.*}
xa=${xa%.*}
eps=${ep}
echo ${phi}
phi=${phi%%.*}
pNum=${pNum%.*}

exit 0
