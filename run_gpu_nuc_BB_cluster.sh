#!/bin/sh
#SBATCH --qos gpu_access                    # quality of service
#SBATCH --gres=gpu:1                        # I want one gpus
#SBATCH --partition=gpu                     # partition to run on
#SBATCH --nodes=1                           # run on one node
#SBATCH --time=11-00:00                     # time (D-HH:MM)


inFile=$1
hoomdPath=$2
gsdPath=$3
pef=$4
pes=$5
xf=$6
ep=$7
myFrame=${8}

echo $inFile
echo $
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo $10
stop

python3.5 $inFile $hoomdPath $gsdPath $pef $pes $xf $ep $myFrame# --mode=gpu