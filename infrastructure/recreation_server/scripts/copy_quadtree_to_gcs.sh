#!/bin/bash
#
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=hns,normal
#SBATCH --output=slurm-%j.%x.out

ml load system rclone
rclone copy --progress \
    $SCRATCH/twitter_quadtree/ \
    gcs-remote:natcap-recreation/twitter_quadtree/
