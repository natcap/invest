#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH --partition normal,hns
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000M
#SBATCH --nodes 1
# Define how long the job will run d-hh:mm:ss
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
# Get email notification when job finishes or fails
#SBATCH --mail-user=dfisher5@stanford.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -J build_quadtree
#SBATCH -o build_quadtree
# ----------------Load Modules--------------------
# ----------------Commands------------------------


CONTAINER=ghcr.io/davemfish/invest:exp.rec-twitter
TWEETS_DIR=/scratch/users/woodsp/invest/csv
find $TWEETS_DIR -name '*.csv' > ~/projects/rec-twitter/tweets_full_list.txt

# invest repo already cloned into ~/invest
cd ~/invest
git checkout exp/REC-twitter

set -x  # Be eXplicit about what's happening.
FAILED=0
singularity run \
    docker://$CONTAINER python scripts/recreation_server/build_twitter_quadtree.py \
    --csv_file_list=$HOME/projects/rec-twitter/tweets_full_list.txt \
    --workspace=$SCRATCH/twitter_quadtree \
    --output_filename=global_twitter_qt.pickle \
    --n_cores=4  # match --cpus-per-task value
