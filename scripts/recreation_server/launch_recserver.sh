#!/bin/bash
source /home/davemfish/miniconda3/etc/profile.d/conda.sh
conda activate ./invest/env/
nohup python -u execute_recmodel_server.py > nohup_recmodel_server.txt 2>&1 &
