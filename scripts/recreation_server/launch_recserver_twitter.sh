#!/bin/bash
# source /home/davemfish/miniconda3/etc/profile.d/conda.sh
# mamba activate /home/davemfish/twitter/invest/env/
nohup python -u execute_recmodel_server_twitter.py -w ~/server > ~/server/recmodel_server_log.txt 2>&1 &
