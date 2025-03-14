#!/bin/bash
# source /home/davemfish/miniconda3/etc/profile.d/conda.sh
# mamba activate /usr/local/recreation-server/invest_3_15_0/invest/env
nohup python -u /usr/local/recreation-server/invest_3_15_0/invest/scripts/recreation_server/execute_recmodel_server.py \
-w /usr/local/recreation-server/invest_3_15_0/server > \
/usr/local/recreation-server/invest_3_15_0/server/log.txt 2>&1 &
