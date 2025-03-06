#!/bin/bash
# source /home/davemfish/miniconda3/etc/profile.d/conda.sh
# mamba activate /usr/local/recreation-server/invest_3_15_0/invest/env
nohup python -u execute_recmodel_server_twitter.py \
-w /usr/local/recreation-server/invest_3_15_0/server > \
/usr/local/recreation-server/invest_3_15_0/server/log.txt 2>&1 &
