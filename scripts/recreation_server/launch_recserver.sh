#!/bin/bash
# activate a python environment before calling this script:
# mamba activate /usr/local/recreation-server/invest_3_15_0/invest/env
# I think I had a hard time doing the activation within this script, for some reason.
nohup python -u /usr/local/recreation-server/invest_3_15_0/invest/scripts/recreation_server/execute_recmodel_server.py \
-w /usr/local/recreation-server/invest_3_15_0/server > \
/usr/local/recreation-server/invest_3_15_0/server/log.txt 2>&1 &
