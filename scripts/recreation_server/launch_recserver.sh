#!/bin/bash
# activate a python environment before calling this script:
# mamba activate /usr/local/recreation-server/invest_3_15_0/invest/env
# I think I had a hard time doing the activation within this script, for some reason.
export PYRO_LOGFILE=/usr/local/recreation-server/invest_3_15_0/server/pyro_invest319.log
export PYRO_LOGLEVEL=DEBUG
export PYRO_THREADPOOL_SIZE_MIN=8
export PYRO_DETAILED_TRACEBACK=true
nohup python -u /usr/local/recreation-server/invest_3_15_0/invest_3_19_0/scripts/recreation_server/execute_recmodel_server.py \
-w /usr/local/recreation-server/invest_3_15_0/server > \
/usr/local/recreation-server/invest_3_15_0/server/pyro_invest319.log 2>&1 &
