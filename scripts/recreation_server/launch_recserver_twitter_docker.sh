#!/bin/bash
# source /home/davemfish/miniconda3/etc/profile.d/conda.sh
# mamba activate /home/davemfish/twitter/invest/env/
# nohup python -u execute_recmodel_server_twitter.py > /home/davemfish/twitter/recmodel_server_log.txt 2>&1 &

SERVER_DIR=/home/dmf/projects/recreation/
DOCKER_DIR=/server_dir/

# -v %appdata%/gcloud:/home/mambauser/.config/gcloud
# -e GOOGLE_CLOUD_PROJECT='natcap-servers'

docker run --rm --cap-add SYS_ADMIN --device /dev/fuse \
-ti -v $SERVER_DIR:$DOCKER_DIR -v $(pwd):/workspace -w /workspace \
ghcr.io/davemfish/invest:exp.rec-twitter \
bash
# python execute_recmodel_server_twitter.py -w $DOCKER_DIR