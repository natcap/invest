# This file is meant to be a record of all the steps needed
# to setup a GCS VM running the recmodel_server.py
# It's adviseable to run these commands individually rather
# than execute this script, since that has not been tested.
# And because some of these steps might be interactive.

# Install GCS Fuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse git gcc g++

cd ~
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
mamba init && source .bashrc

cd /usr/local/recreation-server
mkdir invest_3_15_0
cd invest_3_15_0
git clone https://github.com/natcap/invest.git
cd invest
git checkout 3.15.0
mamba create -p ./env python=3.12
mamba activate ./env
mamba install "gdal>=3.4.2" "pygeoprocessing>=2.4.6" "numpy>=2.0"
pip install .

# Mount GCS Fuse
cd /usr/local/recreation-server/invest_3_15_0 && mkdir server && mkdir server/volume
gcsfuse --implicit-dirs -o ro natcap-recreation server/volume
# Listing all contents should build some indices and improve performance later
ls -R server/volume

# Start the recmodel_server process
# Review or update invest/scripts/recreation_server/execute_recmodel_server.py
# Then launch the Ptyhon process in the background:
chmod 755 invest/scripts/recreation_server/launch_recserver.sh
./invest/scripts/recreation_server/launch_recserver.sh

# Observe the server logfile to see if started up:
tail -f server/log.txt

# After setting up a new VM or new server cache,
# initialize a cron job that periodically clears out the cache.
# Copy the script so that updates to the invest repo don't clobber it
# Edit the paths to workspaces referenced in the script if needed.
cp invest/scripts/recreation_server/cron_find_rm_cached_workspaces.sh server/cron_find_rm_cached_workspaces.sh
chmod 755 server/cron_find_rm_cached_workspaces.sh
sudo crontab -e
# Enter: @daily /usr/local/recreation-server/invest_3_15_0/server/cron_find_rm_cached_workspaces.sh
