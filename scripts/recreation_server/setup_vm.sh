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
cd  invest_3_15_0
git clone https://github.com/davemfish/invest.git
cd invest
git checkout feature/REC-twitter-client
mamba create -p ./env python=3.12
mamba activate ./env
mamba install gdal pygeoprocessing numpy
pip install .

# Mount GCS Fuse
cd /usr/local/recreation-server/invest_3_15_0 && mkdir server && mkdir server/volume
gcsfuse --implicit-dirs -o ro natcap-recreation server/volume
# Listing all contents should build some indices and improve performance later
ls -R server/volume

# Refer to invest/scripts/recreation_server/readme.txt for instructions on
# starting the python processes.
