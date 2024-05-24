# Install GCS Fuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse git gcc g++

cd ~
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source .bashrc

mkdir twitter && cd twitter
git clone https://github.com/davemfish/invest.git
cd invest
git checkout exp/REC-twitter
mamba create -p ./env python=3.11
mamba activate ./env
mamba install gdal pygeoprocessing numpy
pip install .

# Mount GCS Fuse