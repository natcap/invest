# Install a plugin
# Assumes that micromamba is available
set -ex  # Exit the script immediately if any subshell has a nonzero exit code.

PLUGIN_URL=$1  # git https url
DOWNLOAD_DIR=$2
CONFIG_PATH=$3

echo $PLUGIN_URL
echo $DOWNLOAD_DIR
echo $CONFIG_PATH

# Download the plugin from its remote git repo ################################
mkdir -p "$DOWNLOAD_DIR"
PLUGIN_REPO_NAME=$(basename $PLUGIN_URL .git)
cd "$DOWNLOAD_DIR"
rm -rf $PLUGIN_REPO_NAME
git clone --quiet $PLUGIN_URL

# Read in plugin metadata from the pyproject.toml #############################
# pip install tomli
PLUGIN_ID=$(python -c "
import tomli

with open('$DOWNLOAD_DIR/$PLUGIN_REPO_NAME/pyproject.toml', 'rb') as f:
    toml_dict = tomli.load(f)

print(toml_dict['tool']['natcap']['invest']['model_id'])
")
PLUGIN_NAME=$(python -c "
import tomli

with open('$DOWNLOAD_DIR/$PLUGIN_REPO_NAME/pyproject.toml', 'rb') as f:
    toml_dict = tomli.load(f)

print(toml_dict['tool']['natcap']['invest']['model_name'])
")
PLUGIN_PYNAME=$(python -c "
import tomli

with open('$DOWNLOAD_DIR/$PLUGIN_REPO_NAME/pyproject.toml', 'rb') as f:
    toml_dict = tomli.load(f)

print(toml_dict['tool']['natcap']['invest']['pyname'])
")

# Create a conda env containing the plugin and its dependencies ###############
ENV_NAME=natcap_plugin_$PLUGIN_ID
eval "$(micromamba shell hook --shell bash)"
micromamba create --yes --name $ENV_NAME pip gdal "python<3.12"
echo "created env"
micromamba activate $ENV_NAME
echo "activated env"
cd ~/invest
pip install .
pip install "git+$PLUGIN_URL"
echo "installed plugin"

# Write plugin metadata to the workbench's config.json ########################
ENV_PATH=$(micromamba info | grep 'env location' | awk 'NF>1{print $NF}')
micromamba info
echo $ENV_PATH
python -c "
import json

with open('$CONFIG_PATH') as f:
    config = json.load(f)

config['models']['$PLUGIN_ID'] = {
    'model_name': '$PLUGIN_NAME',
    'pyname': '$PLUGIN_PYNAME',
    'type': 'plugin',
    'source': '$PLUGIN_URL',
    'env': '$ENV_PATH'
}
print(config)

with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=4)
"
echo "done"

