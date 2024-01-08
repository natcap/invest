
# https://github.com/emlys/demo-invest-plugin.git
set -e  # Exit the script immediately if any subshell has a nonzero exit code.

PLUGIN_URL=$1
DOWNLOAD_DIR=$2
CONFIG_PATH=$3

echo $PLUGIN_URL
echo $DOWNLOAD_DIR
echo "config path: $CONFIG_PATH"

mkdir -p "$DOWNLOAD_DIR"

PLUGIN_REPO_NAME=$(basename $PLUGIN_URL .git)
cd "$DOWNLOAD_DIR"
rm -rf $PLUGIN_REPO_NAME
git clone --quiet $PLUGIN_URL
ls

# pip install tomli
PLUGIN_NAME=$(python -c "

import tomli

with open('$DOWNLOAD_DIR/$PLUGIN_REPO_NAME/pyproject.toml', 'rb') as f:
    toml_dict = tomli.load(f)

print(toml_dict['project']['name'])
")
echo $PLUGIN_NAME

ENV_NAME=natcap_plugin_$PLUGIN_NAME

micromamba create --yes --name $ENV_NAME pip gdal natcap.invest
echo "created env"
micromamba activate $ENV_NAME
echo "activated env"

ENV_PATH=$(mamba info | grep 'active env location' | awk 'NF>1{print $NF}')
echo $ENV_PATH

pip install "git+$PLUGIN_URL"

python -c "
import json

with open('$CONFIG_PATH') as f:
    config = json.load(f)

config['models']['$PLUGIN_NAME'] = {
    'model_name': '$PLUGIN_NAME',
    'type': 'plugin',
    'source': '$PLUGIN_URL',
    'env': '$ENV_PATH'
}
print(config)

with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f)
"
echo "done"

