# Initial setup for the core invest models
# Assumes that micromamba is available
set -e  # Exit the script immediately if any subshell has a nonzero exit code.

CONFIG_PATH=$1
# curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba
# # export MAMBA_ROOT_PREFIX=/some/prefix  # optional, defaults to ~/micromamba
# eval "$(./bin/micromamba shell hook -s posix)"
# MICROMAMBA_PATH=$(realpath bin/micromamba)
# MICROMAMBA_PATH=/Users/emily/mambaforge/envs/plugin/bin/mamba
eval "$(micromamba shell hook --shell bash)"

# Create a conda env containing core invest and dependencies ##################
ENV_NAME=natcap_core
# micromamba create --yes --name $ENV_NAME pip gdal natcap.invest
# echo "created env"
micromamba activate $ENV_NAME
echo "activated env"
# cd /Users/emily/invest
# pip install .

# Write core metadata to the workbench's config.json ##########################
ENV_PATH=$(mamba info | grep 'active env location' | awk 'NF>1{print $NF}')
echo $ENV_PATH
python -c "
import importlib
import json
from natcap.invest.models import model_id_to_pyname

with open('$CONFIG_PATH') as f:
    config = json.load(f)

config['micromamba_path'] = '$MICROMAMBA_PATH'

if 'models' not in config:
    config['models'] = {}

for model_id, pyname in model_id_to_pyname.items():
    module = importlib.import_module(pyname)
    config['models'][model_id] = {
        'model_name': module.MODEL_SPEC['model_name'],
        'type': 'core',
        'env': '$ENV_PATH'
    }
    if 'sampledata' in module.MODEL_SPEC['ui_spec']:
        config['models'][model_id]['sampledata'] = module.MODEL_SPEC['ui_spec']['sampledata']


print(config)

with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=4)
"
echo "done"

