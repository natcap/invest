set -e  # Exit the script immediately if any subshell has a nonzero exit code.

CONFIG_PATH=$1
# curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba
# # export MAMBA_ROOT_PREFIX=/some/prefix  # optional, defaults to ~/micromamba
# eval "$(./bin/micromamba shell hook -s posix)"

# MICROMAMBA_PATH=$(realpath bin/micromamba)
MICROMAMBA_PATH=/Users/emily/mambaforge/envs/plugin/bin/mamba

ENV_NAME=natcap_core

eval "$(micromamba shell hook --shell bash)"

micromamba create --yes --name $ENV_NAME pip gdal natcap.invest
echo "created env"
micromamba activate $ENV_NAME
echo "activated env"

cd /Users/emily/invest
pip install .

ENV_PATH=$(mamba info | grep 'active env location' | awk 'NF>1{print $NF}')
echo $ENV_PATH

python -c "
import json
from natcap.invest.models import model_id_to_pyname

with open('$CONFIG_PATH') as f:
    config = json.load(f)

config['micromamba_path'] = '$MICROMAMBA_PATH'

if 'models' not in config:
    config['models'] = {}

for model_id, pyname in model_id_to_pyname.items():
    config['models'][model_id] = {
        'model_name': model_id,
        'type': 'core',
        'env': '$ENV_PATH'
    }
print(config)

with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f)
"
echo "done"

