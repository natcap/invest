# Initial setup for the core invest models
# Assumes that micromamba is available
set -e  # Exit the script immediately if any subshell has a nonzero exit code.

CONFIG_PATH=$1
INVEST_EXE=$2

invest list --json > tmp_models.json

# Write core metadata to the workbench's config.json ##########################
python -c "
import importlib
import json
from natcap.invest.models import model_id_to_pyname

with open('$CONFIG_PATH') as f:
    config = json.load(f)

config['micromamba_path'] = '$MICROMAMBA_PATH'

if 'models' not in config:
    config['models'] = {}

with open('tmp_models.json') as f:
    models = json.load(f)

for model_name, info in models.items():
    model_id = info['model_name']
    config['models'][model_id] = {
        'model_name': model_name,
        'type': 'core',
    }
    # if 'sampledata' in module.MODEL_SPEC['ui_spec']:
    #     config['models'][model_id]['sampledata'] = module.MODEL_SPEC['ui_spec']['sampledata']


print(config)

with open('$CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=4)
"
rm tmp_models.json
echo "done"

