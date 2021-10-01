"""A Flask app with HTTP endpoints used by the InVEST Workbench."""
import collections
import importlib
import json
import logging

from osgeo import gdal
from flask import Flask
from flask import request
from natcap.invest import cli
from natcap.invest import datastack
from natcap.invest import install_language
from natcap.invest import spec_utils

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

# Lookup names to pass to `invest run` based on python module names
_UI_META = collections.namedtuple('UIMeta', ['run_name', 'human_name'])
MODULE_MODELRUN_MAP = {
    value.pyname: _UI_META(
        run_name=key,
        human_name=value.humanname)
    for key, value in cli._MODEL_UIS.items()}


def shutdown_server():
    """Shutdown the flask server."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/ready', methods=['GET'])
def get_is_ready():
    """Returns something simple to confirm the server is open."""
    return 'Flask ready'


@app.route('/shutdown', methods=['GET'])
def shutdown():
    """A request to this endpoint shuts down the server."""
    shutdown_server()
    return 'Flask server shutting down...'


@app.route('/models', methods=['GET'])
def get_invest_models():
    """Gets a list of available InVEST models.

    Returns:
        A JSON string
    """
    install_language(request.args.get('language', 'en'))
    reloaded_cli = importlib.reload(cli)
    LOGGER.debug('get model list')
    a = reloaded_cli.build_model_list_json()
    print(a)
    return a


@app.route('/getspec', methods=['POST'])
def get_invest_getspec():
    """Gets the ARGS_SPEC dict from an InVEST model.

    Body (JSON string): "carbon"

    Returns:
        A JSON string.
    """
    # this will be the model key name, not language specific
    target_model = request.get_json()
    target_module = cli._MODEL_UIS[target_model].pyname
    install_language(request.args.get('language', 'en'))
    model_module = importlib.import_module(name=target_module)
    return spec_utils.serialize_args_spec(model_module.ARGS_SPEC)


@app.route('/validate', methods=['POST'])
def get_invest_validate():
    """Gets the return value of an InVEST model's validate function.

    Body (JSON string):
        model_module: string (e.g. natcap.invest.carbon)
        args: JSON string of InVEST model args keys and values

    Returns:
        A JSON string.
    """
    install_language(request.args.get('language', 'en'))

    payload = request.get_json()
    LOGGER.debug(payload)
    target_module = payload['model_module']
    args_dict = json.loads(payload['args'])
    LOGGER.debug(args_dict)
    try:
        limit_to = payload['limit_to']
    except KeyError:
        limit_to = None
    model_module = importlib.import_module(name=target_module)
    results = model_module.validate(args_dict, limit_to=limit_to)
    LOGGER.debug(results)
    return json.dumps(results)


@app.route('/colnames', methods=['POST'])
def get_vector_colnames():
    """Get a list of column names from a vector.
    This is used to fill in dropdown menu options in a couple models.

    Body (JSON string):
        vector_path (string): path to a vector file

    Returns:
        a JSON string.
    """
    payload = request.get_json()
    LOGGER.debug(payload)
    vector_path = payload['vector_path']
    # a lot of times the path will be empty so don't even try to open it
    if vector_path:
        try:
            vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
            colnames = [defn.GetName() for defn in vector.GetLayer().schema]
            LOGGER.debug(colnames)
            return json.dumps(colnames)
        except Exception as e:
            LOGGER.exception(
                f'Could not read column names from {vector_path}. ERROR: {e}')
    else:
        LOGGER.error('Empty vector path.')
    # 422 Unprocessable Entity: the server understands the content type
    # of the request entity, and the syntax of the request entity is
    # correct, but it was unable to process the contained instructions.
    return json.dumps([]), 422


@app.route('/post_datastack_file', methods=['POST'])
def post_datastack_file():
    """Extracts InVEST model args from json, logfiles, or datastacks.

    Body (JSON string): path to file

    Returns:
        A JSON string.
    """
    filepath = request.get_json()
    stack_type, stack_info = datastack.get_datastack_info(
        filepath)
    run_name, human_name = MODULE_MODELRUN_MAP[stack_info.model_name]
    result_dict = {
        'type': stack_type,
        'args': stack_info.args,
        'module_name': stack_info.model_name,
        'model_run_name': run_name,
        'model_human_name': human_name,
        'invest_version': stack_info.invest_version
    }
    return json.dumps(result_dict)


@app.route('/write_parameter_set_file', methods=['POST'])
def write_parameter_set_file():
    """Writes InVEST model args keys and values to a datastack JSON file.

    Body (JSON string):
        parameterSetPath: string
        moduleName: string(e.g. natcap.invest.carbon)
        args: JSON string of InVEST model args keys and values
        relativePaths: boolean

    Returns:
        A string.
    """
    payload = request.get_json()
    filepath = payload['parameterSetPath']
    modulename = payload['moduleName']
    args = json.loads(payload['args'])
    relative_paths = payload['relativePaths']

    datastack.build_parameter_set(
        args, modulename, filepath, relative=relative_paths)
    return 'parameter set saved'


@app.route('/save_to_python', methods=['POST'])
def save_to_python():
    """Writes a python script with a call to an InVEST model execute function.

    Body (JSON string):
        filepath: string
        modelname: string (e.g. carbon)
        args_dict: JSON string of InVEST model args keys and values

    Returns:
        A string.
    """
    payload = request.get_json()
    save_filepath = payload['filepath']
    modelname = payload['modelname']
    args_dict = json.loads(payload['args'])

    cli.export_to_python(
        save_filepath, modelname, args_dict)

    return 'python script saved'
