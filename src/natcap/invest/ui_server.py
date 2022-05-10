"""A Flask app with HTTP endpoints used by the InVEST Workbench."""
import importlib
import json
import logging

from osgeo import gdal
from flask import Flask
from flask import request
from flask_cors import CORS
import natcap.invest
from natcap.invest import cli
from natcap.invest import datastack
from natcap.invest import set_locale
from natcap.invest.model_metadata import MODEL_METADATA
from natcap.invest import spec_utils
from natcap.invest import usage

LOGGER = logging.getLogger(__name__)

PREFIX = 'api'
app = Flask(__name__)
CORS(app, resources={
    f'/{PREFIX}/*': {
        'origins': 'http://localhost:*'
    }
})

PYNAME_TO_MODEL_NAME_MAP = {
    metadata.pyname: model_name
    for model_name, metadata in MODEL_METADATA.items()
}


@app.route(f'/{PREFIX}/ready', methods=['GET'])
def get_is_ready():
    """Returns something simple to confirm the server is open."""
    return 'Flask ready'


@app.route(f'/{PREFIX}/models', methods=['GET'])
def get_invest_models():
    """Gets a list of available InVEST models.

    Accepts a `language` query parameter which should be an ISO 639-1 language
    code. Model names will be translated to the requested language if
    translations are available, or fall back to English otherwise.

    Returns:
        A JSON string
    """
    LOGGER.debug('get model list')
    set_locale(request.args.get('language', 'en'))
    importlib.reload(natcap.invest)
    return cli.build_model_list_json()


@app.route(f'/{PREFIX}/getspec', methods=['POST'])
def get_invest_getspec():
    """Gets the ARGS_SPEC dict from an InVEST model.

    Body (JSON string): "carbon"
    Accepts a `language` query parameter which should be an ISO 639-1 language
    code. Spec 'about' and 'name' values will be translated to the requested
    language if translations are available, or fall back to English otherwise.

    Returns:
        A JSON string.
    """
    set_locale(request.args.get('language', 'en'))
    target_model = request.get_json()
    target_module = MODEL_METADATA[target_model].pyname
    model_module = importlib.reload(
        importlib.import_module(name=target_module))
    return spec_utils.serialize_args_spec(model_module.ARGS_SPEC)


@app.route(f'/{PREFIX}/validate', methods=['POST'])
def get_invest_validate():
    """Gets the return value of an InVEST model's validate function.

    Body (JSON string):
        model_module: string (e.g. natcap.invest.carbon)
        args: JSON string of InVEST model args keys and values

    Accepts a `language` query parameter which should be an ISO 639-1 language
    code. Validation messages will be translated to the requested language if
    translations are available, or fall back to English otherwise.

    Returns:
        A JSON string.
    """
    payload = request.get_json()
    LOGGER.debug(payload)
    try:
        limit_to = payload['limit_to']
    except KeyError:
        limit_to = None

    set_locale(request.args.get('language', 'en'))
    importlib.reload(natcap.invest.validation)
    model_module = importlib.reload(
        importlib.import_module(name=payload['model_module']))

    results = model_module.validate(
        json.loads(payload['args']), limit_to=limit_to)
    LOGGER.debug(results)
    return json.dumps(results)


@app.route(f'/{PREFIX}/colnames', methods=['POST'])
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


@app.route(f'/{PREFIX}/post_datastack_file', methods=['POST'])
def post_datastack_file():
    """Extracts InVEST model args from json, logfiles, or datastacks.

    Body (JSON string): path to file

    Returns:
        A JSON string.
    """
    filepath = request.get_json()
    stack_type, stack_info = datastack.get_datastack_info(
        filepath)
    model_name = PYNAME_TO_MODEL_NAME_MAP[stack_info.model_name]
    result_dict = {
        'type': stack_type,
        'args': stack_info.args,
        'module_name': stack_info.model_name,
        'model_run_name': model_name,
        'model_human_name': MODEL_METADATA[model_name].model_title,
        'invest_version': stack_info.invest_version
    }
    return json.dumps(result_dict)


@app.route(f'/{PREFIX}/write_parameter_set_file', methods=['POST'])
def write_parameter_set_file():
    """Writes InVEST model args keys and values to a datastack JSON file.

    Body (JSON string):
        filepath: string
        moduleName: string(e.g. natcap.invest.carbon)
        args: JSON string of InVEST model args keys and values
        relativePaths: boolean

    Returns:
        A string.
    """
    payload = request.get_json()
    filepath = payload['filepath']
    modulename = payload['moduleName']
    args = json.loads(payload['args'])
    relative_paths = payload['relativePaths']

    datastack.build_parameter_set(
        args, modulename, filepath, relative=relative_paths)
    return 'parameter set saved'


@app.route(f'/{PREFIX}/save_to_python', methods=['POST'])
def save_to_python():
    """Writes a python script with a call to an InVEST model execute function.

    Body (JSON string):
        filepath: string
        modelname: string (a key in natcap.invest.MODEL_METADATA)
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


@app.route(f'/{PREFIX}/build_datastack_archive', methods=['POST'])
def build_datastack_archive():
    """Writes a compressed archive of invest model input data.

    Body (JSON string):
        filepath: string - the target path to save the archive
        moduleName: string (e.g. natcap.invest.carbon) the python module name
        args: JSON string of InVEST model args keys and values

    Returns:
        A string.
    """
    payload = request.get_json()
    datastack.build_datastack_archive(
        json.loads(payload['args']),
        payload['moduleName'],
        payload['filepath'])

    return 'datastack archive created'


@app.route(f'/{PREFIX}/log_model_start', methods=['POST'])
def log_model_start():
    payload = request.get_json()
    usage._log_model(
        payload['model_pyname'],
        json.loads(payload['model_args']),
        payload['invest_interface'],
        payload['session_id'])
    return 'OK'


@app.route(f'/{PREFIX}/log_model_exit', methods=['POST'])
def log_model_exit():
    payload = request.get_json()
    usage._log_exit_status(
        payload['session_id'],
        payload['status'])
    return 'OK'
