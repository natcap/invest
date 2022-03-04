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
from natcap.invest import install_language
from natcap.invest import MODEL_METADATA
from natcap.invest import spec_utils
from natcap.invest import usage

LOGGER = logging.getLogger(__name__)
logging.getLogger('flask_cors').level = logging.DEBUG

app = Flask(__name__)
CORS(app, resources={
    '/api/*': {
        'origins': 'http://localhost:*'
    }
})

PYNAME_TO_MODEL_NAME_MAP = {
    metadata.pyname: model_name
    for model_name, metadata in MODEL_METADATA.items()
}

PREFIX = 'api'


def shutdown_server():
    """Shutdown the flask server."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/api/ready', methods=['GET'])
def get_is_ready():
    """Returns something simple to confirm the server is open."""
    return 'Flask ready'


@app.route('/api/shutdown', methods=['GET'])
def shutdown():
    """A request to this endpoint shuts down the server."""
    shutdown_server()
    return 'Flask server shutting down...'


@app.route('/api/models', methods=['GET'])
def get_invest_models():
    """Gets a list of available InVEST models.

    Accepts a `language` query parameter which should be an ISO 639-1 language
    code. Model names will be translated to the requested language if
    translations are available, or fall back to English otherwise.

    Returns:
        A JSON string
    """
    LOGGER.debug('get model list')
    install_language(request.args.get('language', 'en'))
    importlib.reload(natcap.invest)
    return cli.build_model_list_json()


@app.route('/api/getspec', methods=['POST'])
def get_invest_getspec():
    """Gets the ARGS_SPEC dict from an InVEST model.

    Body (JSON string): "carbon"
    Accepts a `language` query parameter which should be an ISO 639-1 language
    code. Spec 'about' and 'name' values will be translated to the requested
    language if translations are available, or fall back to English otherwise.

    Returns:
        A JSON string.
    """
    install_language(request.args.get('language', 'en'))
    target_model = request.get_json()
    target_module = MODEL_METADATA[target_model].pyname
    model_module = importlib.reload(
        importlib.import_module(name=target_module))
    return spec_utils.serialize_args_spec(model_module.ARGS_SPEC)


@app.route('/api/validate', methods=['POST'])
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

    install_language(request.args.get('language', 'en'))
    importlib.reload(natcap.invest.validation)
    model_module = importlib.reload(
        importlib.import_module(name=payload['model_module']))

    results = model_module.validate(
        json.loads(payload['args']), limit_to=limit_to)
    LOGGER.debug(results)
    return json.dumps(results)


@app.route('/api/colnames', methods=['POST'])
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


@app.route('/api/post_datastack_file', methods=['POST'])
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


@app.route('/api/write_parameter_set_file', methods=['POST'])
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
    filepath = payload['filepath']
    modulename = payload['moduleName']
    args = json.loads(payload['args'])
    relative_paths = payload['relativePaths']

    datastack.build_parameter_set(
        args, modulename, filepath, relative=relative_paths)
    return 'parameter set saved'


@app.route('/api/save_to_python', methods=['POST'])
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


@app.route('/api/build_datastack_archive', methods=['POST'])
def build_datastack_archive():
    """.

    Body (JSON string):
        filepath: string
        moduleName: string (e.g. natcap.invest.carbon)
        args: JSON string of InVEST model args keys and values

    Returns:
        A string.
    """
    payload = request.get_json()
    target_filepath = payload['filepath']
    pyname = payload['moduleName']
    args_dict = json.loads(payload['args'])

    datastack.build_datastack_archive(
        args_dict, pyname, target_filepath)

    return 'datastack archive created'


@app.route('/api/log_model_start', methods=['POST'])
def log_model_start():
    payload = request.get_json()
    usage._log_model(
        payload['model_pyname'],
        json.loads(payload['model_args']),
        payload['invest_interface'],
        payload['session_id'])
    return 'OK'


@app.route('/api/log_model_exit', methods=['POST'])
def log_model_exit():
    payload = request.get_json()
    usage._log_exit_status(
        payload['session_id'],
        payload['status'])
    return 'OK'
