"""
A module for InVEST test-related data storage.
"""

import contextlib
import os
import json
import tarfile
import shutil
import logging
import tempfile
import string
import random
import codecs

from osgeo import gdal
from osgeo import ogr

from . import utils


class UnsupportedFormat(Exception):
    pass


class NotAVector(Exception):
    pass


DATA_ARCHIVES = os.path.join('data', 'regression_archives')
INPUT_ARCHIVES = os.path.join(DATA_ARCHIVES, 'input')
LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def log_to_file(logfile):
    handler = logging.FileHandler(logfile, 'w', encoding='UTF-8')
    formatter = logging.Formatter(
        "%(asctime)s %(name)-18s %(levelname)-8s %(message)s",
        "%m/%d/%Y %H:%M:%S ")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)  # capture everything
    root_logger.addHandler(handler)
    handler.setFormatter(formatter)
    yield
    handler.close()
    root_logger.removeHandler(handler)


def build_scenario(args, out_scenario_path, archive_data):
    # TODO: Add a checksum for each input

    tmp_scenario_dir = tempfile.mkdtemp(prefix='scenario_')
    parameters_path = os.path.join(tmp_scenario_dir, 'parameters')
    log_path = os.path.join(tmp_scenario_dir, 'log')
    data_dir = os.path.join(tmp_scenario_dir, 'data')
    with log_to_file(log_path):
        os.makedirs(data_dir)

        # convert parameters to local filepaths.

        # Write the parameters to a file.
        with codecs.open(parameters_path, 'w', encoding='UTF-8') as params:
            params.write(json.dump(args,
                                   encoding='UTF-8',
                                   indent=4,
                                   sort_keys=True))


def collect_parameters(parameters, archive_uri):
    """Collect an InVEST model's arguments into a dictionary and archive all
        the input data.

        parameters - a dictionary of arguments
        archive_uri - a URI to the target archive.

        Returns nothing."""

    parameters = parameters.copy()
    temp_workspace = tempfile.mkdtemp(prefix='scenario_')
    data_dir = os.path.join(temp_workspace, 'data')
    os.makedirs(data_dir)

    def get_multi_part(filepath):
        # If the user provides a mutli-part file, wrap it into a folder and grab
        # that instead of the individual file.

        with utils.capture_gdal_logging():
            raster = gdal.Open(filepath)
            if raster is not None:
                driver = raster.GetDriver()
                new_path = tempfile.mkdtemp(prefix='raster_', dir=data_dir)
                LOGGER.info('Saving new raster to %s', new_path)
                # driver.CreateCopy returns None if there's an error
                # Common case: driver does not have Create() method implemented
                # ESRI Arc/Binary Grids are a great example of this.
                if not driver.CreateCopy(new_path, raster):
                    LOGGER.info('Manually copying raster files to %s',
                                new_path)
                    for filename in raster.GetFileList():
                        if os.path.isdir(filename) and (
                                os.path.abspath(filename) ==
                                os.path.abspath(filepath)):
                            continue
                        new_filename = os.path.join(
                            new_path,
                            os.path.basename(filename))
                        shutil.copyfile(filename, new_filename)
                driver = None
                raster = None
                return new_path

            vector = ogr.Open(filepath)
            if vector is not None:
                # OGR also reads CSVs; verify this IS actually a vector
                driver = vector.GetDriver()
                new_path = tempfile.mkdtemp(prefix='vector_', dir=data_dir)
                LOGGER.info('Saving new vector to %s', new_path)
                new_vector = driver.CopyDataSource(vector, new_path)
                if not new_vector:
                    new_path = os.path.join(new_path,
                                            os.path.basename(filepath))
                    new_vector = driver.CopyDataSource(vector, new_path)
                new_vector.SyncToDisk()
                driver = None
                vector = None
                return new_path
        return None

    # For tracking existing files so we don't copy things twice
    files_found = {}

    def get_if_file(parameter):
        try:
            uri = files_found[os.path.abspath(parameter)]
            LOGGER.debug('Found %s from a previous parameter', uri)
            return uri
        except KeyError:
            # we haven't found this file before, so we still need to process it.
            pass

        # initialize the return_path
        return_path = None
        try:
            multi_part_folder = get_multi_part(parameter)
            if multi_part_folder is not None:
                LOGGER.debug('%s is a multi-part file', parameter)
                return_path = multi_part_folder

            elif os.path.isfile(parameter):
                LOGGER.debug('%s is a single file', parameter)
                new_filename = os.path.join(data_dir,
                                            os.path.basename(parameter))
                shutil.copyfile(parameter, new_filename)
                return_path = new_filename

            elif os.path.isdir(parameter):
                LOGGER.debug('%s is a directory', parameter)
                # parameter is a folder, so we want to copy the folder and all
                # its contents to the data dir.
                new_foldername = tempfile.mkdtemp(
                    prefix='data_', dir=data_dir)
                for filename in os.listdir(parameter):
                    shutil.copyfile(os.path.join(parameter, filename),
                                    os.path.join(new_foldername, filename))
                return_path = new_foldername

            else:
                # Parameter does not exist on disk.  Print an error to the
                # logger and move on.
                LOGGER.error('File %s does not exist on disk.  Skipping.',
                             parameter)
        except TypeError as e:
            # When the value is not a string.
            LOGGER.warn('%s', e)

        if return_path is not None:
            files_found[os.path.abspath(parameter)] = return_path
            LOGGER.debug('Return path: %s', return_path)
            return return_path

        LOGGER.debug('Returning original parameter %s', parameter)
        return parameter

    # Recurse through the parameters to locate any URIs
    #   If a URI is found, copy that file to a new location in the temp
    #   workspace and update the URI reference.
    #   Duplicate URIs should also have the same replacement URI.
    #
    # If a workspace or suffix is provided, ignore that key.
    LOGGER.debug('Keys: %s', sorted(parameters.keys()))

    def _recurse(args_param):
        if isinstance(args_param, dict):
            new_dict = {}
            for args_key, args_value in args_param.iteritems():
                if args_key not in ('workspace_dir',):
                    new_dict[args_key] = _recurse(args_value)
            return new_dict
        elif isinstance(args_param, list):
            return [_recurse(list_item) for list_item in args_param]
        else:
            if (isinstance(args_param, basestring) and
                    os.path.exists(args_param)):
                # It's a string and exists on disk, it's a file!
                return get_if_file(args_param)
            else:
                # It's not a file or a structure to recurse through, so
                # just return the item verbatim.
                return args_param

    new_args = _recurse(parameters)

    LOGGER.debug('new arguments: %s', new_args)
    # write parameters to a new json file in the temp workspace
    param_file_uri = os.path.join(temp_workspace, 'parameters.json')
    with codecs.open(param_file_uri, 'w', encoding='UTF-8') as params:
        params.write(json.dumps(new_args,
                                encoding='UTF-8',
                                indent=4,
                                sort_keys=True))

    # archive the workspace.
    if archive_uri[-7:] == '.tar.gz':
        archive_uri = archive_uri[:-7]
    shutil.make_archive(archive_uri, 'gztar', root_dir=temp_workspace,
                        logger=LOGGER)


def extract_archive(workspace_dir, archive_uri):
    """Extract a .tar.gzipped file to the given workspace.

        workspace_dir - the folder to which the archive should be extracted
        archive_uri - the uri to the target archive

        Returns nothing."""

    archive = tarfile.open(archive_uri)
    archive.extractall(workspace_dir)
    archive.close()


def format_dictionary(input_dict, types_lookup={}):
    """Recurse through the input dictionary and return a formatted dictionary.

        As each element is encountered, the correct function to use is looked up
        in the types_lookup input.  If a type is not found, we assume that the
        element should be returned verbatim.

        input_dict - a dictionary to process
        types_lookup - a dictionary mapping types to functions.  These functions
            must take a single parameter of the type that is the key.  These
            functions must return a formatted version of the input parameter.

        Returns a formatted dictionary."""

    def format_dict(parameter):
        new_dict = {}
        for key, value in parameter.iteritems():
            try:
                new_dict[key] = types[value.__class__](value)
            except KeyError:
                new_dict[key] = value
        return new_dict

    def format_list(parameter):
        new_list = []
        for item in parameter:
            try:
                new_list.append(types[item.__class__](item))
            except KeyError:
                new_list.append(item)
        return new_list

    types = {
        dict: format_dict,
        list: format_list,
    }

    types.update(types_lookup)

    return format_dict(input_dict)


def extract_parameters_archive(workspace_dir, archive_uri, input_folder=None):
    """Extract the target archive to the target workspace folder.

        workspace_dir - a uri to a folder on disk.  Must be an empty folder.
        archive_uri - a uri to an archive to be unzipped on disk.  Archive must
            be in .tar.gz format.
        input_folder=None - either a URI to a folder on disk or None.  If None,
            temporary folder will be created and then erased using the atexit
            register.

        Returns a dictionary of the model's parameters for this run."""

    # create a new temporary folder just for the input parameters, if the user
    # has not provided one already.
    if input_folder == None:
        input_folder = tempfile.mkdtemp()

    # extract the archive to the workspace
    extract_archive(input_folder, archive_uri)

    # get the arguments dictionary
    arguments_dict = json.load(open(os.path.join(input_folder, 'parameters.json')))

    def _get_if_uri(parameter):
        """If the parameter is a file, returns the filepath relative to the
        extracted workspace.  If the parameter is not a file, returns the
        original parameter."""
        try:
            temp_file_path = os.path.join(input_folder, parameter)
            if os.path.exists(temp_file_path) and not len(parameter) == 0:
                return temp_file_path
        except TypeError:
            # When the parameter is not a string
            pass
        except AttributeError:
            # when the parameter is not a string
            pass

        return parameter

    types = {
        str: _get_if_uri,
        unicode: _get_if_uri,
    }
    formatted_args = format_dictionary(arguments_dict, types)
    formatted_args[u'workspace_dir'] = workspace_dir

    return formatted_args
