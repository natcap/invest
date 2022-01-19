# coding=UTF-8
"""Functions for reading and writing InVEST model parameters.

A **datastack** for InVEST is a compressed archive that includes the
arguments for a model, all of the data files referenced by the arguments, and
a logfile with some extra information about how the archive was created.  The
resulting archive can then be extracted on a different computer and should
have all of the information it needs to run an InVEST model in its entirity.

A **parameter set** for InVEST is a JSON-formatted text file that contains all
of the parameters needed to run the current model, where the parameters are
located on the local hard disk.  Paths to files may be either relative or
absolute.  If paths are relative, they are interpreted as relative to the
location of the parameter set file.

A **logfile** for InVEST is a text file that is written to disk for each model
run.
"""

import ast
import codecs
import collections
import importlib
import json
import logging
import math
import os
import pprint
import re
import shutil
import tarfile
import tempfile
import warnings

from osgeo import gdal

from . import utils

try:
    from . import __version__
except ImportError:
    # The only known case where this will be triggered is when building the API
    # documentation because natcap.invest is not installed into the
    # environment.
    __version__ = 'UNKNOWN'
    warnings.warn(
        '__version__ attribute of natcap.invest could not be imported.',
        RuntimeWarning)


LOGGER = logging.getLogger(__name__)
ARGS_LOG_LEVEL = 100  # define high log level so it should always show in logs
DATASTACK_EXTENSION = '.invest.tar.gz'
PARAMETER_SET_EXTENSION = '.invest.json'
DATASTACK_PARAMETER_FILENAME = 'parameters' + PARAMETER_SET_EXTENSION
UNKNOWN = 'UNKNOWN'

ParameterSet = collections.namedtuple('ParameterSet',
                                      'args model_name invest_version')


def _copy_spatial_files(spatial_filepath, target_dir):
    """Copy spatial files to a new directory.

    Args:
        spatial_filepath (str): The filepath to a GDAL-supported file.
        target_dir (str): The directory where all component files of
            ``spatial_filepath`` should be copied.

    Returns:
        filepath (str): The path to a representative file copied into the
        ``target_dir``.  If possible, this will match the basename of
        ``spatial_filepath``, so if someone provides an ESRI Shapefile called
        ``my_vector.shp``, the return value will be ``os.path.join(target_dir,
        my_vector.shp)``.
    """
    LOGGER.info(f'Copying {spatial_filepath} --> {target_dir}')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    source_basename = os.path.basename(spatial_filepath)
    return_filepath = None

    spatial_file = gdal.OpenEx(spatial_filepath)
    for member_file in spatial_file.GetFileList():
        # ArcGIS Binary/Grid format includes the directory in the file listing.
        # The parent directory isn't strictly needed, so we can just skip it.
        if os.path.isdir(member_file):
            continue

        target_basename = os.path.basename(member_file)
        target_filepath = os.path.join(target_dir, target_basename)
        if source_basename == target_basename:
            return_filepath = target_filepath
        shutil.copyfile(member_file, target_filepath)
    spatial_file = None

    # I can't conceive of a case where the basename of the source file does not
    # match any of the member file basenames, but just in case there's a
    # weird GDAL driver that does this, it seems reasonable to fall back to
    # whichever of the member files was most recent.
    if not return_filepath:
        return_filepath = target_filepath

    return return_filepath


def format_args_dict(args_dict, model_name):
    """Nicely format an arguments dictionary for writing to a stream.

    If printed to a console, the returned string will be aligned in two columns
    representing each key and value in the arg dict.  Keys are in ascending,
    sorted order.  Both columns are left-aligned.

    Args:
        args_dict (dict): The args dictionary to format.
        model_name (string): The model name (in python package import format)

    Returns:
        A formatted, unicode string.
    """
    sorted_args = sorted(args_dict.items(), key=lambda x: x[0])

    max_key_width = 0
    if len(sorted_args) > 0:
        max_key_width = max(len(x[0]) for x in sorted_args)

    format_str = "%-" + str(max_key_width) + "s %s"

    args_string = '\n'.join([format_str % (arg) for arg in sorted_args])
    args_string = "Arguments for InVEST %s %s:\n%s\n" % (model_name,
                                                         __version__,
                                                         args_string)
    return args_string


def get_datastack_info(filepath):
    """Get information about a datastack.

    Args:
        filepath (string): The path to a file on disk that can be extracted as
            a datastack, parameter set, or logfile.

    Returns:
        A 2-tuple.  The first item of the tuple is one of:

            * ``"archive"`` when the file is a datastack archive.
            * ``"json"`` when the file is a json parameter set.
            * ``"logfile"`` when the file is a text logfile.

        The second item of the tuple is a ParameterSet namedtuple with the raw
        parsed args, modelname and invest version that the file was built with.
    """
    if tarfile.is_tarfile(filepath):
        # If it's a tarfile, we need to extract the parameters file to be able
        # to inspect the parameters and model details.
        with tarfile.open(filepath) as archive:
            try:
                temp_directory = tempfile.mkdtemp()
                archive.extract('./' + DATASTACK_PARAMETER_FILENAME,
                                temp_directory)
                return 'archive', extract_parameter_set(
                    os.path.join(temp_directory, DATASTACK_PARAMETER_FILENAME))
            finally:
                try:
                    shutil.rmtree(temp_directory)
                except OSError:
                    # If something happens and we can't remove temp_directory,
                    # just log the exception and continue with program
                    # execution.
                    LOGGER.exception('Could not remove %s', temp_directory)

    try:
        return 'json', extract_parameter_set(filepath)
    except ValueError:
        # When a JSON object can't be decoded, it must not be a paramset.
        pass

    return 'logfile', extract_parameters_from_logfile(filepath)


def build_datastack_archive(args, model_name, datastack_path):
    """Build an InVEST datastack from an arguments dict.

    Args:
        args (dict): The arguments dictionary to include in the datastack.
        model_name (string): The python-importable module string of the model
            these args are for.
        datastack_path (string): The path to where the datastack archive
            should be written.

    Returns:
        ``None``
    """
    module = importlib.import_module(name=model_name)

    # Allow the model to override the common datastack function.  This is
    # useful for tables (like HRA) that are too complicated to describe in the
    # ARGS_SPEC format.
    if hasattr(module, 'build_datastack_archive'):
        return module.build_datastack_archive(args, datastack_path)

    args = args.copy()
    temp_workspace = tempfile.mkdtemp(prefix='datastack_')
    data_dir = os.path.join(temp_workspace, 'data')
    os.makedirs(data_dir)

    # write a logfile to the archive
    logfile = os.path.join(temp_workspace, 'log.txt')
    archive_filehandler = logging.FileHandler(logfile, 'w')
    archive_formatter = logging.Formatter(
        "%(name)-25s %(levelname)-8s %(message)s")
    archive_filehandler.setFormatter(archive_formatter)
    archive_filehandler.setLevel(logging.NOTSET)
    logging.getLogger().addHandler(archive_filehandler)

    # For tracking existing files so we don't copy files in twice
    files_found = {}
    LOGGER.debug(f'Keys: {sorted(args.keys())}')
    args_spec = module.ARGS_SPEC['args']

    spatial_types = {'raster', 'vector'}
    file_based_types = spatial_types.union({'csv', 'file', 'directory'})
    rewritten_args = {}
    for key in args:
        # We don't want to accidentally archive a user's complete workspace
        # directory, complete with prior runs there.
        if key == 'workspace_dir':
            LOGGER.debug(
                f"Skipping workspace directory: {args['workspace_dir']}")
            continue

        LOGGER.info(f'Starting to archive arg "{key}": {args[key]}')
        # Possible that a user might pass an args key that doesn't belong to
        # this model.  Skip if so.
        if key not in args_spec:
            LOGGER.info(f'Skipping arg {key}; not in model ARGS_SPEC')

        input_type = args_spec[key]['type']
        if input_type in file_based_types:
            if args[key] in {None, ''}:
                LOGGER.info(
                    f'Skipping key {key}, value is empty and cannot point to '
                    'a file.')
                rewritten_args[key] = ''
                continue

            # Python can't handle mixed file separators, so let's just
            # standardize on linux filepaths.
            source_path = args[key].replace('\\', '/')

            # If we already know about the parameter, then we can just reuse it
            # and skip the file copying.
            if source_path in files_found:
                LOGGER.debug(
                    f'Key {key} is known: using {files_found[source_path]}')
                rewritten_args[key] = files_found[source_path]
                continue

        if input_type == 'csv':
            # check the CSV for columns that may be spatial.
            # But also, the columns specification might not be listed, so don't
            # require that 'columns' exists in the ARGS_SPEC.
            spatial_columns = []
            if 'columns' in args_spec[key]:
                for col_name, col_definition in (
                        args_spec[key]['columns'].items()):
                    # Type attribute may be a string (one type) or set
                    # (multiple types allowed), so always convert to a set for
                    # easier comparison.
                    col_types = col_definition['type']
                    if isinstance(col_types, str):
                        col_types = set([col_types])
                    if col_types.intersection(spatial_types):
                        spatial_columns.append(col_name)
            LOGGER.debug(f'Detected spatial columns: {spatial_columns}')

            target_csv_path = os.path.join(
                data_dir, f'{key}_csv.csv')
            if not spatial_columns:
                LOGGER.debug(
                    f'No spatial columns, copying to {target_csv_path}')
                shutil.copyfile(source_path, target_csv_path)
            else:
                contained_files_dir = os.path.join(
                    data_dir, f'{key}_csv_data')

                dataframe = utils.read_csv_to_dataframe(
                    source_path, to_lower=True)
                csv_source_dir = os.path.abspath(os.path.dirname(source_path))
                for spatial_column_name in spatial_columns:
                    # Iterate through the spatial columns, identify the set of
                    # unique files and copy them out.
                    # if a string is not a filepath, assume it's supposed to be
                    # there and skip it
                    for row_index, column_value in dataframe[
                            spatial_column_name.lower()].items():
                        if ((isinstance(column_value, float) and
                                math.isnan(column_value)) or
                                column_value == ''):
                            # The table cell is blank, so skip it.
                            # We can't compare nan values directly in a way
                            # that also works for strings, so skip it.
                            continue

                        source_filepath = None
                        for possible_filepath in (
                                column_value,
                                os.path.join(csv_source_dir, column_value)):
                            if os.path.exists(possible_filepath):
                                source_filepath = possible_filepath
                                break

                        # If we didn't end up finding a valid source filepath
                        # for the field value, assume it's supposed to be that
                        # way and leave it alone.
                        if not source_filepath:
                            continue

                        try:
                            # This path is already relative to the data
                            # directory
                            target_filepath = files_found[source_filepath]
                        except KeyError:
                            basename = os.path.splitext(
                                os.path.basename(source_filepath))[0]
                            target_dir = os.path.join(
                                contained_files_dir,
                                f'{row_index}_{basename}')
                            target_filepath = _copy_spatial_files(
                                source_filepath, target_dir)
                            target_filepath = os.path.relpath(
                                target_filepath, data_dir)

                        LOGGER.debug(
                            'Spatial file in CSV copied from '
                            f'{source_filepath} --> {target_filepath}')
                        dataframe.at[
                            row_index, spatial_column_name] = target_filepath
                        files_found[source_filepath] = target_filepath

                LOGGER.debug(
                    f'Rewritten spatial CSV written to {target_csv_path}')
                dataframe.to_csv(target_csv_path)

            target_arg_value = target_csv_path
            files_found[source_path] = target_arg_value

        elif input_type == 'file':
            target_filepath = os.path.join(
                data_dir, f'{key}_file')
            shutil.copyfile(source_path, target_filepath)
            LOGGER.debug(
                f'File copied from {source_path} --> {target_filepath}')
            target_arg_value = target_filepath
            files_found[source_path] = target_arg_value

        elif input_type == 'directory':
            # copy the whole folder
            target_directory = os.path.join(data_dir, f'{key}_directory')
            os.makedirs(target_directory)

            # We want to copy the directory contents into the directory
            # directly, not copy the parent folder into the directory.
            for filename in os.listdir(source_path):
                src_path = os.path.join(source_path, filename)
                dest_path = os.path.join(target_directory, filename)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copyfile(src_path, dest_path)

            LOGGER.debug(
                f'Directory copied from {source_path} --> {target_directory}')
            target_arg_value = target_directory
            files_found[source_path] = target_arg_value

        elif input_type in spatial_types:
            # Create a directory with a readable name, something like
            # "aoi_path_vector" or "lulc_cur_path_raster".
            spatial_dir = os.path.join(data_dir, f'{key}_{input_type}')
            target_arg_value = _copy_spatial_files(
                source_path, spatial_dir)
            files_found[source_path] = target_arg_value

        elif input_type == 'other':
            # Note that no models currently use this to the best of my
            # knowledge, so better to raise a NotImplementedError
            raise NotImplementedError(
                'The "other" ARGS_SPEC input type is not supported')
        else:
            LOGGER.debug(
                f"Type {input_type} is not filesystem-based; "
                "recording value directly")
            # not a filesystem-based type
            # Record the value directly
            target_arg_value = args[key]
        rewritten_args[key] = target_arg_value

    LOGGER.info('Args preprocessing complete')

    LOGGER.debug(f'found files: \n{pprint.pformat(files_found)}')
    LOGGER.debug(f'new arguments: \n{pprint.pformat(rewritten_args)}')
    # write parameters to a new json file in the temp workspace
    param_file_uri = os.path.join(temp_workspace,
                                  'parameters' + PARAMETER_SET_EXTENSION)
    build_parameter_set(
        rewritten_args, model_name, param_file_uri, relative=True)

    # Remove the handler before archiving the working dir (and the logfile)
    archive_filehandler.close()
    logging.getLogger().removeHandler(archive_filehandler)

    # archive the workspace.
    with utils.sandbox_tempdir() as temp_dir:
        temp_archive = os.path.join(temp_dir, 'invest_archive')
        archive_name = shutil.make_archive(
            temp_archive, 'gztar', root_dir=temp_workspace,
            logger=LOGGER, verbose=True)
        shutil.move(archive_name, datastack_path)


def extract_datastack_archive(datastack_path, dest_dir_path):
    """Extract a datastack to a given folder.

    Args:
        datastack_path (string): The path to a datastack archive on disk.
        dest_dir_path (string): The path to a directory.  The contents of the
            demonstration datastack archive will be extracted into this
            directory. If the directory does not exist, it will be created.

    Returns:
        ``args`` (dict): A dictionary of arguments from the extracted
            archive.  Paths to files are absolute paths.
    """
    LOGGER.info('Extracting archive %s to %s', datastack_path, dest_dir_path)
    dest_dir_path = os.path.abspath(dest_dir_path)
    # extract the archive to the workspace
    with tarfile.open(datastack_path) as tar:
        tar.extractall(dest_dir_path)

    # get the arguments dictionary
    arguments_dict = json.load(open(
        os.path.join(dest_dir_path, DATASTACK_PARAMETER_FILENAME)))['args']

    def _rewrite_paths(args_param):
        """Converts paths in `args_param` to paths in `dest_dir_path."""
        if isinstance(args_param, dict):
            _args = {}
            for key, value in args_param.items():
                _args[key] = _rewrite_paths(value)
            return _args
        elif isinstance(args_param, list):
            return [_rewrite_paths(param) for param in args_param]
        elif isinstance(args_param, str):
            # Special case: if the value is blank, return an empty string
            # rather than assuming it's the CWD.
            if args_param.strip() == '':
                return ''

            # Archives always store linux-style paths.  os.path.normpath
            # converts forward slashes to backslashes if we're on Windows.
            data_path = os.path.normpath(
                os.path.join(dest_dir_path, args_param))
            if os.path.exists(data_path):
                return data_path
        return args_param

    new_args = _rewrite_paths(arguments_dict)
    LOGGER.debug('Expanded parameters as \n%s', pprint.pformat(new_args))
    return new_args


def build_parameter_set(args, model_name, paramset_path, relative=False):
    """Record a parameter set to a file on disk.

    Args:
        args (dict): The args dictionary to record to the parameter set.
        model_name (string): An identifier string for the callable or InVEST
            model that would accept the arguments given.
        paramset_path (string): The path to the file on disk where the
            parameters should be recorded.
        relative (bool): Whether to save the paths as relative.  If ``True``,
            The datastack assumes that paths are relative to the parent
            directory of ``paramset_path``.

    Returns:
        ``None``
    """
    def _recurse(args_param):
        if isinstance(args_param, dict):
            return dict((key, _recurse(value))
                        for (key, value) in args_param.items())
        elif isinstance(args_param, list):
            return [_recurse(param) for param in args_param]
        elif isinstance(args_param, str):
            # If args_param is empty string '' os.path.exists will be False
            if os.path.exists(args_param):
                normalized_path = os.path.normpath(args_param)
                if relative:
                    # Handle special case where python assumes that '.'
                    # represents the CWD
                    if (normalized_path == '.' or
                            os.path.dirname(paramset_path) == normalized_path):
                        return '.'
                    temp_rel_path = os.path.relpath(
                        normalized_path, os.path.dirname(paramset_path))
                    # Always save unix paths.
                    linux_style_path = temp_rel_path.replace('\\', '/')
                else:
                    # Always save unix paths.
                    linux_style_path = normalized_path.replace('\\', '/')

                return linux_style_path
        return args_param
    parameter_data = {
        'model_name': model_name,
        'invest_version': __version__,
        'args': _recurse(args)
    }
    with codecs.open(paramset_path, 'w', encoding='UTF-8') as paramset_file:
        paramset_file.write(
            json.dumps(parameter_data,
                       indent=4,
                       sort_keys=True))


def extract_parameter_set(paramset_path):
    """Extract and return attributes from a parameter set.

    Any string values found will have environment variables expanded.  See
    :py:ref:os.path.expandvars and :py:ref:os.path.expanduser for details.

    Args:
        paramset_path (string): The file containing a parameter set.

    Returns:
        A ``ParameterSet`` namedtuple with these attributes::

            args (dict): The arguments dict for the callable
            invest_version (string): The version of InVEST used to record the
                parameter set.
            model_name (string): The name of the callable or model that these
                arguments are intended for.
    """
    paramset_parent_dir = os.path.dirname(os.path.abspath(paramset_path))
    with codecs.open(paramset_path, 'r', encoding='UTF-8') as paramset_file:
        params_raw = paramset_file.read()

    read_params = json.loads(params_raw)

    def _recurse(args_param):
        if isinstance(args_param, dict):
            return dict((key, _recurse(value)) for (key, value) in
                        args_param.items())
        elif isinstance(args_param, list):
            return [_recurse(param) for param in args_param]
        elif isinstance(args_param, str) and len(args_param) > 0:
            # Attempt to parse true/false strings.
            try:
                return {'true': True, 'false': False}[args_param.lower()]
            except KeyError:
                # Probably not a boolean, so continue checking paths.
                pass

            # Convert paths to whatever makes sense for the current OS.
            expanded_param = os.path.expandvars(
                os.path.expanduser(
                    os.path.normpath(args_param)))
            if os.path.isabs(expanded_param):
                return expanded_param
            else:
                paramset_rel_path = os.path.abspath(
                    os.path.join(paramset_parent_dir, args_param))
                if os.path.exists(paramset_rel_path):
                    return paramset_rel_path
        else:
            return args_param
        return args_param

    return ParameterSet(_recurse(read_params['args']),
                        read_params['model_name'],
                        read_params['invest_version'])


def extract_parameters_from_logfile(logfile_path):
    """Parse an InVEST logfile for the parameters (args) dictionary.

    Argument key-value pairs are parsed, one pair per line, starting the line
    after the line starting with ``"Arguments"``, and ending with a blank line.
    If no such section exists within the logfile, ``ValueError`` will be
    raised.

    If possible, the model name and InVEST version will be parsed from the
    same line as ``"Arguments"``, but IUI-formatted logfiles (without model
    name and InVEST version information) are also supported.

    Args:
        logfile_path (string): The path to an InVEST logfile on disk.

    Returns:
        An instance of the ParameterSet namedtuple.  If a model name and InVEST
        version cannot be parsed from the Arguments section of the logfile,
        ``ParameterSet.model_name`` and ``ParameterSet.invest_version`` will be
        set to ``datastack.UNKNOWN``.

    Raises:
        ValueError - when no arguments could be parsed from the logfile.
    """
    with codecs.open(logfile_path, 'r', encoding='utf-8') as logfile:
        detected_args = []
        args_started = False
        for line in logfile:
            line = line.strip()

            if not args_started:
                # Line would look something like this:
                # "Arguments for InVEST natcap.invest.carbon 3.4.1rc1:\n"
                if line.startswith('Arguments'):
                    try:
                        modelname, invest_version = line.split(' ')[3:5]
                        invest_version = invest_version.replace(':', '')
                    except ValueError:
                        # Old-style logfiles don't provide the modelename or
                        # version info.
                        modelname = UNKNOWN
                        invest_version = UNKNOWN
                    args_started = True
                    continue
            else:
                if line == '':
                    break
                detected_args.append(line)

    if not detected_args:
        raise ValueError('No arguments could be parsed from %s' % logfile_path)

    args_dict = {}
    for argument in detected_args:
        # args key is everything before the whitespace
        args_key = re.findall(r'^\w*', argument)[0]
        args_value = re.sub('^%s' % args_key, '', argument).strip()

        def _smart_cast(value):
            """Attempt to cast a value to an int or a float.

            Leave the value be if the value cannot be cast to either one.

            Args:
                value: The parameter value to try to cast.

            Returns:
                A possibly-cast version of the original value.
            """
            for cast_to_type in (float, int):
                try:
                    value = cast_to_type(value)
                    break
                except ValueError:
                    pass
            return value

        try:
            # This will cast values appropriately for int, float, bool,
            # nonetype, list.
            args_value = ast.literal_eval(args_value)
        except (ValueError, SyntaxError):
            # If ast.literal_eval can't evaluate the string, keep the string
            # as it is.
            pass

        args_dict[args_key] = args_value
    return ParameterSet(args_dict, modelname, invest_version)
