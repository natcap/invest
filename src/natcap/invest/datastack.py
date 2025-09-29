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

from . import spec
from . import utils
from . import models


LOGGER = logging.getLogger(__name__)
DATASTACK_EXTENSION = '.invest.tar.gz'
PARAMETER_SET_EXTENSION = '.invest.json'
DATASTACK_PARAMETER_FILENAME = 'parameters' + PARAMETER_SET_EXTENSION


ParameterSet = collections.namedtuple('ParameterSet',
                                      'args model_id')


def _tarfile_safe_extract(archive_path, dest_dir_path):
    """Extract a tarfile in a safe way.

    This function avoids the CVE-2007-4559 exploit that's been a vulnerability
    in the python stdlib for at least 15 years now and should really be patched
    upstream.

    Args:
        archive_path (string): The path to a tarfile, such as a datastack
            archive created by InVEST.
        dest_dir_path (string): The path to the destination directory, where
            the contents should be unzipped.

    Returns:
        ``None``
    """
    # The guts of this function are taken from Trellix's PR to InVEST.  See
    # https://github.com/natcap/invest/pull/1099 for details.
    with tarfile.open(archive_path) as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory

        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(path, members, numeric_owner=numeric_owner)

        safe_extract(tar, dest_dir_path)


def get_datastack_info(filepath, extract_path=None):
    """Get information about a datastack.

    Args:
        filepath (string): The path to a file on disk that can be extracted as
            a datastack, parameter set, or logfile.
        extract_path (str): Path to a directory to extract the datastack, if
            provided as an archive. Will be overwritten if it already exists,
            or created if it does not already exist.

    Returns:
        A 2-tuple.  The first item of the tuple is one of:

            * ``"archive"`` when the file is a datastack archive.
            * ``"json"`` when the file is a json parameter set.
            * ``"logfile"`` when the file is a text logfile.

        The second item of the tuple is a ParameterSet namedtuple with the raw
        parsed args, model id and invest version that the file was built with.
    """
    if tarfile.is_tarfile(filepath):
        if not extract_path:
            raise ValueError('extract_path must be provided if using archive')
        if os.path.isfile(extract_path):
            os.remove(extract_path)
        elif os.path.isdir(extract_path):
            shutil.rmtree(extract_path)
        os.mkdir(extract_path)
        # If it's a tarfile, we need to extract the parameters file to be able
        # to inspect the parameters and model details.
        extract_datastack_archive(filepath, extract_path)
        return 'archive', extract_parameter_set(
            os.path.join(extract_path, DATASTACK_PARAMETER_FILENAME))

    try:
        return 'json', extract_parameter_set(filepath)
    except ValueError:
        # When a JSON object can't be decoded, it must not be a paramset.
        pass

    return 'logfile', extract_parameters_from_logfile(filepath)


def build_datastack_archive(args, model_id, datastack_path):
    """Build an InVEST datastack from an arguments dict.

    Args:
        args (dict): The arguments dictionary to include in the datastack.
        model_id (string): The id the model these args are for. For core models,
            this is the regular id. For plugins, this has the format model_id@version
        datastack_path (string): The path to where the datastack archive
            should be written.

    Returns:
        ``None``
    """
    module = importlib.import_module(
        # For plugins, use the model id before the '@'
        name=models.model_id_to_pyname[model_id.split('@')[0]])

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

    spatial_types = {spec.SingleBandRasterInput, spec.VectorInput,
        spec.RasterOrVectorInput}
    file_based_types = spatial_types.union({
        spec.CSVInput, spec.FileInput, spec.DirectoryInput})
    rewritten_args = {}
    for key in args:
        # Allow the model to override specific arguments in datastack archive
        # prep.  This is useful for tables (like HRA) that are too complicated
        # to describe in the MODEL_SPEC format, but use a common specification
        # for the other args keys.
        override_funcname = f'_override_datastack_archive_{key}'
        if hasattr(module, override_funcname):
            LOGGER.debug(f'Using model override function for key {key}')
            # Notes about the override function:
            #   * Function may modify files_found
            #   * If this function copies data into the data dir, it _should_
            #     be within its own folder (e.g.
            #     {data_dir}/criteria_table_path_data/) to minimize chances of
            #     stomping on other data.  But this is up to the function to
            #     decide.
            #   * The override function is responsible for logging whatever is
            #     useful to include in the logfile.
            rewritten_args[key] = getattr(module, override_funcname)(
                args[key], data_dir, files_found)
            continue

        # We don't want to accidentally archive a user's complete workspace
        # directory, complete with prior runs there.
        if key == 'workspace_dir':
            LOGGER.debug(
                f"Skipping workspace directory: {args['workspace_dir']}")
            continue

        LOGGER.info(f'Starting to archive arg "{key}": {args[key]}')
        # Possible that a user might pass an args key that doesn't belong to
        # this model.  Skip if so.
        if key not in module.MODEL_SPEC.inputs:
            LOGGER.info(f'Skipping arg {key}; not in model MODEL_SPEC')

        input_spec = module.MODEL_SPEC.get_input(key)
        if type(input_spec) in file_based_types:
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

        if type(input_spec) is spec.CSVInput:
            # check the CSV for columns that may be spatial.
            # But also, the columns specification might not be listed, so don't
            # require that 'columns' exists in the MODEL_SPEC.
            spatial_columns = []
            if input_spec.columns:
                for col_spec in input_spec.columns:
                    if type(col_spec) in spatial_types:
                        spatial_columns.append(col_spec.id)

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

                dataframe = input_spec.get_validated_dataframe(source_path)
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
                            target_filepath = utils.copy_spatial_files(
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

        elif type(input_spec) is spec.FileInput:
            target_filepath = os.path.join(
                data_dir, f'{key}_file')
            shutil.copyfile(source_path, target_filepath)
            target_arg_value = target_filepath
            files_found[source_path] = target_arg_value

        elif type(input_spec)is spec.DirectoryInput:
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

        elif type(input_spec) in spatial_types:
            # Create a directory with a readable name, something like
            # "aoi_path_vector" or "lulc_cur_path_raster".
            spatial_dir = os.path.join(data_dir, f'{key}_{input_spec.type}')
            target_arg_value = utils.copy_spatial_files(
                source_path, spatial_dir)
            files_found[source_path] = target_arg_value

        else:
            LOGGER.debug(
                f"Type {type(input_spec)} is not filesystem-based; "
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
    parameter_set = build_parameter_set(
        rewritten_args, model_id, param_file_uri, relative=True)

    # write metadata for all files in args
    keywords = [module.MODEL_SPEC.model_id, 'InVEST']
    for k, v in args.items():
        if isinstance(v, str) and os.path.isfile(v):
            this_arg_spec = module.MODEL_SPEC.get_input(k)
            # write metadata file to target location (in temp dir)
            subdir = os.path.dirname(parameter_set['args'][k])
            target_location = os.path.join(temp_workspace, subdir)
            spec.write_metadata_file(v, this_arg_spec, keywords,
                                           out_workspace=target_location)

    # Remove the handler before archiving the working dir (and the logfile)
    archive_filehandler.close()
    logging.getLogger().removeHandler(archive_filehandler)

    # archive the workspace.
    with tempfile.TemporaryDirectory() as temp_dir:
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
    _tarfile_safe_extract(datastack_path, dest_dir_path)

    # get the arguments dictionary
    with open(os.path.join(
            dest_dir_path, DATASTACK_PARAMETER_FILENAME)) as datastack_file:
        arguments_dict = json.load(datastack_file)['args']

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


def build_parameter_set(args, model_id, paramset_path, relative=False):
    """Record a parameter set to a file on disk.

    Args:
        args (dict): The args dictionary to record to the parameter set.
        model_id (string): An identifier string for the callable or InVEST
            model that would accept the arguments given.
        paramset_path (string): The path to the file on disk where the
            parameters should be recorded.
        relative (bool): Whether to save the paths as relative.  If ``True``,
            The datastack assumes that paths are relative to the parent
            directory of ``paramset_path``.

    Returns:
        parameter dictionary saved in ``paramset_path``

    Raises:
        ValueError if creating a relative path fails.
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
                    try:
                        temp_rel_path = os.path.relpath(
                            normalized_path, os.path.dirname(paramset_path))
                    except ValueError:
                        # On Windows, ValueError is raised when ``path`` and
                        # ``start`` are on different drives
                        raise ValueError(
                            """Error: Cannot save datastack with relative
                            paths across drives. Choose a different save
                            location, or use absolute paths.""")
                    # Always save unix paths.
                    linux_style_path = temp_rel_path.replace('\\', '/')
                else:
                    # Always save unix paths.
                    linux_style_path = normalized_path.replace('\\', '/')

                return linux_style_path
        return args_param
    parameter_data = {
        'model_id': model_id,
        'args': _recurse(args)
    }
    with codecs.open(paramset_path, 'w', encoding='UTF-8') as paramset_file:
        paramset_file.write(
            json.dumps(parameter_data,
                       indent=4,
                       sort_keys=True))

    return parameter_data


def extract_parameter_set(paramset_path):
    """Extract and return attributes from a parameter set.

    Any string values found will have environment variables expanded.  See
    :py:ref:os.path.expandvars and :py:ref:os.path.expanduser for details.

    Args:
        paramset_path (string): The file containing a parameter set.

    Returns:
        A ``ParameterSet`` namedtuple with these attributes::

            args (dict): The arguments dict for the callable
            model_id (string): the ID of the model that these parameters are for
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

            # Don't expand or modify remote paths
            gdal_path = utils._GDALPath.from_uri(args_param)
            if not gdal_path.is_local:
                return args_param

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

    if 'model_id' in read_params:
        # New style datastacks include the model ID
        model_id = read_params['model_id']
    else:
        # Old style datastacks use the pyname (core models only, no plugins)
        model_id = models.pyname_to_model_id[read_params['model_name']]
    return ParameterSet(
        args=_recurse(read_params['args']),
        model_id=model_id)


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
        An instance of the ParameterSet namedtuple.

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
                # "Arguments for InVEST carbon 3.4.1rc1:\n"
                # (new style, using model id)
                # or
                # "Arguments for InVEST natcap.invest.carbon 3.4.1rc1:\n"
                # (old style, using model pyname)
                if line.startswith('Arguments for InVEST'):
                    identifier = line.split(' ')[3]
                    if identifier in models.pyname_to_model_id:
                        # Old style logfiles use the pyname
                        # These will be for core models only, not plugins
                        model_id = models.pyname_to_model_id[identifier]
                    else:
                        # New style logfiles use the model id
                        model_id = identifier
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
    return ParameterSet(args_dict, model_id)
