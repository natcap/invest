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

import os
import json
import tarfile
import shutil
import logging
import tempfile
import codecs
import pprint
import collections
import re
import ast
import warnings

from osgeo import gdal
from osgeo import ogr

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


def _collect_spatial_files(filepath, data_dir):
    """Collect spatial files into the data directory of an archive.

    This function detects whether a filepath is a raster or vector
    recignizeable by GDAL/OGR and does what is needed to copy the dataset
    into the datastack's archive folder.

    Rasters copied into the archive will be stored in a new folder with the
    ``raster_`` prefix.  Vectors will be stored in a new folder with the
    ``vector_`` prefix.

    .. Note :: CSV files are not handled by this function.

        While the CSV format can be read as a vector format by OGR, we
        explicitly exclude CSV files from this function.  This is to maintain
        readibility of the final textfile.

    Parameters:
        filepath (string): The filepath to analyze.
        data_dir (string): The path to the data directory.

    Returns:
        ``None`` If the file is not a spatial file, or the ``path`` to the new
        resting place of the spatial files.
    """
    # If the user provides a mutli-part file, wrap it into a folder and grab
    # that instead of the individual file.

    with utils.capture_gdal_logging():
        raster = gdal.OpenEx(filepath, gdal.OF_RASTER)
        if raster is not None:
            new_path = tempfile.mkdtemp(prefix='raster_', dir=data_dir)
            driver = gdal.GetDriverByName('GTiff')
            LOGGER.info('[%s] Saving new raster to %s',
                        driver.LongName, new_path)
            # driver.CreateCopy returns None if there's an error
            # Common case: driver does not have Create() method implemented
            # ESRI Arc/Binary Grids are a great example of this.
            if not driver.CreateCopy(new_path, raster):
                LOGGER.info('Manually copying raster files to %s', new_path)
                new_files = []
                for filename in raster.GetFileList():
                    if os.path.isdir(filename):
                        # ESRI Arc/Binary grids include the parent folder in
                        # the list of all files in the dataset.
                        continue

                    new_filename = os.path.join(
                        new_path, os.path.basename(filename))
                    shutil.copyfile(filename, new_filename)
                    new_files.append(new_filename)

                # Pass the first file in the file list
                new_path = sorted(new_files)[0]

            driver = None
            raster = None
            return new_path

        vector = gdal.OpenEx(filepath, gdal.OF_VECTOR)
        if vector is not None:
            # OGR also reads CSVs; verify this IS actually a vector
            if vector.GetDriver().ShortName == 'CSV':
                vector = None
                return None

            new_path = tempfile.mkdtemp(prefix='vector_', dir=data_dir)
            driver = gdal.GetDriverByName('ESRI Shapefile')
            LOGGER.info('[%s] Saving new vector to %s',
                        driver.ShortName, new_path)
            new_vector = driver.CreateCopy(new_path, vector)

            # This is needed for copying GeoJSON files, and presumably other
            # formats as well.
            if not new_vector:
                new_path = os.path.join(new_path,
                                        os.path.basename(filepath))
                new_vector = driver.CreateCopy(new_path, vector)
            new_vector.FlushCache()
            driver = None
            vector = None
            return new_path
    return None


def _collect_filepath(path, data_dir):
    """Collect files on disk into the data directory of an archive.

    Parameters:
        path (string): The path to examine.  Must exist on disk.
        data_dir (string): The path to the data directory, where any data
            files will be stored.

    Returns:
        The path to the new filename within ``data_dir``.
    """
    # initialize the return_path
    multi_part_folder = _collect_spatial_files(path, data_dir)
    if multi_part_folder is not None:
        return multi_part_folder

    elif os.path.isfile(path):
        new_filename = os.path.join(data_dir,
                                    os.path.basename(path))
        shutil.copyfile(path, new_filename)
        return new_filename

    elif os.path.isdir(path):
        # path is a folder, so we want to copy the folder and all
        # its contents to the data dir.
        new_foldername = tempfile.mkdtemp(
            prefix='data_', dir=data_dir)
        for filename in os.listdir(path):
            src_path = os.path.join(path, filename)
            dest_path = os.path.join(new_foldername, filename)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copyfile(src_path, dest_path)
        return new_foldername


class _ArgsKeyFilter(logging.Filter):
    """A python logging filter for adding an args_key attribute to records."""

    def __init__(self, args_key):
        """Initialize the filter.

        Parameters:
            args_key (string): The args key to be added to all records.

        Returns:
            ``None``
        """
        logging.Filter.__init__(self)
        self.args_key = args_key

    def filter(self, record):
        """Filter a logging record.

        Adds the ``args_key`` attribute to the record from the ``args_key``
        that this filter was initialized with.

        Parameters:
            record (logging.Record): The log record.

        Returns:
            ``True``.  All log records will be passed through once modified.
        """
        record.args_key = self.args_key
        return True


def format_args_dict(args_dict, model_name):
    """Nicely format an arguments dictionary for writing to a stream.

    If printed to a console, the returned string will be aligned in two columns
    representing each key and value in the arg dict.  Keys are in ascending,
    sorted order.  Both columns are left-aligned.

    Parameters:
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

    Parameters:
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

    Parameters:
        args (dict): The arguments dictionary to include in the datastack.
        model_name (string): The python-importable module string of the model
            these args are for.
        datastack_path (string): The path to where the datastack archive
            should be written.

    Returns:
        ``None``
    """
    args = args.copy()
    temp_workspace = tempfile.mkdtemp(prefix='datastack_')
    logfile = os.path.join(temp_workspace, 'log')
    data_dir = os.path.join(temp_workspace, 'data')
    os.makedirs(data_dir)

    # For tracking existing files so we don't copy things twice
    files_found = {}
    LOGGER.debug('Keys: %s', sorted(args.keys()))

    def _recurse(args_param, handler, nested_key=None):
        if isinstance(args_param, dict):
            new_dict = {}
            for args_key, args_value in args_param.items():
                # log the key via a filter installed to the handler.
                if nested_key:
                    args_key_label = "%s['%s']" % (nested_key, args_key)
                else:
                    args_key_label = "args['%s']" % args_key

                args_key_filter = _ArgsKeyFilter(args_key_label)
                handler.addFilter(args_key_filter)
                if args_key not in ('workspace_dir',):
                    new_dict[args_key] = _recurse(args_value, handler,
                                                  nested_key=args_key_label)
                handler.removeFilter(args_key_filter)
            return new_dict
        elif isinstance(args_param, list):
            return [_recurse(list_item, handler) for list_item in args_param]
        elif isinstance(args_param, str):
            # If the parameter string is blank, return an empty string.
            if args_param.strip() == '':
                return ''

            # It's a string and exists on disk, it's a file!
            possible_path = os.path.normpath(args_param.replace('\\', os.sep))
            if os.path.exists(possible_path):
                try:
                    filepath = files_found[possible_path]
                    LOGGER.debug(('Parameter known from a previous '
                                  'entry: %s, using %s'),
                                 possible_path, filepath)
                    return filepath
                except KeyError:
                    found_filepath = _collect_filepath(possible_path,
                                                       data_dir)

                    # Store only linux-style filepaths.
                    relative_filepath = os.path.relpath(
                        found_filepath, temp_workspace).replace('\\', '/')
                    files_found[possible_path] = relative_filepath
                    LOGGER.debug('Processed path %s to %s',
                                 args_param, relative_filepath)
                    return relative_filepath
        # It's not a file or a structure to recurse through, so
        # just return the item verbatim.
        LOGGER.info('Using verbatim value: %s', args_param)
        return args_param

    log_format = "%(args_key)-25s %(name)-25s %(levelname)-8s %(message)s"
    with utils.log_to_file(logfile, log_fmt=log_format) as handler:
        new_args = {
            'args': _recurse(args, handler),
            'model_name': model_name,
            'invest_version': __version__
        }

    LOGGER.debug('found files: \n%s', pprint.pformat(files_found))
    LOGGER.debug('new arguments: \n%s', pprint.pformat(new_args))
    # write parameters to a new json file in the temp workspace
    param_file_uri = os.path.join(temp_workspace,
                                  'parameters' + PARAMETER_SET_EXTENSION)
    with codecs.open(param_file_uri, 'w', encoding='UTF-8') as params:
        params.write(json.dumps(new_args,
                                indent=4,
                                sort_keys=True))

    # archive the workspace.
    with utils.sandbox_tempdir() as temp_dir:
        temp_archive = os.path.join(temp_dir, 'invest_archive')
        archive_name = shutil.make_archive(
            temp_archive, 'gztar', root_dir=temp_workspace,
            logger=LOGGER, verbose=True)
        shutil.move(archive_name, datastack_path)


def extract_datastack_archive(datastack_path, dest_dir_path):
    """Extract a datastack to a given folder.

    Parameters:
        datastack_path (string): The path to a datastack archive on disk.
        dest_dir_path (string): The path to a directory.  The contents of the
            demonstration datastack archive will be extracted into this
            directory. If the directory does not exist, it will be created.

    Returns:
        ``args`` (dict): A dictionary of arguments from the extracted
            archive.  Paths to files are absolute paths.
    """
    LOGGER.info('Extracting archive %s to %s', datastack_path, dest_dir_path)
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
            data_path = os.path.normpath(os.path.join(dest_dir_path,
                                                      args_param))
            if os.path.exists(data_path):
                return os.path.normpath(data_path)
        return args_param

    new_args = _rewrite_paths(arguments_dict)
    LOGGER.debug('Expanded parameters as \n%s', pprint.pformat(new_args))
    return new_args


def build_parameter_set(args, model_name, paramset_path, relative=False):
    """Record a parameter set to a file on disk.

    Parameters:
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
            possible_path = args_param.replace('\\', '/')
            if os.path.exists(possible_path):
                # Always save unix paths.
                normalized_path = os.path.normpath(possible_path)
                if relative:
                    # Handle special case where python assumes that '.'
                    # represents the CWD
                    if (normalized_path == '.' or
                            os.path.dirname(paramset_path) == normalized_path):
                        return '.'
                    return os.path.relpath(normalized_path,
                                           os.path.dirname(paramset_path))
                return normalized_path
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

    Parameters:
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

    Parameters:
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

            Parameters:
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
