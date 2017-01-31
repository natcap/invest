"""Functions for creating and extracting InVEST demonstration scenarios.

A demonstration scenario for InVEST is a compressed archive that includes the
arguments for a model, all of the data files referenced by the arguments, and
a logfile with some extra information about how the archive was created.  The
resulting archive can then be extracted on a different computer and should
have all of the information it needs to run an InVEST model in its entirity.

"""

import contextlib
import os
import json
import tarfile
import shutil
import logging
import tempfile
import codecs
import pprint

from osgeo import gdal
from osgeo import ogr

from . import utils


LOGGER = logging.getLogger(__name__)

#: This is the template string for writing a primitive OGR VRT vector with a
#: based on a preexisting vector.  To use, be sure to use the ``format``
#: method to set the ``src_layer`` and ``src_vector`` strings.  ``src_vector``
#: is assumed to be relative to the VRT file's location, wherever that's
#: written.
OGRVRTTEMPLATE = """<OGRVRTDataSource>
    <OGRVRTLayer name="{src_layer}">
        <SrcDataSource relativeToVRT="1">{src_vector}</SrcDataSource>
    </OGRVRTLayer>
</OGRVRTDataSource>"""


@contextlib.contextmanager
def log_to_file(logfile):
    """Log all messages within this context to a file.

    Parameters:
        logfile (string): The path to where the logfile will be written.
            If there is already a file at this location, it will be
            overwritten.

    Yields:
        ``handler``: An instance of ``logging.FileHandler`` that
            represents the file that is being written to.

    Returns:
        ``None``"""
    handler = logging.FileHandler(logfile, 'w', encoding='UTF-8')
    formatter = logging.Formatter(
        "%(args_key)-25s %(name)-25s %(levelname)-8s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)  # capture everything
    root_logger.addHandler(handler)
    handler.setFormatter(formatter)
    yield handler
    handler.close()
    root_logger.removeHandler(handler)


@contextlib.contextmanager
def sandbox_tempdir(suffix='', prefix='tmp', dir=None):
    """Create a temporary directory for this context and clean it up on exit.

    Parameters are identical to those for :py:func:`tempfile.mkdtemp`.

    When the context manager exits, the created temporary directory is
    recursively removed.

    Parameters:
        suffix='' (string): a suffix for the name of the directory.
        prefix='tmp' (string): the prefix to use for the directory name.
        dir=None (string or None): If a string, a directory that should be
            the parent directory of the new temporary directory.  If None,
            tempfile will determine the appropriate tempdir to use as the
            parent folder.

    Yields:
        ``sandbox`` (string): The path to the new folder on disk.

    Returns:
        ``None``"""
    sandbox = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        yield sandbox
    finally:
        try:
            shutil.rmtree(sandbox)
        except OSError:
            LOGGER.exception('Could not remove sandbox %s', sandbox)


def _collect_spatial_files(filepath, data_dir, link_data, archive_path):
    # If the user provides a mutli-part file, wrap it into a folder and grab
    # that instead of the individual file.

    with utils.capture_gdal_logging():
        raster = gdal.Open(filepath)
        if raster is not None:
            new_path = tempfile.mkdtemp(prefix='raster_', dir=data_dir)
            if link_data:
                raster_files = raster.GetFileList()
                for raster_file in sorted(raster_files):
                    new_filename = os.path.join(
                        new_path, os.path.basename(raster_file))
                    LOGGER.info('Symlinking %s --> %s',
                                raster_file, new_filename)
                    os.symlink(raster_file, new_filename)

                # pick a file to return as the filename
                return os.path.join(new_path,
                                    os.path.basename(raster_files[0]))

            driver = raster.GetDriver()
            LOGGER.info('[%s] Saving new raster to %s',
                        driver.LongName, new_path)
            # driver.CreateCopy returns None if there's an error
            # Common case: driver does not have Create() method implemented
            # ESRI Arc/Binary Grids are a great example of this.
            if not driver.CreateCopy(new_path, raster, strict=1):
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
            if driver.GetName() == 'CSV':
                driver = None
                vector = None
                return None

            new_path = tempfile.mkdtemp(prefix='vector_', dir=data_dir)
            LOGGER.info('[%s] Saving new vector to %s',
                        driver.GetName(), new_path)
            if link_data:
                vrt_vector_path = os.path.join(new_path, 'linked_vector.vrt')
                with codecs.open(vrt_vector_path,
                                 'w', encoding='utf-8') as vrt:
                    vrt.write(OGRVRTTEMPLATE.format(
                        src_vector=os.path.relpath(
                            filepath,
                            os.path.join(os.path.dirname(archive_path),
                                         'extracted_path',  # placeholder
                                         'data', os.path.basename(new_path))),
                        src_layer=vector.GetLayer().GetName()
                    ))
                driver = None
                vector = None
                return vrt_vector_path

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


def _collect_filepath(parameter, data_dir, link_data, archive_path):
    # initialize the return_path
    multi_part_folder = _collect_spatial_files(parameter, data_dir, link_data,
                                               archive_path)
    if multi_part_folder is not None:
        LOGGER.debug('%s is a multi-part file', parameter)
        return multi_part_folder

    elif os.path.isfile(parameter):
        LOGGER.debug('%s is a single file', parameter)
        new_filename = os.path.join(data_dir,
                                    os.path.basename(parameter))
        if link_data:
            relative_parameter = os.path.relpath(
                parameter, os.path.join(os.path.dirname(archive_path),
                                        'extracted_archive',
                                        'data'))
            os.symlink(relative_parameter, new_filename)
            LOGGER.debug('Symlinking %s against %s as %s', parameter,
                         archive_path, relative_parameter)
        else:
            shutil.copyfile(parameter, new_filename)
        return new_filename

    elif os.path.isdir(parameter):
        LOGGER.debug('%s is a directory', parameter)
        # parameter is a folder, so we want to copy the folder and all
        # its contents to the data dir.
        new_foldername = tempfile.mkdtemp(
            prefix='data_', dir=data_dir)
        for filename in os.listdir(parameter):
            src_path = os.path.join(parameter, filename)
            dest_path = os.path.join(new_foldername, filename)
            if link_data:
                os.symlink(src_path, dest_path)
            else:
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copyfile(src_path, dest_path)
        return new_foldername


class _ArgsKeyFilter(logging.Filter):
    def __init__(self, args_key):
        logging.Filter.__init__(self)
        self.args_key = args_key

    def filter(self, record):
        record.args_key = self.args_key
        return True


def build_scenario(args, scenario_path, link_data=False):
    """Collect an InVEST model's arguments into a dictionary and archive all
        the input data.

        args - a dictionary of arguments
        scenario_path - a URI to the target archive.

        Returns nothing."""

    args = args.copy()
    temp_workspace = tempfile.mkdtemp(prefix='scenario_')
    logfile = os.path.join(temp_workspace, 'log')
    data_dir = os.path.join(temp_workspace, 'data')
    os.makedirs(data_dir)

    # For tracking existing files so we don't copy things twice
    files_found = {}

    # Recurse through the args parameters to locate any URIs
    #   If a URI is found, copy that file to a new location in the temp
    #   workspace and update the URI reference.
    #   Duplicate URIs should also have the same replacement URI.
    #
    # If a workspace or suffix is provided, ignore that key.
    LOGGER.debug('Keys: %s', sorted(args.keys()))

    def _recurse(args_param, handler, nested_key=None):
        if isinstance(args_param, dict):
            new_dict = {}
            for args_key, args_value in args_param.iteritems():
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
        elif isinstance(args_param, basestring):
            # It's a string and exists on disk, it's a file!
            possible_path = os.path.abspath(args_param)
            if os.path.exists(possible_path):
                try:
                    filepath = files_found[possible_path]
                    LOGGER.debug(('Parameter known from a previous '
                                  'entry: %s, using %s'),
                                 possible_path, filepath)
                    return filepath
                except KeyError:
                    found_filepath = _collect_filepath(possible_path,
                                                       data_dir,
                                                       link_data,
                                                       scenario_path)
                    relative_filepath = os.path.relpath(
                        found_filepath, temp_workspace)
                    files_found[possible_path] = relative_filepath
                    LOGGER.debug('Processed path %s to %s',
                                 args_param, relative_filepath)
                    return relative_filepath
        # It's not a file or a structure to recurse through, so
        # just return the item verbatim.
        LOGGER.info('Using verbatim value: %s', args_param)
        return args_param

    with log_to_file(logfile) as handler:
        LOGGER.info('Data are symlinked: %s', link_data,
                    extra={'args_key': ''})
        new_args = _recurse(args, handler)
    LOGGER.debug('found files: \n%s', pprint.pformat(files_found))

    LOGGER.debug('new arguments: \n%s', pprint.pformat(new_args))
    # write parameters to a new json file in the temp workspace
    param_file_uri = os.path.join(temp_workspace, 'parameters.json')
    with codecs.open(param_file_uri, 'w', encoding='UTF-8') as params:
        params.write(json.dumps(new_args,
                                encoding='UTF-8',
                                indent=4,
                                sort_keys=True))

    # archive the workspace.
    with sandbox_tempdir() as temp_dir:
        temp_archive = os.path.join(temp_dir, 'invest_archive')
        archive_name = shutil.make_archive(
            temp_archive, 'gztar', root_dir=temp_workspace,
            logger=LOGGER, verbose=True)
        shutil.move(archive_name, scenario_path)


def extract_scenario(scenario_path, dest_dir_path):
    """Extract a demonstration scenario to a given folder.

    Parameters:
        scenario_path (string): The path to a demonstration scenario archive
            on disk.
        dest_dir_path (string): The path to a directory.  The contents of the
            demonstration scenario archive will be extracted into this
            directory. If the directory does not exist, it will be created.

    Returns:
        ``args`` (dict): A dictionary of arguments from the extracted
            archive"""
    LOGGER.info('Extracting archive %s to %s', scenario_path, dest_dir_path)
    # extract the archive to the workspace
    with tarfile.open(scenario_path) as tar:
        tar.extractall(dest_dir_path)

    # get the arguments dictionary
    arguments_dict = json.load(open(
        os.path.join(dest_dir_path, 'parameters.json')))

    def _recurse(args_param):
        if isinstance(args_param, dict):
            _args = {}
            for key, value in args_param.iteritems():
                _args[key] = _recurse(value)
            return _args
        elif isinstance(args_param, list):
            return [_recurse(param) for param in args_param]
        elif isinstance(args_param, basestring):
            data_path = os.path.join(dest_dir_path, args_param)
            LOGGER.info('Recursing with args param: %s --> %s', args_param,
                        data_path)
            if os.path.exists(data_path):
                return os.path.normpath(data_path)
            else:
                LOGGER.info('Path not found: %s, (Normalized: %s)',
                            data_path, os.path.normpath(data_path))
        return args_param

    new_args = _recurse(arguments_dict)
    LOGGER.debug('Expanded parameters as \n%s', pprint.pformat(new_args))
    return new_args
