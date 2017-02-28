
import hashlib
import logging
import os
import platform

import numpy
from osgeo import gdal


LOGGER = logging.getLogger('natcap.testing.utils')

def digest_file_list(filepath_list, ifdir='skip'):
    """
    Create a single MD5sum from all the files in `filepath_list`.

    This is done by creating an MD5sum from the string MD5sums calculated
    from each individual file in the list. The filepath_list will be sorted
    before generating an MD5sum. If a given file is in this list multiple
    times, it will be double-counted.

    Note:
        When passed a list with a single file in it, this function will produce
        a different MD5sum than if you were to simply take the md5sum of that
        single file.  This is because this function produces an MD5sum of
        MD5sums.

    Parameters:
        filepath_list (list of strings): A list of files to analyze.
        ifdir (string): Either 'skip' or 'raise'.  Indicates what to do
            if a directory is encountered in this list.  If 'skip', the
            directory skipped will be logged.  If 'raise', IOError will
            be raised with the directory name.

    Returns:
        A string MD5sum generated for all of the files in the list.

    Raises:
        IOError: When a file in `filepath_list` is a directory and
            `ifdir == skip` or a file could not be found.
    """
    summary_md5 = hashlib.md5()
    for filepath in sorted(filepath_list):
        if os.path.isdir(filepath):
            # We only want to pass files down to the digest_file function
            message = 'Skipping md5sum for directory %s' % filepath
            if ifdir == 'skip':
                LOGGER.warn(message)
                continue
            else:  # ifdir == 'raise'
                raise IOError(message)
        summary_md5.update(digest_file(filepath))

    return summary_md5.hexdigest()


def digest_folder(folder):
    """
    Create a single MD5sum from all of the files in a folder.  This
    recurses through `folder` and will take the MD5sum of all files found
    within.

    Parameters:
        folder (string): A string path to a folder on disk.

    Note:
        When there is a single file within this folder, the return value
        of this function will be different than if you were to take the MD5sum
        of just that one file.  This is because we are taking an MD5sum of MD5sums.

    Returns:
        A string MD5sum generated from all of the files contained in
            this folder.
    """
    file_list = []
    for path, subpath, files in os.walk(folder):
        for name in files:
            file_list.append(os.path.join(path, name))

    return digest_file_list(file_list)


def digest_file(filepath):
    """
    Get the MD5sum for a single file on disk.  Files are read in
    a memory-efficient fashion.

    Args:
        filepath (string): a string path to the file or folder to be tested
            or a list of files to be analyzed.

    Returns:
        An md5sum of the input file
    """

    block_size = 2**20
    file_handler = open(filepath, 'rb')
    file_md5 = hashlib.md5()
    for chunk in iter(lambda: file_handler.read(block_size), ''):
        file_md5.update(chunk)
    file_handler.close()

    return file_md5.hexdigest()


def checksum_folder(workspace_uri, logfile_uri, style='GNU', ignore_exts=None):
    """Recurse through the workspace_uri and for every file in the workspace,
    record the filepath and md5sum to the logfile.  Additional environment
    metadata will also be recorded to help debug down the road.

    This output logfile will have two sections separated by a blank line.
    The first section will have relevant system information, with keys and
    values separated by '=' and some whitespace.

    This second section will identify the files we're snapshotting and the
    md5sums of these files, separated by '::' and some whitspace on each line.
    MD5sums are determined by calling `natcap.testing.utils.digest_file()`.

    Args:
        workspace_uri (string): A URI to the workspace to analyze
        logfile_uri (string): A URI to the logfile to which md5sums and paths
            will be recorded.
        style='GNU' (string): Either 'GNU' or 'BSD'.  Corresponds to the style
            of the output file.
        ignore_exts=None (list/set of strings or None): Extensions of files to
            ignore when checksumming.  If None, all files will be included.
            If a list or set of extension strings, any file with that extension
            will be skipped.

    Returns:
        Nothing.
    """

    format_styles = {
        'GNU': "{md5}  {filepath}\n",
        'BSD': "MD5 ({filepath}) = {md5}\n",
    }
    try:
        md5sum_string = format_styles[style]
    except KeyError:
        raise IOError('Invalid style: %s.  Valid styles: %s' % (
            style, format_styles.keys()))

    import pygeoprocessing
    logfile = open(logfile_uri, 'w')
    logfile.write('# orig_workspace = %s\n' % os.path.abspath(workspace_uri))
    logfile.write('# OS = %s\n' % platform.system())
    logfile.write('# plat_string = %s\n' % platform.platform())
    logfile.write('# GDAL = %s\n' % gdal.__version__)
    logfile.write('# numpy = %s\n' % numpy.__version__)
    logfile.write('# pygeoprocessing = %s\n' % pygeoprocessing.__version__)
    logfile.write('# checksum_style = %s\n' % style)

    if ignore_exts is None:
        ignore_exts = []

    for dirpath, _, filenames in os.walk(workspace_uri):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            # if the extension is in our set of extensions to ignore, skip it.
            if os.path.splitext(filepath)[-1] in ignore_exts:
                continue

            md5sum = digest_file(filepath)
            relative_filepath = filepath.replace(workspace_uri + os.sep, '')

            # Convert to unix path separators for all cases.
            if platform.system() == 'Windows':
                relative_filepath = relative_filepath.replace(os.sep, '/')

            logfile.write(md5sum_string.format(md5=md5sum,
                                        filepath=relative_filepath))
