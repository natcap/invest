"""InVEST specific code utils."""
import math
import os

import numpy
from osgeo import gdal
from osgeo import osr
import pygeoprocessing


def make_suffix_string(args, suffix_key):
    """Make an InVEST appropriate suffix string.

    Creates an InVEST appropriate suffix string  given the args dictionary and
    suffix key.  In general, prepends an '_' when necessary and generates an
    empty string when necessary.

    Parameters:
        args (dict): the classic InVEST model parameter dictionary that is
            passed to `execute`.
        suffix_key (string): the key used to index the base suffix.

    Returns:
        If `suffix_key` is not in `args`, or `args['suffix_key']` is ""
            return "",
        If `args['suffix_key']` starts with '_' return `args['suffix_key']`
            else return '_'+`args['suffix_key']`
    """
    try:
        file_suffix = args[suffix_key]
        if file_suffix != "" and not file_suffix.startswith('_'):
            file_suffix = '_' + file_suffix
    except KeyError:
        file_suffix = ''

    return file_suffix


def exponential_decay_kernel_raster(expected_distance, kernel_filepath):
    """Create a raster-based exponential decay kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Parameters:
        expected_distance (int or float): The distance (in pixels) of the
            kernel's radius, the distance at which the value of the decay
            function is equal to `1/e`.
        kernel_filepath (string): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.

    Returns:
        None
    """
    max_distance = expected_distance * 5
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_filepath.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS('NAD27')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    integration = 0.0
    for row_block_index in xrange(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in xrange(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = numpy.indices((row_block_width,
                                                      col_block_width),
                                                     dtype=numpy.float)

            row_indices += numpy.float(row_offset - max_distance)
            col_indices += numpy.float(col_offset - max_distance)

            kernel_index_distances = numpy.hypot(
                row_indices, col_indices)
            kernel = numpy.where(
                kernel_index_distances > max_distance, 0.0,
                numpy.exp(-kernel_index_distances / expected_distance))
            integration += numpy.sum(kernel)

            kernel_band.WriteArray(kernel, xoff=col_offset,
                                   yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_dataset.FlushCache()

    for block_data, kernel_block in pygeoprocessing.iterblocks(
            kernel_filepath):
        kernel_block /= integration
        kernel_band.WriteArray(kernel_block, xoff=block_data['xoff'],
                               yoff=block_data['yoff'])


def build_file_registry(base_file_path_list, file_suffix):
    """Combine file suffixes with key names, base filenames, and directories.

    Parameters:
        base_file_tuple_list (list): a list of (dict, path) tuples where
            the dictionaries have a 'file_key': 'basefilename' pair, or
            'file_key': list of 'basefilename's.  'path'
            indicates the file directory path to prepend to the basefile name.
        file_suffix (string): a string to append to every filename, can be
            empty string

    Returns:
        dictionary of 'file_keys' from the dictionaries in
        `base_file_tuple_list` mapping to full file paths with suffixes or
        lists of file paths with suffixes depending on the original type of
        the 'basefilename' pair.

    Raises:
        ValueError if there are duplicate file keys or duplicate file paths.
        ValueError if a path is not a string or a list of strings.
    """
    all_paths = set()
    duplicate_keys = set()
    duplicate_paths = set()
    f_reg = {}

    def _build_path(base_filename, path):
        """Internal helper to avoid code duplication."""
        pre, post = os.path.splitext(base_filename)
        full_path = os.path.join(path, pre+file_suffix+post)

        # Check for duplicate keys or paths
        if full_path in all_paths:
            duplicate_paths.add(full_path)
        else:
            all_paths.add(full_path)
        return full_path

    for base_file_dict, path in base_file_path_list:
        for file_key, file_payload in base_file_dict.iteritems():
            # check for duplicate keys
            if file_key in f_reg:
                duplicate_keys.add(file_key)
            else:
                # handle the case whether it's a filename or a list of strings
                if isinstance(file_payload, basestring):
                    full_path = _build_path(file_payload, path)
                    f_reg[file_key] = full_path
                elif isinstance(file_payload, list):
                    f_reg[file_key] = []
                    for filename in file_payload:
                        full_path = _build_path(filename, path)
                        f_reg[file_key].append(full_path)
                else:
                    raise ValueError(
                        "Unknown type in base_file_dict[%s]=%s" % (
                            file_key, path))

    if len(duplicate_paths) > 0 or len(duplicate_keys):
        raise ValueError(
            "Cannot consolidate because of duplicate paths or keys: "
            "duplicate_keys: %s duplicate_paths: %s" % (
                duplicate_keys, duplicate_paths))

    return f_reg
