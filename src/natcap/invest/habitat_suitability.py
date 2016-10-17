"""Habitat suitability model."""
import os
import logging

import numpy
from osgeo import gdal
import pygeoprocessing.geoprocessing

from . import utils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.habitat_suitability')
_HSI_NODATA = -1.0  # HSI values are floats in [0..1] so -1 as nodata is safe


def execute(args):
    """Habitat Suitability.

    Calculate habitat suitability indexes given biophysical parameters.

    The objective of a habitat suitability index (HSI) is to help users
    identify areas within their AOI that would be most suitable for habitat
    restoration.  The output is a gridded map of the user's AOI in which each
    grid cell is assigned a suitability rank between 0 (not suitable) and 1
    (most suitable).  The suitability rank is generally calculated as the
    weighted geometric mean of several individual input criteria, which have
    also been ranked by suitability from 0-1.  Habitat types (e.g. marsh,
    mangrove, coral, etc.) are treated separately, and each habitat type will
    have a unique set of relevant input criteria and a resultant habitat
    suitability map.

    Parameters:
        args['workspace_dir'] (string): directory path to workspace directory
            for output files.
        args['results_suffix'] (string): (optional) string to append to any
            output file names.
        args['aoi_path'] (string): file path to an area of interest shapefile.
        args['exclusion_path_list'] (list): (optional) a list of file paths to
            shapefiles which define areas which the HSI should be masked out
            in a final output.
        args['output_cell_size'] (float): (optional) size of output cells.
            If not present, the output size will snap to the smallest cell
            size in the HSI range rasters.
        args['habitat_threshold'] (float): a value to threshold the habitat
            score values to 0 and 1.
        args['hsi_ranges'] (dict): a dictionary that describes the habitat
            biophysical base rasters as well as the ranges for optimal and
            tolerable values.  Each biophysical value has a unique key in the
            dictionary that is used to name the mapping of biophysical to
            local HSI value.  Each value is dictionary with keys:

                * 'raster_path': path to disk for biophysical raster.
                * 'range': a 4-tuple in non-decreasing order describing
                  the "tolerable" to "optimal" ranges for those biophysical
                  values.  The endpoints non-inclusively define where the
                  suitability score is 0.0, the two midpoints inclusively
                  define the range where the suitability is 1.0, and the
                  ranges above and below are linearly interpolated between
                  0.0 and 1.0.

                Example::

                    {
                        'depth':
                            {
                                'raster_path': r'C:/path/to/depth.tif',
                                'range': (-50, -30, -10, -10),
                            },
                        'temperature':
                            {
                                'temperature_path': (
                                    r'C:/path/to/temperature.tif'),
                                'range': (5, 7, 12.5, 16),
                            }
                    }

        args['categorical_geometry'] (dict): a dictionary that describes
            categorical vector geometry that directly defines the HSI values.
            The dictionary specifies paths to the vectors and the fieldname
            that provides the raw HSI values with keys:
                'vector_path': a path to disk for the vector coverage polygon
                'fieldname': a string matching a field in the vector polygon
                    with HSI values.

            Example::

                {
                    'categorical_geometry': {
                        'substrate': {
                            'vector_path': r'C:/path/to/Substrate.shp',
                            'fieldname': 'Suitabilit',
                        }
                    }
                }

    Returns:
        None
    """
    _output_base_files = {
        'suitability_path': 'hsi.tif',
        'threshold_suitability_path': 'hsi_threshold.tif',
        'screened_suitability_path': 'hsi_threshold_screened.tif',
        }

    _tmp_base_files = {
        'screened_mask_path': 'screened_mask.tif',
        }

    LOGGER.info('Creating output directories and file registry.')
    output_dir = os.path.join(args['workspace_dir'])
    pygeoprocessing.create_directories([output_dir])

    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    f_reg = utils.build_file_registry(
        [(_output_base_files, output_dir),
         (_tmp_base_files, output_dir)], file_suffix)

    # determine the minimum cell size
    if 'output_cell_size' in args:
        output_cell_size = args['output_cell_size']
    else:
        # cell size is the min cell size of all the biophysical inputs
        output_cell_size = min(
            [pygeoprocessing.get_cell_size_from_uri(entry['raster_path'])
             for entry in args['hsi_ranges'].itervalues()])

    LOGGER.info("Aligning base biophysical raster list.")
    aligned_raster_stack = {}
    out_aligned_raster_list = []
    base_raster_list = []
    for key, entry in args['hsi_ranges'].iteritems():
        # create a new name for aligned raster and register with both the
        # temporary base file and regular registry so it can be deleted later
        aligned_path = os.path.join(output_dir, key + file_suffix + '.tif')
        aligned_raster_stack[key] = aligned_path

        # sanity check to ensure key is not already defined in the registry
        if key in f_reg:
            raise ValueError('%s key already defined in f_reg' % key)
        f_reg[key] = aligned_path
        _tmp_base_files[key] = f_reg[key]
        out_aligned_raster_list.append(aligned_path)
        base_raster_list.append(entry['raster_path'])

    tmp_categorical_files = set()
    for key, entry in args['categorical_geometry'].iteritems():
        # create a new name for aligned raster and register with both the
        # temporary base file and regular registry so it can be deleted later
        aligned_path = os.path.join(output_dir, key + file_suffix + '.tif')
        aligned_raster_stack[key] = aligned_path
        base_path = aligned_path + '_base.tif'

        pygeoprocessing.create_raster_from_vector_extents_uri(
            entry['vector_path'], output_cell_size, gdal.GDT_Float32,
            _HSI_NODATA, base_path)

        # sanity check to ensure key is not already defined in the registry
        if key in f_reg:
            raise ValueError('%s key already defined in f_reg' % key)
        f_reg[key] = aligned_path
        tmp_categorical_files.add(base_path)
        out_aligned_raster_list.append(aligned_path)
        base_raster_list.append(base_path)

    pygeoprocessing.geoprocessing.align_dataset_list(
        base_raster_list, out_aligned_raster_list,
        ['nearest'] * len(base_raster_list),
        output_cell_size, 'intersection', 0, aoi_uri=args['aoi_path'])

    for filename in tmp_categorical_files:
        try:
            os.remove(filename)
        except OSError:
            LOGGER.warn("Can't remove temporary file %s", filename)

    # map biophysical to individual habitat suitability index
    LOGGER.info('Starting biophysical to HSI mapping.')
    base_nodata = None
    suitability_range = None
    suitability_raster_list = []
    for key, entry in args['hsi_ranges'].iteritems():
        LOGGER.info("Mapping biophysical to HSI on %s", key)
        base_raster_path = aligned_raster_stack[key]
        base_nodata = pygeoprocessing.get_nodata_from_uri(base_raster_path)
        suitability_range = entry['suitability_range']
        suitability_key = key+'_suitability%s' % file_suffix

        # sanity check that we aren't overwriting an f_reg entry
        if suitability_key in f_reg:
            raise ValueError(
                '%s key already defined in f_reg' % suitability_key)

        f_reg[suitability_key] = os.path.join(
            output_dir, suitability_key + '.tif')
        suitability_raster_list.append(f_reg[suitability_key])

        def local_map(biophysical_values):
            """Map biophysical values to suitability index values."""
            # the following condition and function lists capture the following
            # ranges in order:
            #   1) range[0] to range[1] noninclusive (linear interp 0-1)
            #   2) range[1] to range[2] inclusive (exactly 1.0)
            #   3) range[2] to range[3] noninclusive (linear interp 1-0)
            #   4) nodata -> nodata
            #   5) 0.0 everywhere else

            condlist = [
                (suitability_range[0] < biophysical_values) &
                (biophysical_values < suitability_range[1]),
                (suitability_range[1] <= biophysical_values) &
                (biophysical_values <= suitability_range[2]),
                (suitability_range[2] < biophysical_values) &
                (biophysical_values < suitability_range[3]),
                biophysical_values == base_nodata]
            funclist = [
                lambda x: (
                    (x-suitability_range[0]) /
                    (suitability_range[1]-suitability_range[0])),
                1.0,
                lambda x: 1.0 - (
                    (x-suitability_range[2]) /
                    (suitability_range[3]-suitability_range[2])),
                _HSI_NODATA,
                0.0]
            # The .astype(numpy.float) is necessary for potential int types
            # that cast vector floating point division to ints
            return numpy.piecewise(
                biophysical_values.astype(numpy.float), condlist, funclist)

        pygeoprocessing.vectorize_datasets(
            [base_raster_path], local_map, f_reg[suitability_key],
            gdal.GDT_Float32, _HSI_NODATA, output_cell_size,
            'intersection', vectorize_op=False)

    # calculate categorical raster
    for key, entry in args['categorical_geometry'].iteritems():
        LOGGER.info("Rasterizing categorical geometry on %s", key)
        base_raster_path = aligned_raster_stack[key]
        base_nodata = pygeoprocessing.get_nodata_from_uri(base_raster_path)
        pygeoprocessing.rasterize_layer_uri(
            base_raster_path, entry['vector_path'],
            option_list=["ATTRIBUTE=%s" % entry['fieldname']])
        suitability_raster_list.append(base_raster_path)

    # calculate geometric mean
    LOGGER.info("Calculate geometric mean of HSIs.")

    def geo_mean_op(*suitability_values):
        """Geometric mean of input suitability_values."""
        running_product = suitability_values[0].astype(numpy.float32)
        running_mask = suitability_values[0] == _HSI_NODATA
        for index in range(1, len(suitability_values)):
            running_product *= suitability_values[index]
            running_mask = running_mask | (
                suitability_values[index] == _HSI_NODATA)
        result = numpy.empty(running_mask.shape, dtype=numpy.float32)
        result[:] = _HSI_NODATA
        result[~running_mask] = (
            running_product[~running_mask]**(1./len(suitability_values)))
        return result

    pygeoprocessing.geoprocessing.vectorize_datasets(
        suitability_raster_list, geo_mean_op, f_reg['suitability_path'],
        gdal.GDT_Float32, _HSI_NODATA, output_cell_size, "intersection",
        dataset_to_align_index=0, vectorize_op=False)

    LOGGER.info(
        "Masking HSI to threshold value of %s", args['habitat_threshold'])

    def threshold_op(hsi_values):
        """Threshold HSI values to user defined value."""
        result = hsi_values[:]
        invalid_mask = (
            (hsi_values == _HSI_NODATA) |
            (hsi_values < args['habitat_threshold']))
        result[invalid_mask] = _HSI_NODATA
        return result

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [f_reg['suitability_path']], threshold_op,
        f_reg['threshold_suitability_path'], gdal.GDT_Float32,
        _HSI_NODATA, output_cell_size, "intersection", vectorize_op=False)

    LOGGER.info("Masking threshold by exclusions.")
    pygeoprocessing.new_raster_from_base_uri(
        f_reg['threshold_suitability_path'],
        f_reg['screened_mask_path'], 'GTiff', _HSI_NODATA,
        gdal.GDT_Byte, fill_value=0)
    if 'exclusion_path_list' in args:
        for exclusion_mask_path in args['exclusion_path_list']:
            LOGGER.info("Building raster mask for %s", exclusion_mask_path)
            pygeoprocessing.rasterize_layer_uri(
                f_reg['screened_mask_path'], exclusion_mask_path,
                burn_values=[1])

    def mask_exclusion_op(base_values, mask_values):
        """Mask the base values to nodata where mask == 1."""
        result = base_values[:]
        result[mask_values == 1] = _HSI_NODATA
        return result

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [f_reg['threshold_suitability_path'],
         f_reg['screened_mask_path']], mask_exclusion_op,
        f_reg['screened_suitability_path'], gdal.GDT_Float32, _HSI_NODATA,
        output_cell_size, "intersection", vectorize_op=False)

    LOGGER.info('Removing temporary files.')
    for tmp_filename_key in _tmp_base_files:
        try:
            os.remove(f_reg[tmp_filename_key])
        except OSError as os_error:
            LOGGER.warn(
                "Can't remove temporary file: %s\nOriginal Exception:\n%s",
                f_reg[tmp_filename_key], os_error)
