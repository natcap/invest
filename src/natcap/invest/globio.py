"""GLOBIO InVEST Model."""
import os
import logging
import collections
import tempfile

import pandas
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import pygeoprocessing
import taskgraph

from . import utils
from . import validation


LOGGER = logging.getLogger(__name__)

# this value of sigma == 9.0 was derived by Justin Johnson as a good
# approximation to use as a gaussian filter to replace the connectivity index.
# I don't have any other documentation than his original code base.
SIGMA = 9.0

ARGS_SPEC = {
    "model_name": "GLOBIO",
    "module": __name__,
    "userguide_html": "../documentation/globio.html",
    "args_with_spatial_overlap": {
        "spatial_keys": [
            "lulc_path", "pasture_path", "potential_vegetation_path",
            "aoi_path", "globio_lulc_path"],
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "predefined_globio": {
            "type": "boolean",
            "required": False,
            "about": "if True then \"mode (b)\" else \"mode (a)\"",
            "name": "Predefined land use map for GLOBIO"
        },
        "lulc_path": {
            "type": "raster",
            "validation_options": {
                "projected": True,
            },
            "required": "not predefined_globio",
            "about": "used in \"mode (a)\" path to a base landcover map with integer codes",
            "name": "Land Use/Cover (Raster)"
        },
        "lulc_to_globio_table_path": {
            "validation_options": {
                "required_fields": ["lucode", "globio_lucode"],
            },
            "type": "csv",
            "required": "not predefined_globio",
            "about": (
                "A CSV table containing model information corresponding to "
                "each of the land use classes in the LULC raster input.  It "
                "must contain the fields 'lucode', 'globio_lucode'.  "
                "See the InVEST User's Guide for more information "
                "about these fields."),
            "name": "Landcover to GLOBIO Landcover Table"
        },
        "infrastructure_dir": {
            "validation_options": {
                "exists": True,
            },
            "type": "directory",
            "required": True,
            "about": (
                "Used in \"mode (a) and (b)\" a path to a folder containing "
                "maps of either GDAL compatible rasters or vectors. "
                "These data will be used in the infrastructure "
                "to calculation of MSA."),
            "name": "Infrastructure Directory"
        },
        "pasture_path": {
            "type": "raster",
            "validation_options": {
                "projected": True,
            },
            "required": "not predefined_globio",
            "about": "used in \"mode (a)\" path to pasture raster",
            "name": "Pasture (Raster)"
        },
        "potential_vegetation_path": {
            "type": "raster",
            "validation_options": {
                "projected": True,
            },
            "required": "not predefined_globio",
            "about": "used in \"mode (a)\" path to potential vegetation raster",
            "name": "Potential Vegetation (Raster)"
        },
        "pasture_threshold": {
            "validation_options": {
                "expression": "(value >= 0) & (value <= 1)",
            },
            "type": "number",
            "required": "not predefined_globio",
            "about": "used in \"mode (a)\"",
            "name": "Pasture Threshold"
        },
        "intensification_fraction": {
            "validation_options": {
                "expression": "(value >= 0) & (value <= 1)",
            },
            "type": "number",
            "required": True,
            "about": (
                "A value between 0 and 1 denoting proportion of total "
                "agriculture that should be classified as 'high input'."),
            "name": "Proportion of of Agriculture Intensified"
        },
        "primary_threshold": {
            "validation_options": {
                "expression": "(value >= 0) & (value <= 1)",
            },
            "type": "number",
            "required": "not predefined_globio",
            "about": "used in \"mode (a)\"",
            "name": "Primary Threshold"
        },
        "msa_parameters_path": {
            "validation_options": {
                "required_fields": [
                    "MSA_type", "measurement", "value", "msa_x", "se"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "A CSV table containing MSA threshold values as defined in "
                "the user's guide.  Provided for advanced users that may "
                "wish to change those values."),
            "name": "MSA Parameter Table"
        },
        "aoi_path": {
            "type": "vector",
            "validation_options": {
                "projected": True,
            },
            "required": False,
            "about": (
                "This is a set of polygons that can be used to aggregate MSA "
                "sum and mean to a polygon."),
            "name": "AOI",
        },
        "globio_lulc_path": {
            "validation_options": {
                "projected": True,
            },
            "type": "raster",
            "required": "predefined_globio",
            "about": "used in \"mode (b)\" path to predefined globio raster.",
            "name": "GLOBIO Classified Land Use"
        }
    }
}


def execute(args):
    """GLOBIO.

    The model operates in two modes.  Mode (a) generates a landcover map
    based on a base landcover map and information about crop yields,
    infrastructure, and more.  Mode (b) assumes the globio landcover
    map is generated.  These modes are used below to describe input
    parameters.

    Parameters:

        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['predefined_globio'] (boolean): if True then "mode (b)" else
            "mode (a)"
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['lulc_path'] (string): used in "mode (a)" path to a base landcover
            map with integer codes
        args['lulc_to_globio_table_path'] (string): used in "mode (a)" path to
            table that translates the land-cover args['lulc_path'] to
            intermediate GLOBIO classes, from which they will be further
            differentiated using the additional data in the model.  Contains
            at least the following fields:

            * 'lucode': Land use and land cover class code of the dataset
              used. LULC codes match the 'values' column in the LULC
              raster of mode (b) and must be numeric and unique.
            * 'globio_lucode': The LULC code corresponding to the GLOBIO class
              to which it should be converted, using intermediate codes
              described in the example below.

        args['infrastructure_dir'] (string): used in "mode (a) and (b)" a path
            to a folder containing maps of either gdal compatible rasters or
            OGR compatible shapefiles.  These data will be used in the
            infrastructure to calculation of MSA.
        args['pasture_path'] (string): used in "mode (a)" path to pasture raster
        args['potential_vegetation_path'] (string): used in "mode (a)" path to
            potential vegetation raster
        args['pasture_threshold'] (float): used in "mode (a)"
        args['intensification_fraction'] (float): used in "mode (a)"; a value
            between 0 and 1 denoting proportion of total agriculture that
            should be classified as 'high input'
        args['primary_threshold'] (float): used in "mode (a)"
        args['msa_parameters_path'] (string): path to MSA classification
            parameters
        args['aoi_path'] (string): (optional) if it exists then final MSA raster
            is summarized by AOI
        args['globio_lulc_path'] (string): used in "mode (b)" path to predefined
            globio raster.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None

    """
    msa_parameter_table = load_msa_parameter_table(
        args['msa_parameters_path'], float(args['intensification_fraction']))
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'])
    # For intermediate files that users may want to explore:
    intermediate_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    # For intermediate files that users probably don't need to see,
    # but should persist for taskgraph purposes:
    tmp_dir = os.path.join(intermediate_dir, 'tmp')
    utils.make_directories(
        [output_dir, intermediate_dir, tmp_dir])

    # Initialize a TaskGraph
    taskgraph_db_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_db_dir, n_workers)

    gaussian_kernel_path = os.path.join(
        tmp_dir, 'gaussian_kernel%s.tif' % file_suffix)
    make_gaussian_kernel_task = task_graph.add_task(
        func=make_gaussian_kernel_path,
        args=(SIGMA, gaussian_kernel_path),
        target_path_list=[gaussian_kernel_path],
        task_name='gaussian_kernel')

    calculate_globio_task_list = []
    # get base raster cell size and nodata from whichever lulc is
    # provided in args
    if not args['predefined_globio']:
        globio_lulc_path = os.path.join(
            intermediate_dir, 'globio_lulc%s.tif' % file_suffix)
        base_lulc_info = pygeoprocessing.get_raster_info(args['lulc_path'])
        out_pixel_size = (abs(base_lulc_info['pixel_size'][0]) +
                          abs(base_lulc_info['pixel_size'][0])) / 2
        globio_nodata = -1
        globio_lulc_task = _calculate_globio_lulc_map(
            args['lulc_to_globio_table_path'], args['lulc_path'],
            args['potential_vegetation_path'], args['pasture_path'],
            gaussian_kernel_path, float(args['pasture_threshold']),
            float(args['primary_threshold']), file_suffix,
            tmp_dir, globio_lulc_path, globio_nodata, task_graph)
        calculate_globio_task_list.append(globio_lulc_task)
    else:
        LOGGER.info('no need to calculate GLOBIO LULC because it is passed in')
        globio_lulc_path = args['globio_lulc_path']
        globio_lulc_info = pygeoprocessing.get_raster_info(globio_lulc_path)
        out_pixel_size = (abs(globio_lulc_info['pixel_size'][0]) +
                          abs(globio_lulc_info['pixel_size'][0])) / 2
        globio_nodata = globio_lulc_info['nodata'][0]

    infrastructure_path = os.path.join(
        tmp_dir, 'combined_infrastructure%s.tif' % file_suffix)
    combine_infrastructure_task = task_graph.add_task(
        func=_collapse_infrastructure_layers,
        args=(args['infrastructure_dir'], globio_lulc_path, infrastructure_path,
              tmp_dir),
        target_path_list=[infrastructure_path],
        dependent_task_list=calculate_globio_task_list,
        task_name='combine_infrastructure')

    # calc_msa_f
    primary_veg_mask_path = os.path.join(
        tmp_dir, 'primary_veg_mask%s.tif' % file_suffix)
    primary_veg_mask_nodata = -1

    LOGGER.info("create mask of primary veg areas")
    # lucodes for primary veg are hardcoded in the local_op
    mask_primary_veg_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(globio_lulc_path, 1), (globio_nodata, 'raw'),
              (primary_veg_mask_nodata, 'raw')], _primary_veg_mask_op,
              primary_veg_mask_path, gdal.GDT_Int16, primary_veg_mask_nodata),
        target_path_list=[primary_veg_mask_path],
        dependent_task_list=calculate_globio_task_list,
        task_name='mask_primary_veg')

    LOGGER.info('smooth primary veg areas with gaussian filter')
    smoothed_primary_veg_mask_path = os.path.join(
        tmp_dir, 'smoothed_primary_veg_mask%s.tif' % file_suffix)
    smooth_primary_veg_mask_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=((primary_veg_mask_path, 1), (gaussian_kernel_path, 1),
              smoothed_primary_veg_mask_path),
        target_path_list=[smoothed_primary_veg_mask_path],
        dependent_task_list=[mask_primary_veg_task, make_gaussian_kernel_task],
        task_name='smooth_primary_veg_mask')

    LOGGER.info('calculate primary_veg_smooth')
    # Passing the filter over the veg mask means veg has bled outside the mask,
    # so mask it again to get the final ffqi
    primary_veg_smooth_path = os.path.join(
        intermediate_dir, 'primary_veg_smooth%s.tif' % file_suffix)
    smooth_primary_veg_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(primary_veg_mask_path, 1), (smoothed_primary_veg_mask_path, 1),
               (primary_veg_mask_nodata, 'raw')],
              _ffqi_op, primary_veg_smooth_path, gdal.GDT_Float32,
              primary_veg_mask_nodata),
        target_path_list=[primary_veg_smooth_path],
        dependent_task_list=[smooth_primary_veg_mask_task],
        task_name='smooth_primary_veg')

    LOGGER.info('calculate msa_f')
    msa_nodata = -1
    msa_f_table = msa_parameter_table['msa_f']
    msa_f_path = os.path.join(output_dir, 'msa_f%s.tif' % file_suffix)

    calculate_msa_f_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(primary_veg_smooth_path, 1), (primary_veg_mask_nodata, 'raw'),
               (msa_f_table, 'raw'), (msa_nodata, 'raw')],
              _msa_f_op, msa_f_path, gdal.GDT_Float32, msa_nodata),
        target_path_list=[msa_f_path],
        dependent_task_list=[smooth_primary_veg_task],
        task_name='calculate_msa_f')

    # calc_msa_i
    msa_i_other_table = msa_parameter_table['msa_i_other']
    msa_i_primary_table = msa_parameter_table['msa_i_primary']

    LOGGER.info('distance transform infrasture raster')
    distance_to_infrastructure_path = os.path.join(
        intermediate_dir, 'distance_to_infrastructure%s.tif' % file_suffix)
    distance_to_infrastructure_task = task_graph.add_task(
        func=pygeoprocessing.distance_transform_edt,
        args=((infrastructure_path, 1), distance_to_infrastructure_path),
        target_path_list=[distance_to_infrastructure_path],
        dependent_task_list=[combine_infrastructure_task],
        task_name='distance_to_infrastructure')

    LOGGER.info('calculate msa_i')
    msa_i_path = os.path.join(output_dir, 'msa_i%s.tif' % file_suffix)
    calculate_msa_i_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(globio_lulc_path, 1), (distance_to_infrastructure_path, 1),
               (out_pixel_size, 'raw'), (msa_i_primary_table, 'raw'),
               (msa_i_other_table, 'raw')],
              _msa_i_op, msa_i_path, gdal.GDT_Float32, msa_nodata),
        target_path_list=[msa_i_path],
        dependent_task_list=[distance_to_infrastructure_task],
        task_name='calculate_msa_i')

    # calc_msa_lu
    msa_lu_path = os.path.join(
        output_dir, 'msa_lu%s.tif' % file_suffix)
    LOGGER.info('calculate msa_lu')
    calculate_msa_lu_task = task_graph.add_task(
        func=pygeoprocessing.reclassify_raster,
        args=((globio_lulc_path, 1), msa_parameter_table['msa_lu'], msa_lu_path,
              gdal.GDT_Float32, globio_nodata),
        target_path_list=[msa_lu_path],
        dependent_task_list=calculate_globio_task_list,
        task_name='calculate_msa_lu')

    LOGGER.info('calculate msa')
    msa_path = os.path.join(
        output_dir, 'msa%s.tif' % file_suffix)
    calculate_msa_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(msa_f_path, 1), (msa_lu_path, 1), (msa_i_path, 1),
               (globio_nodata, 'raw')],
              _msa_op, msa_path, gdal.GDT_Float32, msa_nodata),
        target_path_list=[msa_path],
        dependent_task_list=[
            calculate_msa_f_task, calculate_msa_i_task, calculate_msa_lu_task],
        task_name='calculate_msa')

    LOGGER.info('summarize msa result in AOI polygons')
    # the AOI is an optional argument, so check for its existence
    if 'aoi_path' in args and len(args['aoi_path']) > 0:
        summary_aoi_path = os.path.join(
            output_dir, 'aoi_summary%s.shp' % file_suffix)
        task_graph.add_task(
            func=_summarize_results_in_aoi,
            args=(args['aoi_path'], summary_aoi_path, msa_path),
            target_path_list=[summary_aoi_path],
            dependent_task_list=[calculate_msa_task],
            task_name='summarize_msa_in_aoi')

    task_graph.close()
    task_graph.join()


def _summarize_results_in_aoi(aoi_path, summary_aoi_path, msa_path):
    """Aggregate MSA results to AOI polygons with zonal statistics.

    Parameters:
        aoi_path (string): path to aoi shapefile containing polygons.
        summary_aoi_path (string):
            path to copy of aoi shapefile with summary stats added.
        msa_path (string): path to msa results raster to summarize.

    Returns:
        None

    """
    # copy the aoi to an output shapefile
    original_datasource = gdal.OpenEx(aoi_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)
    # Delete if existing shapefile with the same name
    if os.path.isfile(summary_aoi_path):
        os.remove(summary_aoi_path)
    # Copy the input shapefile into the designated output folder
    driver = gdal.GetDriverByName('ESRI Shapefile')
    datasource_copy = driver.CreateCopy(
        summary_aoi_path, original_datasource)
    layer = datasource_copy.GetLayer()
    msa_summary_field_def = ogr.FieldDefn('msa_mean', ogr.OFTReal)
    msa_summary_field_def.SetWidth(24)
    msa_summary_field_def.SetPrecision(11)
    layer.CreateField(msa_summary_field_def)
    layer.SyncToDisk()

    msa_summary = pygeoprocessing.zonal_statistics(
        (msa_path, 1), summary_aoi_path)
    for feature in layer:
        feature_fid = feature.GetFID()
        # count == 0 if polygon outside raster bounds or only over nodata
        if msa_summary[feature_fid]['count'] != 0:
            field_val = (
                float(msa_summary[feature_fid]['sum'])
                / float(msa_summary[feature_fid]['count']))
            feature.SetField('msa_mean', field_val)
            layer.SetFeature(feature)


def _primary_veg_mask_op(lulc_array, globio_nodata, primary_veg_mask_nodata):
    """Masking out natural areas."""
    # lulc_array and nodata could conceivably be a float here,
    # if it's the user-provided globio dataset
    valid_mask = ~numpy.isclose(lulc_array, globio_nodata)
    # landcover type 1 in the GLOBIO schema represents primary vegetation
    result = numpy.empty_like(lulc_array, dtype=numpy.int16)
    result[:] = primary_veg_mask_nodata
    result[valid_mask] = lulc_array[valid_mask] == 1
    return result


def _ffqi_op(forest_areas_array, smoothed_forest_areas, forest_areas_nodata):
    """Mask out ffqi only where there's an ffqi."""
    result = numpy.empty_like(forest_areas_array, dtype=numpy.float32)
    result[:] = forest_areas_nodata
    # forest_areas_array and _nodata are integer types and not user-defined
    valid_mask = forest_areas_array != forest_areas_nodata
    result[valid_mask] = (
        forest_areas_array[valid_mask] * smoothed_forest_areas[valid_mask])
    return result


def _msa_f_op(
        primary_veg_smooth, primary_veg_mask_nodata, msa_f_table,
        msa_nodata):
    """Calculate msa fragmentation.

    Bin ffqi values based on rules defined in msa_parameters.csv.

    Parameters:
        primary_veg_smooth (array): float values representing ffqi.
        primary_veg_mask_nodata (int/float)
        msa_f_table (dict):
            subset of msa_parameters.csv with fragmentation bins defined.
        msa_nodata (int/float)

    Returns:
        Array with float values. One component of final MSA score.

    """
    nodata_mask = numpy.isclose(primary_veg_mask_nodata, primary_veg_smooth)
    msa_f = numpy.empty(primary_veg_smooth.shape)

    less_than = msa_f_table.pop('<', None)
    greater_than = msa_f_table.pop('>', None)
    if greater_than:
        msa_f[primary_veg_smooth > greater_than[0]] = (
                greater_than[1])
    for key in reversed(sorted(msa_f_table)):
        msa_f[primary_veg_smooth <= key] = msa_f_table[key]
    if less_than:
        msa_f[primary_veg_smooth < less_than[0]] = (
            less_than[1])

    msa_f[nodata_mask] = msa_nodata

    return msa_f


def _msa_i_op(
        lulc_array, distance_to_infrastructure, out_pixel_size,
        msa_i_primary_table, msa_i_other_table):
    """Calculate msa infrastructure.

    Bin distance_to_infrastructure values according to rules defined
    in msa_parameters.csv.

    Parameters:
        lulc_array (array): integer values representing globio landcover codes.
        distance_to_infrastructure (array):
            float values measuring distance from nearest infrastructure present
            in layers from args['infrastructure_dir'].
        out_pixel_size (float): from the globio lulc raster info.
        msa_i_primary_table (dict):
            subset of msa_parameters.csv with distance to infrastructure bins
            defined. These bins are applied to areas of primary veg.
        msa_i_other_table (dict):
            subset of msa_parameters.csv with distance to infrastructure bins
            defined. These bins are applied to areas of not primary veg.

    Returns:
        Array with float values. One component of final MSA score.

    """
    distance_to_infrastructure *= out_pixel_size  # convert to meters
    msa_i_primary = numpy.empty(lulc_array.shape)
    msa_i_other = numpy.empty(lulc_array.shape)

    primary_less_than = msa_i_primary_table.pop('<', None)
    primary_greater_than = msa_i_primary_table.pop('>', None)
    if primary_greater_than:
        msa_i_primary[distance_to_infrastructure > primary_greater_than[0]] = (
                primary_greater_than[1])
    for key in reversed(sorted(msa_i_primary_table)):
        msa_i_primary[distance_to_infrastructure <= key] = (
            msa_i_primary_table[key])
    if primary_less_than:
        msa_i_primary[distance_to_infrastructure < primary_less_than[0]] = (
            primary_less_than[1])

    other_less_than = msa_i_other_table.pop('<', None)
    other_greater_than = msa_i_other_table.pop('>', None)
    if other_greater_than:
        msa_i_other[distance_to_infrastructure > other_greater_than[0]] = (
                other_greater_than[1])
    for key in reversed(sorted(msa_i_other_table)):
        msa_i_other[distance_to_infrastructure <= key] = (
            msa_i_other_table[key])
    if other_less_than:
        msa_i_other[distance_to_infrastructure < other_less_than[0]] = (
            other_less_than[1])

    # lulc code 1 is primary veg
    msa_i = numpy.where(lulc_array == 1, msa_i_primary, msa_i_other)
    return msa_i


def _msa_op(msa_f, msa_lu, msa_i, globio_nodata):
        """Calculate the MSA which is the product of the sub MSAs."""
        result = numpy.empty_like(msa_f, dtype=numpy.float32)
        result[:] = globio_nodata
        valid_mask = ~numpy.isclose(msa_f, globio_nodata)
        result[valid_mask] = msa_f[valid_mask] * msa_lu[valid_mask] * msa_i[valid_mask]
        return result

def make_gaussian_kernel_path(sigma, kernel_path):
    """Create a gaussian kernel raster."""
    max_distance = sigma * 5
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_path.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=['BIGTIFF=IF_SAFER'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    srs = osr.SpatialReference()
    srs.SetUTM(11, 1)
    srs.SetWellKnownGeogCS('NAD27')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    col_index = numpy.array(range(kernel_size))
    integration = 0.0
    for row_index in range(kernel_size):
        kernel = numpy.exp(
            -((row_index - max_distance)**2 +
              (col_index - max_distance) ** 2)/(2.0*sigma**2)).reshape(
                  1, kernel_size)

        integration += numpy.sum(kernel)
        kernel_band.WriteArray(kernel, xoff=0, yoff=row_index)

    for row_index in range(kernel_size):
        kernel_row = kernel_band.ReadAsArray(
            xoff=0, yoff=row_index, win_xsize=kernel_size, win_ysize=1)
        kernel_row /= integration
        kernel_band.WriteArray(kernel_row, 0, row_index)


def load_msa_parameter_table(
        msa_parameter_table_filename, intensification_fraction):
    """Load parameter table to a dict that to define the MSA ranges.

    Parameters:
        msa_parameter_table_filename (string): path to msa csv table
        intensification_fraction (float): a number between 0 and 1 indicating
            what level between msa_lu 8 and 9 to define the general GLOBIO
            code "12" to.


        returns a dictionary of the form
            {
                'msa_f': {
                    valuea: msa_f_value, ...
                    valueb: ...
                    '<': (bound, msa_f_value),
                    '>': (bound, msa_f_value)}
                'msa_i_other_table': {
                    valuea: msa_i_value, ...
                    valueb: ...
                    '<': (bound, msa_i_other_value),
                    '>': (bound, msa_i_other_value)}
                'msa_i_primary': {
                    valuea: msa_i_primary_value, ...
                    valueb: ...
                    '<': (bound, msa_i_primary_value),
                    '>': (bound, msa_i_primary_value)}
                'msa_lu': {
                    valuea: msa_lu_value, ...
                    valueb: ...
                    '<': (bound, msa_lu_value),
                    '>': (bound, msa_lu_value)
                    12: (msa_lu_8 * (1.0 - intensification_fraction) +
                         msa_lu_9 * intensification_fraction}
            }

    """
    msa_table = pandas.read_csv(
        msa_parameter_table_filename, sep=None, engine='python')
    msa_dict = collections.defaultdict(dict)
    for _, row in msa_table.iterrows():
        if row['Value'][0] in ['<', '>']:
            # put the limit and the MSA value in a tub
            value = row['Value'][0]
            # take 1: because it starts with a < or >
            msa_dict[row['MSA_type']][value] = (
                float(row['Value'][1:]), float(row['MSA_x']))
            continue
        elif '-' in row['Value']:
            value = float(row['Value'].split('-')[1])
        else:
            value = float(row['Value'])
        msa_dict[row['MSA_type']][value] = float(row['MSA_x'])
    # landcover ID 12 is a linear interpolation between 8 and 9
    msa_dict['msa_lu'][12] = (
        msa_dict['msa_lu'][8] * (1.0 - intensification_fraction) +
        msa_dict['msa_lu'][9] * intensification_fraction)
    return dict(msa_dict)


def _calculate_globio_lulc_map(
        lulc_to_globio_table_path, lulc_path, potential_vegetation_path,
        pasture_path, gaussian_kernel_path, pasture_threshold,
        primary_threshold, file_suffix, tmp_dir,
        globio_lulc_path, globio_nodata, task_graph):
    """Translate a general landcover map into a GLOBIO version.

    Parameters:
        lulc_to_globio_table_path (string): a table that maps arbitrary
            landcover values to globio equivalents.
        lulc_path (string): path to the raw landcover map.
        potential_vegetation_path (string): a landcover map that indicates what
            the vegetation types would be if left to revert to natural state
        pasture_path (string): a path to a raster that indicates the percent
            of pasture contained in the pixel.  used to classify forest types
            from scrubland.
        gaussian_kernel_path (string): path to gaussian kernel raster
            passed to convolution.
        pasture_threshold (float): the threshold to classify pixels in pasture
            as potential forest or scrub
        primary_threshold (float): the threshold to classify the calculated
            FFQI pixels into core forest or secondary
        file_suffix - (string) to append on output file
        tmp_dir (string): path to location for intermediate files that users
            probably don't need to see, but should persist for taskgraph.
            The following files are created:
                'intermediate_globio_lulc.tif': reclassified landcover map
                    to globio landcover codes
                'ffqi.tif': index of fragmentation due to infrastructure and
                    original values of landscape
        globio_lulc_path (string): Path to globio lulc raster. Primary output
            of the function, starts with intermeidate globio and modifies based
            on the other biophysical parameters to the function as described in
            the GLOBIO process.
        globio_nodata (int): nodata value assigned to globio_lulc_path raster
        task_graph (TaskGraph): in-memory object from taskgraph.TaskGraph()

    Returns:
        The ultimate task in this branch of the task_graph

    """
    lulc_to_globio_table = utils.build_lookup_from_csv(
        lulc_to_globio_table_path, 'lucode')

    lulc_to_globio = dict(
        [(lulc_code, int(table['globio_lucode'])) for
         (lulc_code, table) in lulc_to_globio_table.items()])

    intermediate_globio_lulc_path = os.path.join(
        tmp_dir, 'intermediate_globio_lulc%s.tif' % file_suffix)
    reclass_lulc_to_globio_task = task_graph.add_task(
        func=pygeoprocessing.reclassify_raster,
        args=((lulc_path, 1), lulc_to_globio, intermediate_globio_lulc_path,
              gdal.GDT_Int32, globio_nodata),
        target_path_list=[intermediate_globio_lulc_path],
        task_name='reclassify_lulc_to_globio')

    forest_areas_path = os.path.join(
        tmp_dir, 'forest_areas%s.tif' % file_suffix)
    forest_areas_nodata = -1

    LOGGER.info("create mask of natural areas")
    mask_forests_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(intermediate_globio_lulc_path, 1), (globio_nodata, 'raw'),
               (forest_areas_nodata, 'raw')], _forest_area_mask_op,
              forest_areas_path, gdal.GDT_Int32, forest_areas_nodata),
        target_path_list=[forest_areas_path],
        dependent_task_list=[reclass_lulc_to_globio_task],
        task_name='mask_forest_area')

    LOGGER.info('smooth natural areas with gaussian filter')
    smoothed_forest_areas_path = os.path.join(
        tmp_dir, 'smoothed_forest_areas%s.tif' % file_suffix)
    smooth_forest_areas_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=((forest_areas_path, 1), (gaussian_kernel_path, 1),
              smoothed_forest_areas_path),
        target_path_list=[smoothed_forest_areas_path],
        dependent_task_list=[mask_forests_task],
        task_name='smooth_forest_areas')

    ffqi_path = os.path.join(
        tmp_dir, 'ffqi%s.tif' % file_suffix)
    LOGGER.info('calculate ffqi')
    calculate_ffqi_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(forest_areas_path, 1), (smoothed_forest_areas_path, 1),
               (forest_areas_nodata, 'raw')], _ffqi_op,
              ffqi_path, gdal.GDT_Float32, forest_areas_nodata),
        target_path_list=[ffqi_path],
        dependent_task_list=[smooth_forest_areas_task],
        task_name='ffqi')

    LOGGER.info('create the globio lulc')
    # The veg and pasture rasters are user inputs,
    # so may not be aligned yet with lulc
    base_raster_align_list = [potential_vegetation_path, pasture_path]
    target_raster_align_list = [os.path.join(
        tmp_dir, os.path.basename(x))
        for x in base_raster_align_list]
    base_raster_info = pygeoprocessing.get_raster_info(lulc_path)
    align_veg_pasture_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(base_raster_align_list,
              target_raster_align_list,
              ['near', 'bilinear'],
              base_raster_info['pixel_size'],
              base_raster_info['bounding_box']),
        target_path_list=target_raster_align_list,
        task_name='align_veg_pasture_rasters')

    calculate_globio_lulc_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(intermediate_globio_lulc_path, 1),
               (target_raster_align_list[0], 1),
               (target_raster_align_list[1], 1),
               (ffqi_path, 1), (globio_nodata, 'raw'),
               (pasture_threshold, 'raw'),
               (primary_threshold, 'raw')],
              _create_globio_lulc_op, globio_lulc_path, gdal.GDT_Int16,
              globio_nodata),
        target_path_list=[globio_lulc_path],
        dependent_task_list=[align_veg_pasture_task, calculate_ffqi_task],
        task_name='calculate_globio_lulc')

    return calculate_globio_lulc_task


def _forest_area_mask_op(lulc_array, globio_nodata, forest_areas_nodata):
    """Masking out forest areas."""
    valid_mask = ~numpy.isclose(lulc_array, globio_nodata)
    # landcover code 130 represents all MODIS forest codes which originate
    # as 1-5
    result = numpy.empty_like(lulc_array, dtype=numpy.int16)
    result[:] = forest_areas_nodata
    result[valid_mask] = lulc_array[valid_mask] == 130
    return result


def _create_globio_lulc_op(
        lulc_array, potential_vegetation_array, pasture_array,
        ffqi, globio_nodata, pasture_threshold, primary_threshold):
    """Construct GLOBIO lulc given relevant biophysical parameters."""
    result = numpy.empty_like(lulc_array, dtype=numpy.int16)
    result[:] = globio_nodata
    valid_mask = lulc_array != globio_nodata
    valid_result = result[valid_mask]

    # Split Shrublands and grasslands into primary vegetations,
    # livestock grazing areas, and man-made pastures. Landcover
    # 131 represents grassland/shrubland in the GLOBIO classification.
    grass_shrub_mask = lulc_array[valid_mask] == 131
    grass_shrub_result = valid_result[grass_shrub_mask]
    # fill with livestock grazing, then re-assign to pasture, primary veg.
    grass_shrub_result[:] = 5

    # man-made pasture
    valid_pasture_mask = potential_vegetation_array[valid_mask][grass_shrub_mask] <= 8
    grass_shrub_result[valid_pasture_mask] = 6

    # primary vegetation
    valid_primary_veg_mask = ~valid_pasture_mask & (
        pasture_array[valid_mask][grass_shrub_mask] < pasture_threshold)
    grass_shrub_result[valid_primary_veg_mask] = 1

    valid_result[grass_shrub_mask] = grass_shrub_result

    # Outside of the grass/shrub categories, carry over the original codes:
    valid_result[~grass_shrub_mask] = lulc_array[valid_mask][~grass_shrub_mask]

    # Step 1.4a: Split Forests into Primary, Secondary
    # 1 is primary forest
    # 3 is secondary forest
    valid_modis_forest_mask = lulc_array[valid_mask] == 130
    forest_result = valid_result[valid_modis_forest_mask]
    forest_result[:] = 1
    forest_result[
        ffqi[valid_mask][valid_modis_forest_mask] < primary_threshold] = 3
    valid_result[valid_modis_forest_mask] = forest_result

    # Classify all ag classes as a new LULC value "12" per our custom design
    # of agriculture. Landcover 132 represents agriculture landcover types
    # in the GLOBIO classification scheme
    valid_ag_mask = lulc_array[valid_mask] == 132
    valid_result[valid_ag_mask] = 12

    result[valid_mask] = valid_result
    return result


def _collapse_infrastructure_layers(
        infrastructure_dir, base_raster_path, infrastructure_path,
        tmp_dir):
    """Collapse all GIS infrastructure layers to one raster.

    Gathers all the GIS layers in the given directory and collapses them
    to a single byte raster mask where 1 indicates a pixel overlapping with
    one of the original infrastructure layers, 0 does not, and nodata
    indicates a region that has no layers that overlap but are still contained
    in the bounding box.

    Parameters:
        infrastructure_dir (string): path to a directory containing maps of
            either gdal compatible rasters or OGR compatible shapefiles.
        base_raster_path (string): a path to a file that has the dimensions and
            projection of the desired output infrastructure file.
        infrastructure_path (string): (output) path to a file that will be a
            byte raster with 1s everywhere there was a GIS layer present in
            the GIS layers in `infrastructure_dir`.
        tmp_dir (string): path to folder to store inetermediate datasets such
            as aligned versions of infrastructure rasters.

    Returns:
        None

    """
    # load the infrastructure layers from disk
    infrastructure_filenames = []
    infrastructure_nodata_list = []
    infrastructure_tmp_filenames = []
    # in case we need to rasterize some vector inputs:
    tmp_rasterize_dir = os.path.join(tmp_dir, 'rasterized')
    for root_directory, _, filename_list in os.walk(infrastructure_dir):
        for filename in filename_list:
            if filename.lower().endswith(".tif"):
                infrastructure_filenames.append(
                    os.path.join(root_directory, filename))
                infrastructure_nodata_list.append(
                    pygeoprocessing.get_raster_info(
                        infrastructure_filenames[-1])['nodata'][0])

            if filename.lower().endswith(".shp"):
                utils.make_directories([tmp_rasterize_dir])
                file_handle, tmp_raster_path = tempfile.mkstemp(
                    dir=tmp_rasterize_dir, suffix='.tif')
                os.close(file_handle)

                pygeoprocessing.new_raster_from_base(
                    base_raster_path, tmp_raster_path,
                    gdal.GDT_Int32, [-1.0], fill_value_list=[0])
                pygeoprocessing.rasterize(
                    os.path.join(root_directory, filename),
                    tmp_raster_path, burn_values=[1],
                    option_list=["ALL_TOUCHED=TRUE"])

                infrastructure_filenames.append(tmp_raster_path)
                infrastructure_tmp_filenames.append(tmp_raster_path)
                infrastructure_nodata_list.append(
                    pygeoprocessing.get_raster_info(
                        infrastructure_filenames[-1])['nodata'][0])

    if len(infrastructure_filenames) == 0:
        raise ValueError(
            "infrastructure directory didn't have any rasters or "
            "vectors at %s", infrastructure_dir)

    infrastructure_nodata = -1

    def _collapse_infrastructure_op(*infrastructure_array_list):
        """For each pixel, create mask 1 if all valid, else set to nodata."""
        nodata_mask = (
            numpy.isclose(
                infrastructure_array_list[0], infrastructure_nodata_list[0]))
        infrastructure_result = infrastructure_array_list[0] > 0
        for index in range(1, len(infrastructure_array_list)):
            current_nodata = numpy.isclose(
                infrastructure_array_list[index],
                infrastructure_nodata_list[index])

            infrastructure_result = (
                infrastructure_result |
                ((infrastructure_array_list[index] > 0) & ~current_nodata))

            nodata_mask = (
                nodata_mask & current_nodata)

        infrastructure_result[nodata_mask] = infrastructure_nodata
        return infrastructure_result

    LOGGER.info('collapse infrastructure into one raster')
    aligned_infrastructure_target_list = [os.path.join(
        tmp_dir, os.path.basename(x))
        for x in infrastructure_filenames]
    base_raster_info = pygeoprocessing.get_raster_info(
        base_raster_path)

    pygeoprocessing.align_and_resize_raster_stack(
        infrastructure_filenames,
        aligned_infrastructure_target_list,
        ['near'] * len(infrastructure_filenames),
        base_raster_info['pixel_size'],
        base_raster_info['bounding_box'])
    infra_filename_band_list = [
        (x, 1) for x in aligned_infrastructure_target_list]
    pygeoprocessing.raster_calculator(
        infra_filename_band_list, _collapse_infrastructure_op,
        infrastructure_path, gdal.GDT_Byte, infrastructure_nodata)

    # clean up the temporary filenames
    if os.path.isdir(tmp_rasterize_dir):
        for filename in infrastructure_tmp_filenames:
            os.remove(filename)
        os.rmdir(tmp_rasterize_dir)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Parameters:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    return validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])
