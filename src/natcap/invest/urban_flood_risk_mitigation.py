"""Carbon Storage and Sequestration."""
from __future__ import absolute_import
import logging
import os

from osgeo import gdal
from osgeo import ogr
import pygeoprocessing
import taskgraph
import pandas
import numpy
import scipy

from . import validation
from . import utils

LOGGER = logging.getLogger(__name__)


def execute(args):
    """Urban Flood Risk Mitigation model.

    The model computes the peak flow attenuation for each pixel, delineates
    areas benefiting from this service, then calculates the monetary value of
    potential avoided damage to built infrastructure.

    Parameters:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] (string): appended to any output file name.
        args['dem_path'] (string): path to the DEM that will be used to
            delineate watersheds.
        args['aoi_watersheds_path'] (string): path to a shapefile of
            (sub)watersheds or sewersheds used to indicate spatial area of
            interest.
        args['rainfall_depth'] (float): depth of rainfall in mm.
        args['lulc_path'] (string): path to a landcover raster.
        args['soils_hydrological_group_raster_path'] (string): Raster with
            values equal to 1, 2, 3, 4, corresponding to soil hydrologic group
            A, B, C, or D, respectively (used to derive the CN number).
        args['curve_number_table_path'] (string): path to a CSV table that
            contains at least the headers 'lucode', 'CN_A', 'CN_B', 'CN_C',
            'CN_D'.
        args['flood_prone_areas_vector_path'] (string): path to vector of
            polygon areas of known occurrence of flooding where peakflow
            retention will be more critical.
        args['built_infrastructure_vector_path'] (string): path to a vector
            with built infrastructure footprints. Attribute table contains a
            column 'Type' with integers (e.g. 1=residential, 2=office, etc.).
        args['infrastructure_damage_loss_table_path'] (string): path to a
            a CSV table with columns 'Type' and 'Damage' with values of built
            infrastructure type from the 'Type' field in
            `args['built_infrastructure_vector_path']` and potential damage
            loss (in $/m^2).

    Returns:
        None.

    """
    temporary_working_dir = os.path.join(
        args['workspace_dir'], 'temp_working_dir')
    utils.make_directories([args['workspace_dir'], temporary_working_dir])

    task_graph = taskgraph.TaskGraph(temporary_working_dir, -1)

    # Align LULC with soils
    aligned_lulc_path = os.path.join(
        temporary_working_dir, 'aligned_lulc.tif')
    aligned_soils_path = os.path.join(
        temporary_working_dir, 'aligned_soils_hydrological_group.tif')

    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_path'])
    target_pixel_size = lulc_raster_info['pixel_size']
    target_sr_wkt = lulc_raster_info['projection']

    soil_raster_info = pygeoprocessing.get_raster_info(
        args['soils_hydrological_group_raster_path'])

    align_raster_stack_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            [args['lulc_path'], args['soils_hydrological_group_raster_path']],
            [aligned_lulc_path, aligned_soils_path],
            ['mode', 'mode'],
            target_pixel_size, 'intersection'),
        kwargs={
            'target_sr_wkt': target_sr_wkt,
            'base_vector_path_list': [args['aoi_watersheds_path']],
            'raster_align_index': 0},
        target_path_list=[aligned_lulc_path, aligned_soils_path],
        task_name='align raster stack')

    # Load CN table
    cn_table = utils.build_lookup_from_csv(
        args['curve_number_table_path'], 'lucode')

    # make cn_table into a 2d array where first dim is lucode, second is
    # 0..3 to correspond to CN_A..CN_D
    data = []
    row_ind = []
    col_ind = []
    for lucode in cn_table:
        data.extend([
            cn_table[lucode]['cn_%s' % soil_id]
            for soil_id in ['a', 'b', 'c', 'd']])
        row_ind.extend([int(lucode)] * 4)
    col_ind = [0, 1, 2, 3] * (len(row_ind) // 4)
    lucode_to_cn_table = scipy.sparse.csr_matrix((data, (row_ind, col_ind)))

    cn_nodata = -1
    lucode_nodata = lulc_raster_info['nodata'][0]
    soil_type_nodata = soil_raster_info['nodata'][0]

    def lu_to_cn(lucode_array, soil_type_array):
        """Map combination landcover soil type map to curve number raster."""
        result = numpy.empty_like(lucode_array, dtype=numpy.float32)
        result[:] = cn_nodata
        valid_mask = (
            (lucode_array != lucode_nodata) &
            (soil_type_array != soil_type_nodata))

        # this is an array where each column represents a valid landcover
        # pixel and the rows are the curve number index for the landcover
        # type under that pixel (0..3 are CN_A..CN_D and 4 is "unknown")
        per_pixel_cn_array = (
            lucode_to_cn_table[lucode_array[valid_mask]].toarray().reshape(
                (-1, 4))).transpose()

        # this is the soil type array with values ranging from 0..4 that will
        # choose the appropriate row for each pixel colum in
        # `per_pixel_cn_array`
        soil_choose_array = (
            soil_type_array[valid_mask].astype(numpy.int8))-1

        # soil arrays are 1.0 - 4.0, remap to 0 - 3 and choose from the per
        # pixel CN array
        result[valid_mask] = numpy.choose(
            soil_choose_array,
            per_pixel_cn_array)

        return result

    cn_raster_path = os.path.join(args['workspace_dir'], 'cn_raster.tif')

    align_raster_stack_task.join()

    pygeoprocessing.raster_calculator(
        [(aligned_lulc_path, 1), (aligned_soils_path, 1)], lu_to_cn,
        cn_raster_path, gdal.GDT_Float32, cn_nodata)

    # Generate Smax
    # Generate Qpi

    task_graph.close()
    task_graph.join()


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
    missing_key_list = []
    no_value_list = []
    validation_error_list = []

    required_keys = [
        'workspace_dir',
        ]

    for key in required_keys:
        if limit_to is None or limit_to == key:
            if key not in args:
                missing_key_list.append(key)
            elif args[key] in ['', None]:
                no_value_list.append(key)

    if len(missing_key_list) > 0:
        # if there are missing keys, we have raise KeyError to stop hard
        raise KeyError(
            "The following keys were expected in `args` but were missing " +
            ', '.join(missing_key_list))

    if len(no_value_list) > 0:
        validation_error_list.append(
            (no_value_list, 'parameter has no value'))

    file_type_list = [
        ]

    # check that existing/optional files are the correct types
    with utils.capture_gdal_logging():
        for key, key_type in file_type_list:
            if ((limit_to is None or limit_to == key) and
                    key in args and key in required_keys):
                if not os.path.exists(args[key]):
                    validation_error_list.append(
                        ([key], 'not found on disk'))
                    continue
                if key_type == 'raster':
                    raster = gdal.Open(args[key])
                    if raster is None:
                        validation_error_list.append(
                            ([key], 'not a raster'))
                    del raster
                elif key_type == 'vector':
                    vector = ogr.Open(args[key])
                    if vector is None:
                        validation_error_list.append(
                            ([key], 'not a vector'))
                    del vector

    return validation_error_list
