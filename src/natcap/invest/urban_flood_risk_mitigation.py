"""Urban Flood Risk Mitigation model."""
from __future__ import absolute_import
import logging
import os
import time

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pygeoprocessing
import taskgraph
import numpy
import scipy
import rtree
import shapely.wkb
import shapely.prepared

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

    cn_raster_path = os.path.join(args['workspace_dir'], 'cn_raster.tif')
    align_raster_stack_task.join()

    cn_raster_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(aligned_lulc_path, 1), (aligned_soils_path, 1),
             (lucode_nodata, 'raw'), (soil_type_nodata, 'raw'),
             (cn_nodata, 'raw'), (lucode_to_cn_table, 'raw')], lu_to_cn_op,
            cn_raster_path, gdal.GDT_Float32, cn_nodata),
        target_path_list=[cn_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create cn raster')

    # Generate S_max
    s_max_nodata = -9999
    s_max_raster_path = os.path.join(args['workspace_dir'], 's_max.tif')
    s_max_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(cn_raster_path, 1), (cn_nodata, 'raw'), (s_max_nodata, 'raw')],
            s_max_op, s_max_raster_path, gdal.GDT_Float32, s_max_nodata),
        target_path_list=[s_max_raster_path],
        dependent_task_list=[cn_raster_task],
        task_name='create s_max')

    # Generate Qpi
    q_pi_nodata = -9999.
    q_pi_raster_path = os.path.join(args['workspace_dir'], 'q_pi.tif')
    q_pi_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(float(args['rainfall_depth']), 'raw'), (s_max_raster_path, 1),
             (s_max_nodata, 'raw'), (q_pi_nodata, 'raw')], q_pi_op,
            q_pi_raster_path, gdal.GDT_Float32, q_pi_nodata),
        target_path_list=[q_pi_raster_path],
        dependent_task_list=[s_max_task],
        task_name='create q_pi')

    # Genereate Peak flow Retention
    peak_flow_nodata = -9999.
    peak_flow_raster_path = os.path.join(args['workspace_dir'], 'R_i.tif')
    peak_flow_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (float(args['rainfall_depth']), 'raw'), (q_pi_raster_path, 1),
            (q_pi_nodata, 'raw'), (peak_flow_nodata, 'raw')], peak_flow_op,
            peak_flow_raster_path, gdal.GDT_Float32, peak_flow_nodata),
        target_path_list=[peak_flow_raster_path],
        dependent_task_list=[q_pi_task],
        task_name='create peak flow')

    # intersect built_infrastructure_vector_path with aoi_watersheds_path
    target_watershed_result_vector_path = os.path.join(
        args['workspace_dir'], 'flood_risk_service.gpkg')
    build_service_vector_task = task_graph.add_task(
        func=build_service_vector,
        args=(
            args['aoi_watersheds_path'], target_sr_wkt,
            args['infrastructure_damage_loss_table_path'],
            args['built_infrastructure_vector_path'],
            target_watershed_result_vector_path),
        target_path_list=[target_watershed_result_vector_path],
        task_name='build_service_vector_task')

    q_pi_zonal_stats_task = task_graph.add_task(
        func=add_zonal_stats,
        args=(
            peak_flow_raster_path, target_watershed_result_vector_path,
            'OBJECTID', 'q_pi'),
        target_path_list=[],
        dependent_task_list=[peak_flow_task, build_service_vector_task],
        task_name='q_pi stats')

    task_graph.close()
    task_graph.join()


def add_zonal_stats(
        base_raster_path, aggregate_vector_path, aggregate_field_name,
        target_field_name):
    """Add watershed scale values of the given base_raster.

    Parameters:
        base_raster_path (str): path to raster to aggregate over.
        aggregate_vector_path (str): path to vector to aggregate values over
        aggregate_field_name (str): key field in `aggregate_vector_path`
            that can be used to index the per-feature results.

    Return:
        None.

    """
    LOGGER.info("Processing zonal stats for %s", target_field_name)
    stats = pygeoprocessing.zonal_statistics(
        (base_raster_path, 1), aggregate_vector_path,
        aggregate_field_name)
    LOGGER.debug(stats)

    aggregate_vector = gdal.OpenEx(
        aggregate_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    aggregate_layer = aggregate_vector.GetLayer()

    for summary_op_name in ['min', 'max', 'sum']:
        aggregate_layer.CreateField(
            ogr.FieldDefn('%s_%s' % (
                target_field_name, summary_op_name), ogr.OFTReal))

    last_time = time.time()
    for aggregate_index, aggregate_feature in enumerate(aggregate_layer):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                "processing watershed result %.2f%%",
                (100.0 * (aggregate_index+1)) /
                aggregate_layer.GetFeatureCount())
            last_time = current_time

        for summary_op_name in ['min', 'max', 'sum']:
            LOGGER.debug(stats[aggregate_feature.GetField(
                    aggregate_field_name)])
            aggregate_feature.SetField(
                '%s_%s' % (target_field_name, summary_op_name),
                float(stats[aggregate_feature.GetField(
                    aggregate_field_name)][summary_op_name]))


def build_service_vector(
        base_watershed_vector_path, target_wkt, damage_table_path,
        built_infrastructure_vector_path,
        target_watershed_result_vector_path):
    """Construct the service polygon.

    The ``base_watershed_vector_path`` should be intersected with the
    ``built_infrastructure_vector_path`` to get
        * affected population ?
        * affected build?

    Parameters:
        base_watershed_vector_path (str): path to base watershed vector,
        target_wkt (str): desired target projection.
        built_infrastructure_vector_path (str): path to infrastructure vector
            containing at least the integer field 'Type'.
        damage_table_path (str): path to a CSV table containing fields
            'Type' and 'Damage'. For every value of 'Type' in the
            built_infrastructure_vector there must be a corresponding entry
            in this table.
        target_watershed_result_vector_path (str): path to desired target
            watershed result vector with watershed scale values of stats.

    Returns:
        None.

    """
    damage_type_map = utils.build_lookup_from_csv(
        damage_table_path, 'type', to_lower=True, warn_if_missing=True)

    pygeoprocessing.reproject_vector(
        base_watershed_vector_path, target_wkt,
        target_watershed_result_vector_path, layer_index=0,
        driver_name='GPKG')

    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(target_wkt)

    infrastructure_rtree = rtree.index.Index()
    infrastructure_geometry_list = []
    infrastructure_vector = gdal.OpenEx(
        built_infrastructure_vector_path, gdal.OF_VECTOR)
    infrastructure_layer = infrastructure_vector.GetLayer()

    infrastructure_srs = infrastructure_layer.GetSpatialRef()
    infrastructure_to_target = osr.CoordinateTransformation(
        infrastructure_srs, target_srs)

    infrastructure_layer_defn = infrastructure_layer.GetLayerDefn()
    for field_name in ['type', 'Type', 'TYPE']:
        type_index = infrastructure_layer_defn.GetFieldIndex(field_name)
        if type_index != -1:
            break
    if type_index == -1:
        raise ValueError(
            "Could not find field 'Type' in %s",
            built_infrastructure_vector_path)

    LOGGER.info("building infrastructure lookup dict")
    for infrastructure_feature in infrastructure_layer:
        infrastructure_geom = infrastructure_feature.GetGeometryRef().Clone()
        infrastructure_geom.Transform(infrastructure_to_target)
        infrastructure_geometry_list.append({
            'geom': shapely.wkb.loads(
                infrastructure_geom.ExportToWkb()),
            'damage': damage_type_map[
                infrastructure_feature.GetField(type_index)]['damage']
        })
        infrastructure_rtree.insert(
            len(infrastructure_geometry_list)-1,
            infrastructure_geometry_list[-1]['geom'].bounds)

    infrastructure_vector = None
    infrastructure_layer = None

    watershed_vector = gdal.OpenEx(
        target_watershed_result_vector_path, gdal.OF_VECTOR | gdal.OF_UPDATE)
    watershed_layer = watershed_vector.GetLayer()
    watershed_layer.CreateField(ogr.FieldDefn('Affected.Build', ogr.OFTReal))
    watershed_layer.SyncToDisk()

    last_time = time.time()
    for watershed_index, watershed_feature in enumerate(watershed_layer):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                "processing watershed result %.2f%%",
                (100.0 * (watershed_index+1)) /
                watershed_layer.GetFeatureCount())
            last_time = current_time
        watershed_shapely = shapely.wkb.loads(
            watershed_feature.GetGeometryRef().ExportToWkb())
        watershed_prep_geom = shapely.prepared.prep(watershed_shapely)
        intersect_area = 0.0
        for infrastructure_index in infrastructure_rtree.intersection(
                watershed_shapely.bounds):
            infrastructure_geom = infrastructure_geometry_list[
                infrastructure_index]['geom']
            if watershed_prep_geom.intersects(infrastructure_geom):
                intersect_area += (
                    watershed_shapely.intersection(infrastructure_geom).area *
                    infrastructure_geometry_list[infrastructure_index][
                        'damage'])

        watershed_feature.SetField('Affected.Build', intersect_area)
        watershed_layer.SetFeature(watershed_feature)


def peak_flow_op(p_value, q_pi_array, q_pi_nodata, result_nodata):
    """Calculate peak flow retention."""
    result = numpy.empty_like(q_pi_array)
    result[:] = result_nodata
    valid_mask = numpy.ones(q_pi_array.shape, dtype=numpy.bool)
    if q_pi_nodata:
        valid_mask[:] = ~numpy.isclose(q_pi_array, q_pi_nodata)
    result[valid_mask] = 1.0 - q_pi_array[valid_mask] / p_value
    return result


def q_pi_op(p_value, s_max_array, s_max_nodata, result_nodata):
    """Calculate peak flow Q (mm) with the Curve Number method."""
    lam = 0.2  # this value of lambda is hard-coded in the design doc.
    result = numpy.empty_like(s_max_array)
    result[:] = result_nodata

    zero_mask = (p_value <= lam * s_max_array)
    non_nodata_mask = numpy.ones(s_max_array.shape, dtype=numpy.bool)
    if s_max_nodata:
        non_nodata_mask[:] = ~numpy.isclose(s_max_array, s_max_nodata)

    # valid if not nodata and not going to be set to 0.
    valid_mask = non_nodata_mask & ~zero_mask
    result[valid_mask] = (
        p_value - lam * s_max_array[valid_mask])**2.0 / (
            p_value + (1 - lam) * s_max_array[valid_mask])
    # any non-nodata result that should be zero is set so.
    result[zero_mask & non_nodata_mask] = 0.0
    return result


def s_max_op(cn_array, cn_nodata, result_nodata):
    """Calculate S_max from the curve number."""
    result = numpy.empty_like(cn_array)
    result[:] = result_nodata
    zero_mask = cn_array == 0
    valid_mask = ~zero_mask
    if cn_nodata:
        valid_mask[:] &= ~numpy.isclose(cn_array, cn_nodata)
    result[valid_mask] = 25400.0 / cn_array[valid_mask] - 254.0
    result[zero_mask] = 0.0
    return result


def lu_to_cn_op(
        lucode_array, soil_type_array, lucode_nodata, soil_type_nodata,
        cn_nodata, lucode_to_cn_table):
    """Map combination landcover soil type map to curve number raster."""
    result = numpy.empty_like(lucode_array, dtype=numpy.float32)
    result[:] = cn_nodata
    valid_mask = numpy.ones(lucode_array.shape, dtype=numpy.bool)
    if lucode_nodata:
        valid_mask[:] &= ~numpy.isclose(lucode_array, lucode_nodata)
    if soil_type_nodata:
        valid_mask[:] &= ~numpy.isclose(soil_type_array, soil_type_nodata)

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
