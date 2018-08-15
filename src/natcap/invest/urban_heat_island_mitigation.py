"""Urban Heat Island Mitigation model."""
from __future__ import absolute_import
import logging
import os
import time
import multiprocessing
import uuid
import pickle

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
TARGET_NODATA = -1


def execute(args):
    """Urban Flood Heat Island Mitigation model.

    Parameters:
        args['workspace_dir'] (str): path to target output directory.
        args['air_temp_raster_path'] (str): raster of air temperature.
        args['lulc_raster_path'] (str): path to landcover raster.
        args['ref_eto_raster_path'] (str): path to evapotranspiration raster.
        args['et_max'] (float): maximum evapotranspiration.
        args['aoi_vector_path'] (str): path to desired AOI.
        args['biophysical_table_path'] (str): table to map landcover codes to
            Shade, Kc, and Albedo values. Must contain the fields 'lucode',
            'shade', 'kc', and 'albedo'.
        args['urban_park_cooling_distance'] (float): Distance (in m) over
            which large urban parks (> 2 ha) will have a cooling effect.
        args['uhi_max'] (float): Magnitude of the UHI effect.
        args['building_vector_path']: path to a vector of building footprints
            that contains at least the field 'type'.
        args['energy_consumption_table_path'] (str): path to a table that
            maps building types to energy consumption. Must contain at least
            the fields 'type' and 'consumption'.

    Returns:
        None.

    """
    temporary_working_dir = os.path.join(
        args['workspace_dir'], 'temp_working_dir')
    utils.make_directories([args['workspace_dir'], temporary_working_dir])
    biophysical_lucode_map = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'lucode', to_lower=True,
        warn_if_missing=True)

    task_graph = taskgraph.TaskGraph(
        temporary_working_dir, -1) #max(1, multiprocessing.cpu_count()))

    # align all the input rasters.
    aligned_air_temp_raster_path = os.path.join(
        temporary_working_dir, 'air_temp.tif')
    aligned_lulc_raster_path = os.path.join(
        temporary_working_dir, 'lulc.tif')
    aligned_ref_eto_raster_path = os.path.join(
        temporary_working_dir, 'ref_eto.tif')

    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_raster_path'])

    aligned_raster_path_list = [
        aligned_air_temp_raster_path, aligned_lulc_raster_path,
        aligned_ref_eto_raster_path]
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            [args['air_temp_raster_path'], args['lulc_raster_path'],
             args['ref_eto_raster_path']], aligned_raster_path_list, [
                'cubicspline', 'mode', 'cubicspline'],
            lulc_raster_info['pixel_size'], 'intersection'),
        kwargs={
            'base_vector_path_list': [args['aoi_vector_path']],
            'raster_align_index': 1,
            'target_sr_wkt': lulc_raster_info['projection']},
        target_path_list=aligned_raster_path_list,
        task_name='align rasters')

    task_path_prop_map = {}

    for prop in ['kc', 'shade', 'albedo']:
        prop_map = dict([
            (lucode, x[prop])
            for lucode, x in biophysical_lucode_map.items()])

        prop_raster_path = os.path.join(
            temporary_working_dir, '%s.tif' % prop)
        prop_task = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (aligned_lulc_raster_path, 1), prop_map, prop_raster_path,
                gdal.GDT_Float32, TARGET_NODATA),
            kwargs={'values_required': True},
            target_path_list=[prop_raster_path],
            dependent_task_list=[align_task],
            task_name='reclassify to %s' % prop)
        task_path_prop_map[prop] = (prop_task, prop_raster_path)

    eto_nodata = pygeoprocessing.get_raster_info(
        args['ref_eto_raster_path'])['nodata'][0]
    eti_raster_path = os.path.join(args['workspace_dir'], 'eti.tif')
    eti_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(task_path_prop_map['kc'][1], 1), (TARGET_NODATA, 'raw'),
             (aligned_ref_eto_raster_path, 1), (eto_nodata, 'raw'),
             (float(args['et_max']), 'raw'), (TARGET_NODATA, 'raw')],
            calc_eti_op, eti_raster_path, gdal.GDT_Float32, TARGET_NODATA),
        target_path_list=[eti_raster_path],
        dependent_task_list=[task_path_prop_map['kc'][0]],
        task_name='calculate eti')

    cc_raster_path = os.path.join(args['workspace_dir'], 'cc.tif')
    cc_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (task_path_prop_map['shade'][1], 1),
            (task_path_prop_map['albedo'][1], 1),
            (eti_raster_path, 1)], calc_cc_op, cc_raster_path,
            gdal.GDT_Float32, TARGET_NODATA),
        target_path_list=[cc_raster_path],
        dependent_task_list=[
            task_path_prop_map['shade'][0], task_path_prop_map['albedo'][0],
            eti_task],
        task_name='calculate cc index')

    air_temp_nodata = pygeoprocessing.get_raster_info(
        args['air_temp_raster_path'])['nodata'][0]
    t_air_raster_path = os.path.join(args['workspace_dir'], 'T_air.tif')
    t_air_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (aligned_air_temp_raster_path, 1), (air_temp_nodata, 'raw'),
            (cc_raster_path, 1), (float(args['uhi_max']), 'raw')],
            calc_t_air_op, t_air_raster_path, gdal.GDT_Float32,
            TARGET_NODATA),
        target_path_list=[t_air_raster_path],
        dependent_task_list=[cc_task, align_task],
        task_name='calculate T air')

    # intersect built_infrastructure_vector_path with aoi_watersheds_path
    intermediate_building_vector_path = os.path.join(
        temporary_working_dir, 'intermediate_building_vector.gpkg')
    # this is the field name that can be used to uniquely identify a feature
    key_field_id = 'objectid_invest_natcap'
    intermediate_building_vector_task = task_graph.add_task(
        func=reproject_and_label_vector,
        args=(
            args['building_vector_path'], lulc_raster_info['projection'],
            key_field_id, intermediate_building_vector_path),
        target_path_list=[intermediate_building_vector_path],
        task_name='reproject and label building vector')

    # zonal stats over buildings for t_air
    t_air_stats_pickle_path = os.path.join(
        temporary_working_dir, 't_air_stats.pickle')
    pickle_t_air_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_building_vector_path, 'OBJECTID', t_air_raster_path,
            t_air_stats_pickle_path),
        target_path_list=[t_air_stats_pickle_path],
        dependent_task_list=[t_air_task, intermediate_building_vector_task],
        task_name='pickle t-air stats')

    t_ref_stats_pickle_path = os.path.join(
        temporary_working_dir, 't_ref_stats.pickle')
    pickle_t_ref_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_building_vector_path, 'OBJECTID',
            aligned_air_temp_raster_path, t_ref_stats_pickle_path),
        target_path_list=[
            t_ref_stats_pickle_path, align_task,
            intermediate_building_vector_task],
        task_name='pickle t-ref stats')

    task_graph.close()
    task_graph.join()


def reproject_and_label_vector(
        base_vector_path, target_projection_wkt, target_key_field_id,
        target_vector_path):
    """Reproject to wkt and label features for unique ID.

    Parameters:
        base_vector_path (path): path to vector file.
        target_projection_wkt (str): desired target projection in WKT.
        target_key_field_id (str): field to add to target vector that will
            have a unique feature integer id for each feature.
        target_vector_path (str): path to desired output target vector.

    Return:
        None.

    """
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(target_projection_wkt)
    epsg_code = int(target_srs.GetAttrValue("AUTHORITY", 1))
    LOGGER.debug('epsg target code: %s', epsg_code)

    driver = gdal.GetDriverByName('GPKG')
    base_vector = gdal.OpenEx(base_vector_path)
    driver.CreateCopy(target_vector_path, base_vector)
    target_vector = gdal.OpenEx(
        target_vector_path, gdal.OF_VECTOR | gdal.GA_Update)

    target_layer = target_vector.GetLayer()
    target_layer.CreateField(ogr.FieldDefn(target_key_field_id, ogr.OFTReal))
    target_layer.SyncToDisk()
    geometry_field = target_layer.GetGeometryColumn()

    target_vector.ExecuteSQL(
        'UPDATE %s SET %s=Transform(geom, 4326), %s=rowid' % (
            target_layer.GetName(), geometry_field,
            target_key_field_id))

    #LOGGER.debug('reprojecting %s', base_vector_path)
    #pygeoprocessing.reproject_vector(
    #    base_vector_path, target_projection_wkt,
    #    target_vector_path, layer_index=0, driver_name='GPKG')


def pickle_zonal_stats(
        base_vector_path, key_field, base_raster_path, target_pickle_path):
    """Calculate Zonal Stats for a vector/raster pair and pickle result.

    Parameters:
        base_vector_path (str): path to vector file
        key_field (str): field in `base_vector_path` file that uniquely
            identifies each feature.
        base_raster_path (str): path to raster file to aggregate over.
        target_pickle_path (str): path to desired target pickle file that will
            be a pickle of the pygeoprocessing.zonal_stats function.

    Returns:
        None.

    """
    zonal_stats = pygeoprocessing.zonal_statistics(
        (base_raster_path, 1), base_vector_path, key_field)
    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(zonal_stats, pickle_file)


def calc_t_air_op(t_air_ref_array, t_air_ref_nodata, hm_array, uhi_max):
    """Calculate air temperature T_(air,i)=T_(air,ref)+(1-HM_i)*UHI_max."""
    result = numpy.empty(hm_array.shape, dtype=numpy.float32)
    result[:] = TARGET_NODATA
    valid_mask = ~(
        numpy.isclose(hm_array, TARGET_NODATA) |
        numpy.isclose(t_air_ref_array, t_air_ref_nodata))
    result[valid_mask] = t_air_ref_array[valid_mask] + (
        1-hm_array[valid_mask]) * uhi_max
    return result


def calc_cc_op(shade_array, albedo_array, eti_array):
    """Calculate the cooling capacity index CC_i=.6*shade+.2*albedo+.2*ETI."""
    result = numpy.empty(shade_array.shape, dtype=numpy.float32)
    result[:] = TARGET_NODATA
    valid_mask = ~(
        numpy.isclose(shade_array, TARGET_NODATA) |
        numpy.isclose(albedo_array, TARGET_NODATA) |
        numpy.isclose(eti_array, TARGET_NODATA))
    result[valid_mask] = (
        0.6*shade_array[valid_mask] +
        0.2*albedo_array[valid_mask] +
        0.2*eti_array[valid_mask])
    return result


def calc_eti_op(
        kc_array, kc_nodata, et0_array, et0_nodata, et_max, target_nodata):
    """Calculate ETI =(K_c ET_0)/ET_max ."""
    result = numpy.empty(kc_array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = ~(
        numpy.isclose(kc_array,  kc_nodata) |
        numpy.isclose(et0_array, et0_nodata))
    result[valid_mask] = (
        kc_array[valid_mask] * et0_array[valid_mask] / et_max)
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
