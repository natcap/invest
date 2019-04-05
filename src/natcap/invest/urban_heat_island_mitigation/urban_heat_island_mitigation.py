"""Urban Heat Island Mitigation model."""
from __future__ import absolute_import
import logging
import os
import pickle
import time

from osgeo import gdal
from osgeo import ogr
import pygeoprocessing
import taskgraph
import numpy
import shapely.wkb
import shapely.prepared
import rtree

from .. import validation
from .. import utils
from . import urban_heat_island_mitigation_core

LOGGER = logging.getLogger(__name__)
TARGET_NODATA = -1
_LOGGING_PERIOD = 5.0


def execute(args):
    """Urban Flood Heat Island Mitigation model.

    Parameters:
        args['workspace_dir'] (str): path to target output directory.
        args['t_air_ref_raster_path'] (str): raster of air temperature.
        args['lulc_raster_path'] (str): path to landcover raster.
        args['ref_eto_raster_path'] (str): path to evapotranspiration raster.
        args['et_max'] (float): maximum evapotranspiration.
        args['aoi_vector_path'] (str): path to desired AOI.
        args['biophysical_table_path'] (str): table to map landcover codes to
            Shade, Kc, and Albedo values. Must contain the fields 'lucode',
            'shade', 'kc', and 'albedo', and 'green_area'.
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

    task_graph = taskgraph.TaskGraph(temporary_working_dir, -1)

    # align all the input rasters.
    aligned_t_air_ref_raster_path = os.path.join(
        temporary_working_dir, 't_air_ref.tif')
    aligned_lulc_raster_path = os.path.join(
        temporary_working_dir, 'lulc.tif')
    aligned_ref_eto_raster_path = os.path.join(
        temporary_working_dir, 'ref_eto.tif')

    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_raster_path'])

    aligned_raster_path_list = [
        aligned_t_air_ref_raster_path, aligned_lulc_raster_path,
        aligned_ref_eto_raster_path]
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            [args['t_air_ref_raster_path'], args['lulc_raster_path'],
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

    for prop in ['kc', 'shade', 'albedo', 'green_area']:
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

    target_blob_id_raster_path = os.path.join(
        temporary_working_dir, 'green_blob_id.tif')
    id_count_map_pickle_path = os.path.join(
        temporary_working_dir, 'green_blob_map.pickle')

    blob_green_task = task_graph.add_task(
        func=urban_heat_island_mitigation_core.blob_mask,
        args=(
            (task_path_prop_map['green_area'][1], 1),
            target_blob_id_raster_path, id_count_map_pickle_path),
        target_path_list=[target_blob_id_raster_path],
        dependent_task_list=[task_path_prop_map['green_area'][0]],
        task_name='blob green mask')

    task_graph.close()
    task_graph.join()
    return

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
        args['t_air_ref_raster_path'])['nodata'][0]
    t_air_raster_path = os.path.join(args['workspace_dir'], 'T_air.tif')
    t_air_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (aligned_t_air_ref_raster_path, 1), (air_temp_nodata, 'raw'),
            (cc_raster_path, 1), (float(args['uhi_max']), 'raw')],
            calc_t_air_op, t_air_raster_path, gdal.GDT_Float32,
            TARGET_NODATA),
        target_path_list=[t_air_raster_path],
        dependent_task_list=[cc_task, align_task],
        task_name='calculate T air')

    intermediate_building_vector_path = os.path.join(
        temporary_working_dir, 'intermediate_building_vector.gpkg')
    # this is the field name that can be used to uniquely identify a feature
    intermediate_building_vector_task = task_graph.add_task(
        func=pygeoprocessing.reproject_vector,
        args=(
            args['building_vector_path'], lulc_raster_info['projection'],
            intermediate_building_vector_path),
        kwargs={'driver_name': 'GPKG'},
        target_path_list=[intermediate_building_vector_path],
        task_name='reproject building vector')

    # zonal stats over buildings for t_air
    t_air_stats_pickle_path = os.path.join(
        temporary_working_dir, 't_air_stats.pickle')
    pickle_t_air_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_building_vector_path,
            t_air_raster_path, t_air_stats_pickle_path),
        target_path_list=[t_air_stats_pickle_path],
        dependent_task_list=[t_air_task, intermediate_building_vector_task],
        task_name='pickle t-air stats')

    t_ref_stats_pickle_path = os.path.join(
        temporary_working_dir, 't_ref_stats.pickle')
    pickle_t_ref_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_building_vector_path,
            aligned_t_air_ref_raster_path, t_ref_stats_pickle_path),
        target_path_list=[t_ref_stats_pickle_path],
        dependent_task_list=[align_task, intermediate_building_vector_task],
        task_name='pickle t-ref stats')

    energy_consumption_vector_path = os.path.join(
        args['workspace_dir'], 'buildings_with_stats.gpkg')
    calculate_energy_savings_task = task_graph.add_task(
        func=calculate_energy_savings,
        args=(
            t_air_stats_pickle_path, t_ref_stats_pickle_path,
            float(args['uhi_max']), args['energy_consumption_table_path'],
            intermediate_building_vector_path,
            energy_consumption_vector_path),
        target_path_list=[energy_consumption_vector_path],
        dependent_task_list=[
            pickle_t_ref_task, pickle_t_air_task,
            intermediate_building_vector_task],
        task_name='calculate energy savings task')

    intermediate_aoi_vector_path = os.path.join(
        temporary_working_dir, 'intermediate_aoi.gpkg')
    intermediate_uhi_result_vector_task = task_graph.add_task(
        func=pygeoprocessing.reproject_vector,
        args=(
            args['aoi_vector_path'], lulc_raster_info['projection'],
            intermediate_aoi_vector_path),
        kwargs={'driver_name': 'GPKG'},
        target_path_list=[intermediate_aoi_vector_path],
        task_name='reproject and label aoi')

    cc_aoi_stats_pickle_path = os.path.join(
        temporary_working_dir, 'cc_ref_aoi_stats.pickle')
    pickle_cc_aoi_stats_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_aoi_vector_path,
            cc_raster_path, cc_aoi_stats_pickle_path),
        target_path_list=[cc_aoi_stats_pickle_path],
        dependent_task_list=[cc_task, intermediate_uhi_result_vector_task],
        task_name='pickle cc ref stats')

    t_air_ref_aoi_stats_pickle_path = os.path.join(
        temporary_working_dir, 't_ref_aoi_stats.pickle')
    pickle_t_air_ref_aoi_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_aoi_vector_path,
            aligned_t_air_ref_raster_path, t_air_ref_aoi_stats_pickle_path),
        target_path_list=[t_air_ref_aoi_stats_pickle_path],
        dependent_task_list=[align_task, intermediate_uhi_result_vector_task],
        task_name='pickle t-ref stats')

    t_air_aoi_stats_pickle_path = os.path.join(
        temporary_working_dir, 't_air_aoi_stats.pickle')
    pickle_t_air_aoi_task = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            intermediate_aoi_vector_path,
            t_air_raster_path, t_air_aoi_stats_pickle_path),
        target_path_list=[t_air_aoi_stats_pickle_path],
        dependent_task_list=[t_air_task, intermediate_uhi_result_vector_task],
        task_name='pickle t-air stats')

    target_uhi_vector_path = os.path.join(
        args['workspace_dir'], 'uhi_results.gpkg')
    calculate_uhi_result_task = task_graph.add_task(
        func=calculate_uhi_result_vector,
        args=(
            intermediate_aoi_vector_path,
            t_air_aoi_stats_pickle_path, t_air_ref_aoi_stats_pickle_path,
            cc_aoi_stats_pickle_path,
            energy_consumption_vector_path,
            target_uhi_vector_path),
        target_path_list=[target_uhi_vector_path],
        dependent_task_list=[
            pickle_t_air_aoi_task, pickle_t_air_ref_aoi_task,
            pickle_cc_aoi_stats_task, calculate_energy_savings_task,
            intermediate_uhi_result_vector_task],
        task_name='calculate uhi results')

    task_graph.close()
    task_graph.join()


def calculate_uhi_result_vector(
        base_aoi_path, t_air_stats_pickle_path,
        t_air_ref_stats_pickle_path, cc_stats_pickle_path,
        energy_consumption_vector_path, target_uhi_vector_path):
    """Summarize UHI results.

    Output vector will have fields with attributes summarizing:
        * average cc value
        * average temperature value
        * average temperature anomaly
        * avoided energy consumption

    Parameters:
        base_aoi_path (str): path to AOI vector.
        energy_consumption_vector_path (str): path to vector that contains
            building footprints with the field 'energy_savings'.
        target_uhi_vector_path (str): path to UHI vector created for result.
            Will contain the fields:
                * average_cc_value
                * average_temp_value
                * average_temp_anom
                * avoided_energy_consumption

    Returns:
        None.

    """
    LOGGER.info(
        "Calculate UHI summary results %s", os.path.basename(
            target_uhi_vector_path))

    LOGGER.info("load t_air_stats")
    with open(t_air_stats_pickle_path, 'rb') as t_air_stats_pickle_file:
        t_air_stats = pickle.load(t_air_stats_pickle_file)
    LOGGER.info("load t_air_ref_stats")
    with open(t_air_ref_stats_pickle_path, 'rb') as \
            t_air_ref_stats_pickle_file:
        t_air_ref_stats = pickle.load(t_air_ref_stats_pickle_file)
    LOGGER.info("load cc_stats")
    with open(cc_stats_pickle_path, 'rb') as cc_stats_pickle_file:
        cc_stats = pickle.load(cc_stats_pickle_file)

    energy_consumption_vector = gdal.OpenEx(
        energy_consumption_vector_path, gdal.OF_VECTOR)
    energy_consumption_layer = energy_consumption_vector.GetLayer()

    LOGGER.info('parsing building footprint geometry')
    building_shapely_polygon_lookup = dict([
        (poly_feat.GetFID(),
         shapely.wkb.loads(poly_feat.GetGeometryRef().ExportToWkb()))
        for poly_feat in energy_consumption_layer])

    LOGGER.info("constructing building footprint spatial index")
    poly_rtree_index = rtree.index.Index(
        [(poly_fid, poly.bounds, None)
         for poly_fid, poly in building_shapely_polygon_lookup.items()])

    base_aoi_vector = gdal.OpenEx(base_aoi_path, gdal.OF_VECTOR)
    gpkg_driver = gdal.GetDriverByName('GPKG')
    LOGGER.info("creating %s", os.path.basename(target_uhi_vector_path))
    gpkg_driver.CreateCopy(
        target_uhi_vector_path, base_aoi_vector)
    base_aoi_vector = None
    target_uhi_vector = gdal.OpenEx(
        target_uhi_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_uhi_layer = target_uhi_vector.GetLayer()

    for field_id in [
            'average_cc_value', 'average_temp_value', 'average_temp_anom',
            'avoided_energy_consumption']:
        target_uhi_layer.CreateField(ogr.FieldDefn(field_id, ogr.OFTReal))

    target_uhi_layer.StartTransaction()
    for feature in target_uhi_layer:
        feature_id = feature.GetFID()
        if feature_id in cc_stats and cc_stats[feature_id]['count'] > 0:
            mean_cc = (
                cc_stats[feature_id]['sum'] / cc_stats[feature_id]['count'])
            feature.SetField('average_cc_value', mean_cc)
        mean_t_air = None
        if feature_id in t_air_stats and t_air_stats[feature_id]['count'] > 0:
            mean_t_air = (
                t_air_stats[feature_id]['sum'] /
                t_air_stats[feature_id]['count'])
            feature.SetField('average_temp_value', mean_t_air)

        mean_t_air_ref = None
        if feature_id in t_air_ref_stats and t_air_ref_stats[
                feature_id]['count'] > 0:
            mean_t_air_ref = (
                t_air_ref_stats[feature_id]['sum'] /
                t_air_ref_stats[feature_id]['count'])

        if mean_t_air and mean_t_air_ref:
            feature.SetField(
                'average_temp_anom', mean_t_air-mean_t_air_ref)

        aoi_geometry = feature.GetGeometryRef()
        aoi_shapely_geometry = shapely.wkb.loads(aoi_geometry.ExportToWkb())
        aoi_shapely_geometry_prep = shapely.prepared.prep(
            aoi_shapely_geometry)
        avoided_energy_consumption = 0.0
        for building_id in poly_rtree_index.intersection(
                aoi_shapely_geometry.bounds):
            if aoi_shapely_geometry_prep.intersects(
                    building_shapely_polygon_lookup[building_id]):
                energy_consumption_value = (
                    energy_consumption_layer.GetFeature(
                        building_id).GetField('energy_savings'))
                if energy_consumption_value:
                    # this step lets us skip values that might be in nodata
                    # ranges that we can't help.
                    avoided_energy_consumption += float(
                        energy_consumption_value)
        feature.SetField(
            'avoided_energy_consumption', avoided_energy_consumption)

        target_uhi_layer.SetFeature(feature)
    target_uhi_layer.CommitTransaction()


def calculate_energy_savings(
        t_air_stats_pickle_path, t_ref_stats_pickle_path, uhi_max,
        energy_consumption_table_path, base_building_vector_path,
        target_building_vector_path):
    """Add watershed scale values of the given base_raster.

    Parameters:
        t_air_stats_pickle_path (str): path to t_air zonal stats indexed by
            FID.
        t_ref_stats_pickle_path (str): path to t_ref zonal stats indexed by
            FID.
        uhi_max (float): UHI max parameter from documentation.
        base_building_vector_path (str): path to existing vector to copy for
            the target vector that contains at least the field 'type'.
        energy_consumption_table_path (str): path to energy consumption table
            that contains at least the columns 'type', and 'consumption'.
        target_building_vector_path (str): path to target vector that
            will contain the additional field 'energy_savings' calculated as
            consumption.increase(b) * ((T_(air,MAX)  - T_(air,i)))

    Return:
        None.

    """
    LOGGER.info(
        "Calculate energy savings for %s", target_building_vector_path)
    LOGGER.info("load t_air_stats")
    with open(t_air_stats_pickle_path, 'rb') as t_air_stats_pickle_file:
        t_air_stats = pickle.load(t_air_stats_pickle_file)
    LOGGER.info("load t_ref_stats")
    with open(t_ref_stats_pickle_path, 'rb') as t_ref_stats_pickle_file:
        t_ref_stats = pickle.load(t_ref_stats_pickle_file)

    base_building_vector = gdal.OpenEx(
        base_building_vector_path, gdal.OF_VECTOR)
    gpkg_driver = gdal.GetDriverByName('GPKG')
    LOGGER.info("creating %s", os.path.basename(target_building_vector_path))
    gpkg_driver.CreateCopy(
        target_building_vector_path, base_building_vector)
    base_building_vector = None
    target_building_vector = gdal.OpenEx(
        target_building_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_building_layer = target_building_vector.GetLayer()
    target_building_layer.CreateField(
        ogr.FieldDefn('energy_savings', ogr.OFTReal))
    target_building_layer.CreateField(
        ogr.FieldDefn('mean_t_air', ogr.OFTReal))
    target_building_layer.CreateField(
        ogr.FieldDefn('mean_t_ref', ogr.OFTReal))

    target_building_layer_defn = target_building_layer.GetLayerDefn()
    for field_name in ['Type', 'type', 'TYPE']:
        type_field_index = target_building_layer_defn.GetFieldIndex(
            field_name)
        if type_field_index != -1:
            break
    if type_field_index == -1:
        raise ValueError(
            "Could not find field 'Type' in %s", target_building_vector_path)

    energy_consumption_table = utils.build_lookup_from_csv(
        energy_consumption_table_path, 'type', to_lower=True,
        warn_if_missing=True)

    target_building_layer.StartTransaction()
    last_time = time.time()
    for target_index, target_feature in enumerate(target_building_layer):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "energy savings approximately %.1f%% complete ",
                100.0 * float(target_index+1) /
                target_building_layer.GetFeatureCount()),
            _LOGGING_PERIOD)
        feature_id = target_feature.GetFID()
        t_air_mean = None
        if feature_id in t_air_stats:
            pixel_count = t_air_stats[feature_id]['count']
            if pixel_count > 0:
                t_air_mean = (
                    t_air_stats[feature_id]['sum'] /
                    float(pixel_count))
                target_feature.SetField('mean_t_air', float(t_air_mean))

        t_ref_mean = None
        if feature_id in t_ref_stats:
            pixel_count = t_ref_stats[feature_id]['count']
            if pixel_count > 0:
                t_ref_mean = (
                    t_ref_stats[feature_id]['sum'] /
                    float(pixel_count))
                target_feature.SetField('mean_t_ref', float(t_ref_mean))

        target_type = target_feature.GetField(int(type_field_index))
        if target_type not in energy_consumption_table:
            raise ValueError(
                "Encountered a building 'type' of: '%s' in "
                "FID: %d in the building vector layer that has no "
                "corresponding entry in the energy consumption table "
                "at %s" % (
                    target_type, target_feature.GetFID(),
                    energy_consumption_table_path))
        consumption_increase = float(
            energy_consumption_table[target_type]['consumption'])
        if t_air_mean and t_ref_mean:
            target_feature.SetField(
                'energy_savings', consumption_increase * (
                    t_ref_mean-t_air_mean + uhi_max))

        target_building_layer.SetFeature(target_feature)
    target_building_layer.CommitTransaction()
    target_building_layer.SyncToDisk()


def pickle_zonal_stats(
        base_vector_path, base_raster_path, target_pickle_path):
    """Calculate Zonal Stats for a vector/raster pair and pickle result.

    Parameters:
        base_vector_path (str): path to vector file
        base_raster_path (str): path to raster file to aggregate over.
        target_pickle_path (str): path to desired target pickle file that will
            be a pickle of the pygeoprocessing.zonal_stats function.

    Returns:
        None.

    """
    zonal_stats = pygeoprocessing.zonal_statistics(
        (base_raster_path, 1), base_vector_path,
        polygons_might_overlap=True)
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
        't_air_ref_raster_path',
        'lulc_raster_path',
        'ref_eto_raster_path',
        'et_max',
        'aoi_vector_path',
        'biophysical_table_path',
        'urban_park_cooling_distance',
        'uhi_max',
        'building_vector_path',
        'energy_consumption_table_path',
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
        ('t_air_ref_raster_path', 'raster'),
        ('lulc_raster_path', 'raster'),
        ('ref_eto_raster_path', 'raster'),
        ('aoi_vector_path', 'vector'),
        ('building_vector_path', 'vector'),
        ('biophysical_table_path', 'table'),
        ('energy_consumption_table_path', 'table'),
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
        if limit_to in ['biophysical_table_path', None]:
            try:
                biophysical_table = pygeoprocessing.build_lookup_from_csv(
                    args['biophysical_table_path'], 'lucode')
            except ValueError as e:
                validation_error_list(
                    [key], 'lucode might not be defined (%s)' % e)
            table_columns = next(biophysical_table.keys())
            for column_id in ['shade', 'kc', 'albedo']:
                if column_id not in table_columns:
                    validation_error_list(
                        [key], '"%s" expected but not a header this table' % (
                            column_id))
        if limit_to in ['energy_consumption_table_path', None]:
            try:
                energy_consumption_table = (
                    pygeoprocessing.build_lookup_from_csv(
                        args['energy_consumption_table_path'], 'type'))
            except ValueError as e:
                validation_error_list(
                    [key], 'type might not be defined (%s)' % e)
            table_columns = next(energy_consumption_table.keys())
            for column_id in ['consumption']:
                if column_id not in table_columns:
                    validation_error_list(
                        [key], '"%s" expected but not a header this table')
        if limit_to in ['energy_consumption_table_path', None]:
            building_vector = gdal.OpenEx(
                args['building_vector_path'], gdal.OF_VECTOR)
            building_layer = building_vector.GetLayer()
            building_layer_defn = building_layer.GetLayerDefn()
            for field_id in ['type', 'consumption']:
                type_index = building_layer_defn.GetFieldIndex('type')
                if type_index < 0:
                    validation_error_list(
                        [key], '"type" field expected in %s '
                        'but not defined' % (args['building_vector_path']))
                else:
                    for feature in building_layer:
                        raw_val = feature.GetField(type_index)
                        try:
                            _ = float(raw_val)
                        except TypeError:
                            validation_error_list(
                                [key],
                                'feature "type" fields of %s should be floating '
                                'point numbers, but at least one is not. '
                                '(raw val: %s)' % (
                                    args['building_vector_path'], raw_val))

    return validation_error_list


def _invoke_timed_callback(
        reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Parameters:
        reference_time (float): time to base `callback_period` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and `reference_time` has exceeded `callback_period`.
        callback_period (float): time in seconds to pass until
            `callback_lambda` is invoked.

    Returns:
        `reference_time` if `callback_lambda` not invoked, otherwise the time
        when `callback_lambda` was invoked.

    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time
