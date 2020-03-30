"""Urban Flood Risk Mitigation model."""
import logging
import os
import time
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

ARGS_SPEC = {
    "model_name": "Urban Flood Risk Mitigation",
    "module": __name__,
    "userguide_html": "urban_flood_risk_mitigation.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["aoi_watersheds_path", "lulc_path",
                         "built_infrastructure_vector_path",
                         "soils_hydrological_group_raster_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "aoi_watersheds_path": {
            "type": "vector",
            "required": True,
            "about": (
                "Path to a vector of (sub)watersheds or sewersheds used to "
                "indicate spatial area of interest."),
            "name": "Watershed Vector"
        },
        "rainfall_depth": {
            "validation_options": {
                "expression": "value > 0",
            },
            "type": "number",
            "required": True,
            "about": "Depth of rainfall in mm.",
            "name": "Depth of rainfall in mm"
        },
        "lulc_path": {
            "type": "raster",
            "validation_options": {
                "projected": True,
            },
            "required": True,
            "about": "Path to a landcover raster",
            "name": "Landcover Raster"
        },
        "soils_hydrological_group_raster_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "Raster with values equal to 1, 2, 3, 4, corresponding to "
                "soil hydrologic group A, B, C, or D, respectively (used to "
                "derive the CN number)"),
            "name": "Soils Hydrological Group Raster"
        },
        "curve_number_table_path": {
            "validation_options": {
                "required_fields": ["lucode", "CN_A", "CN_B", "CN_C", "CN_D"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "Path to a CSV table that to map landcover codes to curve "
                "numbers and contains at least the headers 'lucode', "
                "'CN_A', 'CN_B', 'CN_C', 'CN_D'"),
            "name": "Biophysical Table"
        },
        "built_infrastructure_vector_path": {
            "validation_options": {
                "required_fields": ["type"],
            },
            "type": "vector",
            "required": False,
            "about": (
                "Path to a vector with built infrastructure footprints. "
                "Attribute table contains a column 'Type' with integers "
                "(e.g. 1=residential, 2=office, etc.)."),
            "name": "Built Infrastructure Vector"
        },
        "infrastructure_damage_loss_table_path": {
            "validation_options": {
                "required_fields": ["type", "damage"],
            },
            "type": "csv",
            "required": "built_infrastructure_vector_path",
            "about": (
                "Path to a a CSV table with columns 'Type' and 'Damage' with "
                "values of built infrastructure type from the 'Type' field "
                "in the 'Built Infrastructure Vector' and potential damage "
                "loss (in $/m^2). Required if the built infrastructure vector "
                "is provided."),
            "name": "Built Infrastructure Damage Loss Table"
        }
    }
}




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
        args['built_infrastructure_vector_path'] (string): (optional) path to
            a vector with built infrastructure footprints. Attribute table
            contains a column 'Type' with integers (e.g. 1=residential,
            2=office, etc.).
        args['infrastructure_damage_loss_table_path'] (string): (optional)
            path to a a CSV table with columns 'Type' and 'Damage' with values
            of built infrastructure type from the 'Type' field in
            `args['built_infrastructure_vector_path']` and potential damage
            loss (in $/m^2).
        args['n_workers'] (int): (optional) if present, indicates how many
            worker processes should be used in parallel processing. -1
            indicates single process mode, 0 is single process but
            non-blocking mode, and >= 1 is number of processes.

    Returns:
        None.

    """
    if 'built_infrastructure_vector_path' in args and (
            args['built_infrastructure_vector_path'] != ''):
        infrastructure_damage_loss_table_path = (
            args['infrastructure_damage_loss_table_path'])
    else:
        infrastructure_damage_loss_table_path = None

    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    temporary_working_dir = os.path.join(
        args['workspace_dir'], 'temp_working_dir_not_for_humans')
    intermediate_dir = os.path.join(
        args['workspace_dir'], 'intermediate_files')
    utils.make_directories([
        args['workspace_dir'], intermediate_dir, temporary_working_dir])

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(temporary_working_dir, n_workers)

    # Align LULC with soils
    aligned_lulc_path = os.path.join(
        temporary_working_dir, 'aligned_lulc%s.tif' % file_suffix)
    aligned_soils_path = os.path.join(
        temporary_working_dir,
        'aligned_soils_hydrological_group%s.tif' % file_suffix)

    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_path'])
    target_pixel_size = lulc_raster_info['pixel_size']
    pixel_area = abs(target_pixel_size[0] * target_pixel_size[1])
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

    cn_raster_path = os.path.join(
        temporary_working_dir, 'cn_raster%s.tif' % file_suffix)
    align_raster_stack_task.join()

    cn_raster_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(aligned_lulc_path, 1), (aligned_soils_path, 1),
             (lucode_nodata, 'raw'), (soil_type_nodata, 'raw'),
             (cn_nodata, 'raw'), (lucode_to_cn_table, 'raw')], _lu_to_cn_op,
            cn_raster_path, gdal.GDT_Float32, cn_nodata),
        target_path_list=[cn_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create cn raster')

    # Generate S_max
    s_max_nodata = -9999
    s_max_raster_path = os.path.join(
        temporary_working_dir, 's_max%s.tif' % file_suffix)
    s_max_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(cn_raster_path, 1), (cn_nodata, 'raw'), (s_max_nodata, 'raw')],
            _s_max_op, s_max_raster_path, gdal.GDT_Float32, s_max_nodata),
        target_path_list=[s_max_raster_path],
        dependent_task_list=[cn_raster_task],
        task_name='create s_max')

    # Generate Qpi
    q_pi_nodata = -9999.
    q_pi_raster_path = os.path.join(
        intermediate_dir, 'Q_mm%s.tif' % file_suffix)
    q_pi_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(float(args['rainfall_depth']), 'raw'), (s_max_raster_path, 1),
             (s_max_nodata, 'raw'), (q_pi_nodata, 'raw')], _q_pi_op,
            q_pi_raster_path, gdal.GDT_Float32, q_pi_nodata),
        target_path_list=[q_pi_raster_path],
        dependent_task_list=[s_max_task],
        task_name='create q_pi')

    # Generate Runoff Retention
    runoff_retention_nodata = -9999.
    runoff_retention_raster_path = os.path.join(
        args['workspace_dir'], 'Runoff_retention%s.tif' % file_suffix)
    runoff_retention_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (q_pi_raster_path, 1), (float(args['rainfall_depth']), 'raw'),
            (q_pi_nodata, 'raw'), (runoff_retention_nodata, 'raw')],
            _runoff_retention_op, runoff_retention_raster_path,
            gdal.GDT_Float32, runoff_retention_nodata),
        target_path_list=[runoff_retention_raster_path],
        dependent_task_list=[q_pi_task],
        task_name='generate runoff retention')

    # calculate runoff retention volumne
    runoff_retention_ret_vol_raster_path = os.path.join(
        args['workspace_dir'], 'Runoff_retention_m3%s.tif' % file_suffix)
    runoff_retention_ret_vol_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (runoff_retention_raster_path, 1),
            (runoff_retention_nodata, 'raw'),
            (float(args['rainfall_depth']), 'raw'),
            (abs(target_pixel_size[0]*target_pixel_size[1]), 'raw'),
            (runoff_retention_nodata, 'raw')], _runoff_retention_ret_vol_op,
            runoff_retention_ret_vol_raster_path, gdal.GDT_Float32,
            runoff_retention_nodata),
        target_path_list=[runoff_retention_ret_vol_raster_path],
        dependent_task_list=[runoff_retention_task],
        task_name='calculate runoff retention vol')

    # calculate flood vol raster
    flood_vol_raster_path = os.path.join(
        intermediate_dir, 'Q_m3%s.tif' % file_suffix)
    flood_vol_nodata = -1
    flood_vol_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(float(args['rainfall_depth']), 'raw'),
             (q_pi_raster_path, 1), (q_pi_nodata, 'raw'),
             (pixel_area, 'raw'), (flood_vol_nodata, 'raw')],
            _flood_vol_op, flood_vol_raster_path, gdal.GDT_Float32,
            flood_vol_nodata),
        target_path_list=[flood_vol_raster_path],
        dependent_task_list=[q_pi_task],
        task_name='calculate service built raster')

    if 'built_infrastructure_vector_path' not in args or (
            args['built_infrastructure_vector_path'] in ('', None)):
        task_graph.close()
        task_graph.join()
        return

    # intersect built_infrastructure_vector_path with aoi_watersheds_path
    intermediate_target_watershed_result_vector_path = os.path.join(
        temporary_working_dir,
        'intermediate_flood_risk_service%s.gpkg' % file_suffix)

    # this is the field name that can be used to uniquely identify a feature
    intermediate_affected_vector_task = task_graph.add_task(
        func=_build_affected_vector,
        args=(
            args['aoi_watersheds_path'], target_sr_wkt,
            infrastructure_damage_loss_table_path,
            args['built_infrastructure_vector_path'],
            intermediate_target_watershed_result_vector_path),
        target_path_list=[intermediate_target_watershed_result_vector_path],
        task_name='build affected vector')

    # do the pickle
    runoff_retention_pickle_path = os.path.join(
        temporary_working_dir,
        'runoff_retention_stats%s.pickle' % file_suffix)
    runoff_retention_pickle_task = task_graph.add_task(
        func=_pickle_zonal_stats,
        args=(
            intermediate_target_watershed_result_vector_path,
            runoff_retention_raster_path, runoff_retention_pickle_path),
        dependent_task_list=[
            intermediate_affected_vector_task, runoff_retention_task],
        target_path_list=[runoff_retention_pickle_path],
        task_name='pickle runoff index stats')

    runoff_retention_ret_vol_pickle_path = os.path.join(
        temporary_working_dir,
        'runoff_retention_ret_vol_stats%s.pickle' % file_suffix)
    runoff_retention_ret_vol_pickle_task = task_graph.add_task(
        func=_pickle_zonal_stats,
        args=(
            intermediate_target_watershed_result_vector_path,
            runoff_retention_ret_vol_raster_path,
            runoff_retention_ret_vol_pickle_path),
        dependent_task_list=[
            intermediate_affected_vector_task, runoff_retention_ret_vol_task],
        target_path_list=[runoff_retention_ret_vol_pickle_path],
        task_name='pickle runoff retention volume stats')

    flood_vol_pickle_path = os.path.join(
        temporary_working_dir, 'flood_vol_stats%s.pickle' % file_suffix)
    flood_vol_pickle_task = task_graph.add_task(
        func=_pickle_zonal_stats,
        args=(
            intermediate_target_watershed_result_vector_path,
            flood_vol_raster_path, flood_vol_pickle_path),
        dependent_task_list=[
            intermediate_affected_vector_task, flood_vol_task],
        target_path_list=[flood_vol_pickle_path],
        task_name='pickle flood volume stats')

    target_watershed_result_vector_path = os.path.join(
        args['workspace_dir'], 'flood_risk_service%s.shp' % file_suffix)

    task_graph.add_task(
        func=_add_zonal_stats,
        args=(
            runoff_retention_pickle_path,
            runoff_retention_ret_vol_pickle_path,
            flood_vol_pickle_path,
            intermediate_target_watershed_result_vector_path,
            target_watershed_result_vector_path),
        target_path_list=[target_watershed_result_vector_path],
        dependent_task_list=[
            flood_vol_pickle_task, runoff_retention_ret_vol_pickle_task,
            runoff_retention_pickle_task, intermediate_affected_vector_task],
        task_name='add zonal stats')

    task_graph.close()
    task_graph.join()


def _pickle_zonal_stats(
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
        (base_raster_path, 1), base_vector_path)
    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(zonal_stats, pickle_file)


def _flood_vol_op(
        rainfall_depth, q_pi_array, q_pi_nodata, pixel_area, target_nodata):
    """Calculate vol of flood water.

    Parmeters:
        rainfall_depth (float): depth of rainfal in mm.
        q_pi_array (numpy.ndarray): quick flow array.
        q_pi_nodata (float): nodata for q_pi.
        pixel_area (float): area of pixel in m^2.
        target_nodata (float): output nodata value.

    Returns:
        numpy array of flood volume per pixel in m^3.

    """
    result = numpy.empty(q_pi_array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = q_pi_array != q_pi_nodata
    # 0.001 converts mm (rainfall depth) to m (pixel area units)
    result[valid_mask] = (
        0.001 * (rainfall_depth - q_pi_array[valid_mask]) * pixel_area)
    return result


def _runoff_retention_ret_vol_op(
        runoff_retention_array, runoff_retention_nodata, p_value,
        cell_area, target_nodata):
    """Calculate peak flow retention as a vol.

    Parameters:
        runoff_retention_array (numpy.ndarray): proportion of pixel retention.
        runoff_retention_nodata (float): nodata value for corresponding array.
        p_value (float): precipitation depth in mm.
        cell_area (float): area of cell in m^2.
        target_nodata (float): target nodata to write.

    Returns:
        (R_i*Qpi*p_val*cell_size/1000.)

    """
    result = numpy.empty(runoff_retention_array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = runoff_retention_array != runoff_retention_nodata
    # the 1e-3 converts the mm of p_value to meters.
    result[valid_mask] = (
        runoff_retention_array[valid_mask] * p_value * cell_area * 1e-3)
    return result


def _add_zonal_stats(
        runoff_retention_pickle_path,
        runoff_retention_ret_vol_pickle_path,
        flood_vol_pickle_path,
        base_watershed_result_vector_path,
        target_watershed_result_vector_path):
    """Add watershed scale values of the given base_raster.

    Parameters:
        runoff_retention_pickle_path (str): path to runoff retention
            zonal stats pickle file.
        runoff_retention_ret_vol_pickle_path (str): path to runoff
            retention volume zonal stats pickle file.
        flood_vol_pickle_path (str): path to flood volume zonal stats
            pickle file.
        base_watershed_result_vector_path (str): path to existing vector
            to copy for the target vector.
        target_watershed_result_vector_path (str): path to target vector that
            will contain the additional fields:
                * rnf_rt_idx
                * rnf_rt_m3
                * serv_bld

    Return:
        None.

    """
    LOGGER.info(
        "Processing zonal stats for %s", target_watershed_result_vector_path)

    with open(runoff_retention_pickle_path, 'rb') as runoff_retention_file:
        runoff_retention_stats = pickle.load(runoff_retention_file)
    with open(runoff_retention_ret_vol_pickle_path, 'rb') as (
            runoff_retention_ret_vol_file):
        runoff_retention_vol_stats = pickle.load(runoff_retention_ret_vol_file)
    with open(flood_vol_pickle_path, 'rb') as flood_vol_pickle_file:
        flood_vol_stats = pickle.load(flood_vol_pickle_file)

    base_sr_wkt = pygeoprocessing.get_vector_info(
        base_watershed_result_vector_path)['projection']
    base_watershed_vector = gdal.OpenEx(
        base_watershed_result_vector_path, gdal.OF_VECTOR)
    base_watershed_layer = base_watershed_vector.GetLayer()
    base_geom_type = base_watershed_layer.GetGeomType()
    base_sr = osr.SpatialReference()
    base_sr.ImportFromWkt(base_sr_wkt)

    if os.path.exists(target_watershed_result_vector_path):
        LOGGER.warn(
            "deleting existing target result at %s",
            target_watershed_result_vector_path)
        os.remove(target_watershed_result_vector_path)
    esri_driver = gdal.GetDriverByName('ESRI Shapefile')
    target_watershed_vector = esri_driver.Create(
        target_watershed_result_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    layer_name = str(os.path.splitext(os.path.basename(
            target_watershed_result_vector_path))[0])
    LOGGER.debug("creating layer %s", layer_name)
    target_watershed_layer = target_watershed_vector.CreateLayer(
        str(layer_name), base_sr, base_geom_type)

    for field_name in ['aff_bld', 'rnf_rt_idx', 'rnf_rt_m3', 'serv_bld']:
        field_def = ogr.FieldDefn(field_name, ogr.OFTReal)
        field_def.SetWidth(24)
        field_def.SetPrecision(11)
        target_watershed_layer.CreateField(field_def)

    target_layer_defn = target_watershed_layer.GetLayerDefn()

    for base_feature in base_watershed_layer:
        feature_id = base_feature.GetFID()
        target_feature = ogr.Feature(target_layer_defn)
        base_geom_ref = base_feature.GetGeometryRef()
        target_feature.SetGeometry(base_geom_ref.Clone())
        base_geom_ref = None

        if feature_id in runoff_retention_stats:
            pixel_count = runoff_retention_stats[feature_id]['count']
            if pixel_count > 0:
                mean_value = (
                    runoff_retention_stats[feature_id]['sum'] /
                    float(pixel_count))
                target_feature.SetField(
                    'rnf_rt_idx', float(mean_value))

        if feature_id in runoff_retention_vol_stats:
            target_feature.SetField(
                'rnf_rt_m3', float(
                    runoff_retention_vol_stats[feature_id]['sum']))

        if feature_id in flood_vol_stats:
            pixel_count = flood_vol_stats[feature_id]['count']
            if pixel_count > 0:
                affected_build = base_feature.GetField('aff_bld')
                target_feature.SetField('aff_bld', affected_build)
                target_feature.SetField(
                    'serv_bld', affected_build * float(
                        runoff_retention_vol_stats[feature_id]['sum']))

        target_watershed_layer.CreateFeature(target_feature)
    target_watershed_layer.SyncToDisk()
    target_watershed_layer = None
    target_watershed_vector = None


def _build_affected_vector(
        base_watershed_vector_path, target_wkt, damage_table_path,
        built_infrastructure_vector_path,
        target_watershed_result_vector_path):
    """Construct the affected area vector.

    The ``base_watershed_vector_path`` will be intersected with the
    ``built_infrastructure_vector_path`` to get the affected build area.

    Parameters:
        base_watershed_vector_path (str): path to base watershed vector,
        target_wkt (str): desired target projection.
        damage_table_path (None or str): path to a CSV table containing fields
            'Type' and 'Damage'. For every value of 'Type' in the
            built_infrastructure_vector there must be a corresponding entry
            in this table. If None, this field is ignored.
        built_infrastructure_vector_path (str): path to infrastructure vector
            containing at least the integer field 'Type'.
        target_watershed_result_vector_path (str): path to desired target
            watershed result vector that will have an additional field called
            'aff_bld'.

    Returns:
        None.

    """
    if damage_table_path is not None and damage_table_path != '':
        damage_type_map = utils.build_lookup_from_csv(
            damage_table_path, 'type', to_lower=True, warn_if_missing=True)
    else:
        damage_type_map = None

    if os.path.exists(target_watershed_result_vector_path):
        LOGGER.warn(
            '%s exists, removing to make a current one',
            target_watershed_result_vector_path)
        os.remove(target_watershed_result_vector_path)

    pygeoprocessing.reproject_vector(
        base_watershed_vector_path, target_wkt,
        target_watershed_result_vector_path,
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
        })
        if damage_type_map is not None:
            infrastructure_geometry_list[-1]['damage'] = (
                damage_type_map[
                    infrastructure_feature.GetField(type_index)]['damage'])
        infrastructure_rtree.insert(
            len(infrastructure_geometry_list)-1,
            infrastructure_geometry_list[-1]['geom'].bounds)

    infrastructure_vector = None
    infrastructure_layer = None

    watershed_vector = gdal.OpenEx(
        target_watershed_result_vector_path, gdal.OF_VECTOR | gdal.OF_UPDATE)
    watershed_layer = watershed_vector.GetLayer()
    watershed_layer.CreateField(ogr.FieldDefn('aff_bld', ogr.OFTReal))
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
        total_damage = 0.0
        for infrastructure_index in infrastructure_rtree.intersection(
                watershed_shapely.bounds):
            infrastructure_geom = infrastructure_geometry_list[
                infrastructure_index]['geom']
            if damage_type_map:
                if watershed_prep_geom.intersects(infrastructure_geom):
                    total_damage += (
                        watershed_shapely.intersection(
                            infrastructure_geom).area *
                        infrastructure_geometry_list[infrastructure_index][
                            'damage'])

        if damage_type_map:
            watershed_feature.SetField('aff_bld', total_damage)
        watershed_layer.SetFeature(watershed_feature)
    watershed_layer.SyncToDisk()
    watershed_layer = None
    watershed_vector = None


def _runoff_retention_op(q_pi_array, p_value, q_pi_nodata, result_nodata):
    """Calculate peak flow retention.

    Parameters:
        q_pi_array (numpy.ndarray): quick flow array.
        p_value (float): precipition in mm.
        q_pi_nodata (float): nodata for q_pi.
        pixel_area (float): area of pixel in m^2.
        target_nodata (float): output nodata value.

    Returns:
        1.0 - q_pi/p

    """
    result = numpy.empty_like(q_pi_array)
    result[:] = result_nodata
    valid_mask = numpy.ones(q_pi_array.shape, dtype=numpy.bool)
    if q_pi_nodata:
        valid_mask[:] = ~numpy.isclose(q_pi_array, q_pi_nodata)
    result[valid_mask] = 1.0 - q_pi_array[valid_mask] / p_value
    return result


def _q_pi_op(p_value, s_max_array, s_max_nodata, result_nodata):
    """Calculate peak flow Q (mm) with the Curve Number method.

    Parameters:
        p_value (float): precipitation in mm.
        s_max_array (numpy.ndarray): max S value per pixel.
        s_max_nodata (float): nodata value for s_max_array.
        result_nodata (float): return value nodata.

    Returns:
        ndarray of peak flow.

    """
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


def _s_max_op(cn_array, cn_nodata, result_nodata):
    """Calculate S_max from the curve number.

    Parameters:
        cn_array (numpy.ndarray): curve number array.
        cn_nodata (float): nodata value for cn_array.
        result_nodata (float): output nodata value.

    Return:
        ndarray of Smax calcualted from curve number.

    """
    result = numpy.empty_like(cn_array)
    result[:] = result_nodata
    zero_mask = cn_array == 0
    valid_mask = ~zero_mask
    if cn_nodata:
        valid_mask[:] &= ~numpy.isclose(cn_array, cn_nodata)
    result[valid_mask] = 25400.0 / cn_array[valid_mask] - 254.0
    result[zero_mask] = 0.0
    return result


def _lu_to_cn_op(
        lucode_array, soil_type_array, lucode_nodata, soil_type_nodata,
        cn_nodata, lucode_to_cn_table):
    """Map combination landcover soil type map to curve number raster.

    Parameters:
        lucode_array (numpy.ndarray): array of landcover codes.
        soil_type_array (numpy.ndarray): array of soil type values.
        lucode_nodata  (float): nodata value for corresponding array.
        soil_type_nodata (float): nodata value for corresponding array.
        cn_nodata (float): nodata value for return value array.
        lucode_to_cn_table (scipy.sparse.csr.csr_matrix):

    Returns:
        ndarray of curve numbers by looking up landcover type to soil type
        to then soil value.

    """
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
        lucode_to_cn_table[lucode_array[valid_mask].astype(int)].toarray().reshape(
            (-1, 4))).transpose()

    # this is the soil type array with values ranging from 0..4 that will
    # choose the appropriate row for each pixel colum in
    # `per_pixel_cn_array`
    soil_choose_array = (
        soil_type_array[valid_mask].astype(numpy.int8))-1

    # soil arrays are 1.0 - 4.0, remap to 0 - 3 and choose from the per
    # pixel CN array
    try:
        result[valid_mask] = numpy.choose(
            soil_choose_array,
            per_pixel_cn_array)
    except ValueError as error:
        err_msg = 'Check that the Soil Group raster does not contain values ' \
            'other than (1, 2, 3, 4)'
        raise ValueError(str(error) + '\n' + err_msg)

    return result


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Parameters:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
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
    return validation.validate(args, ARGS_SPEC['args'],
                               ARGS_SPEC['args_with_spatial_overlap'])
