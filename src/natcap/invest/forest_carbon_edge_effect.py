"""InVEST Carbon Edge Effect Model.

An implementation of the model described in 'Degradation in carbon stocks
near tropical forest edges', by Chaplin-Kramer et. al (2015).
"""
import os
import logging
import time
import uuid

import pickle
import numpy
from osgeo import gdal
from osgeo import ogr
import pygeoprocessing
import scipy.spatial
import taskgraph

from . import utils
from . import spec_utils
from .spec_utils import u
from . import validation
from .model_metadata import MODEL_METADATA
from . import gettext


LOGGER = logging.getLogger(__name__)

# grid cells are 100km. Becky says 500km is a good upper bound to search
DISTANCE_UPPER_BOUND = 500e3

# helpful to have a global nodata defined for the whole model
NODATA_VALUE = -1

ARGS_SPEC = {
    "model_name": MODEL_METADATA["forest_carbon_edge_effect"].model_title,
    "pyname": MODEL_METADATA["forest_carbon_edge_effect"].pyname,
    "userguide": MODEL_METADATA["forest_carbon_edge_effect"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["aoi_vector_path", "lulc_raster_path"],
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "n_nearest_model_points": {
            "expression": "value > 0 and value.is_integer()",
            "type": "number",
            "units": u.none,
            "required": "compute_forest_edge_effects",
            "about": gettext(
                "Number of closest regression models that are used when "
                "calculating the total biomass. Each local model is linearly "
                "weighted by distance such that the pixel's biomass is a "
                "function of each of these points with the closest point "
                "having the largest effect. Must be an integer greater than "
                "0. Required if Compute Forest Edge Effects is selected."
            ),
            "name": gettext("number of points to average")
        },
        "aoi_vector_path": {
            **spec_utils.AOI,
            "projected": True,
            "required": False
        },
        "biophysical_table_path": {
            "type": "csv",
            "columns": {
                "lucode": {
                    "type": "integer",
                    "about": gettext(
                        "Code for this LULC class from the LULC map. Every "
                        "value in the LULC raster must have a corresponding "
                        "entry in this column.")},
                "is_tropical_forest": {
                    "type": "boolean",
                    "about": gettext(
                        "Enter 1 if the LULC class is tropical forest, 0 if "
                        "it is not tropical forest.")},
                "c_above": {
                    "type": "number",
                    "units": u.metric_ton/u.hectare,
                    "about": gettext(
                        "Carbon density value for the aboveground carbon "
                        "pool.")
                },
                "c_below": {
                    "type": "number",
                    "units": u.metric_ton/u.hectare,
                    "required": "pools_to_calculate == 'all'",
                    "about": gettext(
                        "Carbon density value for the belowground carbon "
                        "pool. Required if calculating all pools.")
                },
                "c_soil": {
                    "type": "number",
                    "units": u.metric_ton/u.hectare,
                    "required": "pools_to_calculate == 'all'",
                    "about": gettext(
                        "Carbon density value for the soil carbon pool. "
                        "Required if calculating all pools.")
                },
                "c_dead": {
                    "type": "number",
                    "units": u.metric_ton/u.hectare,
                    "required": "pools_to_calculate == 'all'",
                    "about": gettext(
                        "Carbon density value for the dead matter carbon "
                        "pool. Required if calculating all pools.")
                },
            },
            "about": gettext(
                "A table mapping each LULC code from the LULC map to "
                "biophysical data for that LULC class."),
            "name": gettext("biophysical table")
        },
        "lulc_raster_path": {
            **spec_utils.LULC,
            "about": gettext(
                f"{spec_utils.LULC['about']} All values in this raster must "
                "have corresponding entries in the Biophysical Table."),
            "projected": True
        },
        "pools_to_calculate": {
            "type": "option_string",
            "options": {
                "all": {
                    "display_name": gettext("all"),
                    "description": gettext(
                        "Use all pools (aboveground, belowground, soil, and "
                        "dead matter) in the carbon pool calculation.")},
                "above_ground": {
                    "display_name": gettext("aboveground only"),
                    "description": gettext(
                        "Only use the aboveground pool in the carbon pool "
                        "calculation.")}
            },
            "about": gettext("Which carbon pools to consider."),
            "name": gettext("carbon pools to calculate")
        },
        "compute_forest_edge_effects": {
            "type": "boolean",
            "about": gettext("Account for forest edge effects on aboveground carbon."),
            "name": gettext("compute forest edge effects")
        },
        "tropical_forest_edge_carbon_model_vector_path": {
            "type": "vector",
            "fields": {
                "method": {
                    "type": "option_string",
                    "options": {
                        "1": {"description": gettext("asymptotic")},
                        "2": {"description": gettext("logarithmic")},
                        "3": {"description": gettext("linear")}
                    },
                    "about": gettext("Optimal regression model for the area.")
                },
                "theta1": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("θ₁ parameter for the regression equation.")},
                "theta2": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("θ₂ parameter for the regression equation.")},
                "theta3": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "θ₃ parameter for the regression equation. "
                        "Used only for the asymptotic model.")}
            },
            "geometries": spec_utils.POLYGONS,
            "required": "compute_forest_edge_effects",
            "about": gettext(
                "Map storing the optimal regression model for each tropical "
                "subregion and the corresponding theta parameters for that "
                "regression equation. Default data is provided. Required if "
                "Compute Forest Edge Effects is selected."),
            "name": gettext("global regression models")
        },
        "biomass_to_carbon_conversion_factor": {
            "type": "ratio",
            "required": "compute_forest_edge_effects",
            "about": gettext(
                "Proportion of forest edge biomass that is elemental carbon. "
                "Required if Compute Forest Edge Effects is selected."),
            "name": gettext("forest edge biomass to carbon conversion factor")
        }
    }
}


def execute(args):
    """Forest Carbon Edge Effect.

    InVEST Carbon Edge Model calculates the carbon due to edge effects in
    tropical forest pixels.

    Args:
        args['workspace_dir'] (string): a path to the directory that will write
            output and other temporary files during calculation. (required)
        args['results_suffix'] (string): a string to append to any output file
            name (optional)
        args['n_nearest_model_points'] (int): number of nearest neighbor model
            points to search for
        args['aoi_vector_path'] (string): (optional) if present, a path to a
            shapefile that will be used to aggregate carbon stock results at
            the end of the run.
        args['biophysical_table_path'] (string): a path to a CSV table that has
            at least the fields 'lucode' and 'c_above'. If
            ``args['compute_forest_edge_effects'] == True``, table must
            also contain an 'is_tropical_forest' field.  If
            ``args['pools_to_calculate'] == 'all'``, this table must contain
            the fields 'c_below', 'c_dead', and 'c_soil'.

                * ``lucode``: an integer that corresponds to landcover codes in
                  the raster ``args['lulc_raster_path']``
                * ``is_tropical_forest``: either 0 or 1 indicating whether the
                  landcover type is forest (1) or not (0).  If 1, the value
                  in ``c_above`` is ignored and instead calculated from the
                  edge regression model.
                * ``c_above``: floating point number indicating tons of above
                  ground carbon per hectare for that landcover type
                * ``{'c_below', 'c_dead', 'c_soil'}``: three other optional
                  carbon pools that will statically map landcover types to the
                  carbon densities in the table.

                Example::

                    lucode,is_tropical_forest,c_above,c_soil,c_dead,c_below
                    0,0,32.8,5,5.2,2.1
                    1,1,n/a,2.5,0.0,0.0
                    2,1,n/a,1.8,1.0,0.0
                    16,0,28.1,4.3,0.0,2.0

                Note the "n/a" in ``c_above`` are optional since that field
                is ignored when ``is_tropical_forest==1``.
        args['lulc_raster_path'] (string): path to a integer landcover code
            raster
        args['pools_to_calculate'] (string): if "all" then all carbon pools
            will be calculted.  If any other value only above ground carbon
            pools will be calculated and expect only a 'c_above' header in
            the biophysical table. If "all" model expects 'c_above',
            'c_below', 'c_dead', 'c_soil' in header of biophysical_table and
            will make a translated carbon map for each based off the landcover
            map.
        args['compute_forest_edge_effects'] (boolean): if True, requires
            biophysical table to have 'is_tropical_forest' forest field, and
            any landcover codes that have a 1 in this column calculate carbon
            stocks using the Chaplin-Kramer et. al method and ignore 'c_above'.
        args['tropical_forest_edge_carbon_model_vector_path'] (string):
            path to a shapefile that defines the regions for the local carbon
            edge models.  Has at least the fields 'method', 'theta1', 'theta2',
            'theta3'.  Where 'method' is an int between 1..3 describing the
            biomass regression model, and the thetas are floating point numbers
            that have different meanings depending on the 'method' parameter.
            Specifically,

                * method 1 (asymptotic model)::

                    biomass = theta1 - theta2 * exp(-theta3 * edge_dist_km)

                * method 2 (logarithmic model)::

                    # NOTE: theta3 is ignored for this method
                    biomass = theta1 + theta2 * numpy.log(edge_dist_km)

                * method 3 (linear regression)::

                    biomass = theta1 + theta2 * edge_dist_km
        args['biomass_to_carbon_conversion_factor'] (string/float): Number by
            which to multiply forest biomass to convert to carbon in the edge
            effect calculation.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None

    """
    # just check that the AOI exists since it wouldn't crash until the end of
    # the whole model run if it didn't.
    if 'aoi_vector_path' in args and args['aoi_vector_path'] != '':
        aoi_vector = gdal.OpenEx(args['aoi_vector_path'], gdal.OF_VECTOR)
        if not aoi_vector:
            raise ValueError(
                f"Unable to open aoi at: {args['aoi_vector_path']}")
        else:
            aoi_vector = None
            lulc_raster_bb = pygeoprocessing.get_raster_info(
                args['lulc_raster_path'])['bounding_box']
            aoi_vector_bb = pygeoprocessing.get_vector_info(
                args['aoi_vector_path'])['bounding_box']
            try:
                merged_bb = pygeoprocessing.merge_bounding_box_list(
                    [lulc_raster_bb, aoi_vector_bb], 'intersection')
                LOGGER.debug(f"merged bounding boxes: {merged_bb}")
            except ValueError:
                raise ValueError(
                    f"The landcover raster {args['lulc_raster_path']} and AOI "
                    f"{args['aoi_vector_path']} do not touch each other.")

    output_dir = args['workspace_dir']
    intermediate_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    utils.make_directories([output_dir, intermediate_dir])
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Initialize a TaskGraph
    taskgraph_working_dir = os.path.join(
        intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    task_graph = taskgraph.TaskGraph(taskgraph_working_dir, n_workers)

    # used to keep track of files generated by this module
    output_file_registry = {
        'c_above_map': os.path.join(
            intermediate_dir, f'c_above_carbon_stocks{file_suffix}.tif'),
        'carbon_map': os.path.join(
            output_dir, f'carbon_map{file_suffix}.tif'),
        'aggregated_result_vector': os.path.join(
            output_dir, f'aggregated_carbon_stocks{file_suffix}.shp')
    }

    if args['pools_to_calculate'] == 'all':
        output_file_registry['c_below_map'] = os.path.join(
            intermediate_dir, f'c_below_carbon_stocks{file_suffix}.tif')
        output_file_registry['c_soil_map'] = os.path.join(
            intermediate_dir, f'c_soil_carbon_stocks{file_suffix}.tif')
        output_file_registry['c_dead_map'] = os.path.join(
            intermediate_dir, f'c_dead_carbon_stocks{file_suffix}.tif')

    if args['compute_forest_edge_effects']:
        output_file_registry['spatial_index_pickle'] = os.path.join(
            intermediate_dir, f'spatial_index{file_suffix}.pickle')
        output_file_registry['edge_distance'] = os.path.join(
            intermediate_dir, f'edge_distance{file_suffix}.tif')
        output_file_registry['tropical_forest_edge_carbon_map'] = os.path.join(
            intermediate_dir,
            f'tropical_forest_edge_carbon_stocks{file_suffix}.tif')
        output_file_registry['non_forest_mask'] = os.path.join(
            intermediate_dir, f'non_forest_mask{file_suffix}.tif')

    # Map non-forest landcover codes to carbon biomasses
    LOGGER.info('Calculating direct mapped carbon stocks')
    carbon_maps = []
    biophysical_table = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'lucode', to_lower=False)
    biophysical_keys = [
        x.lower() for x in list(biophysical_table.values())[0].keys()]
    pool_list = [('c_above', True)]
    if args['pools_to_calculate'] == 'all':
        pool_list.extend([
            ('c_below', False), ('c_soil', False), ('c_dead', False)])
    for carbon_pool_type, ignore_tropical_type in pool_list:
        if carbon_pool_type in biophysical_keys:
            carbon_maps.append(
                output_file_registry[carbon_pool_type+'_map'])
            task_graph.add_task(
                func=_calculate_lulc_carbon_map,
                args=(args['lulc_raster_path'], args['biophysical_table_path'],
                      carbon_pool_type, ignore_tropical_type,
                      args['compute_forest_edge_effects'], carbon_maps[-1]),
                target_path_list=[carbon_maps[-1]],
                task_name=f'calculate_lulc_{carbon_pool_type}_map')

    if args['compute_forest_edge_effects']:
        # generate a map of pixel distance to forest edge from the landcover
        # map
        LOGGER.info('Calculating distance from forest edge')
        map_distance_task = task_graph.add_task(
            func=_map_distance_from_tropical_forest_edge,
            args=(args['lulc_raster_path'], args['biophysical_table_path'],
                  output_file_registry['edge_distance'],
                  output_file_registry['non_forest_mask']),
            target_path_list=[output_file_registry['edge_distance'],
                              output_file_registry['non_forest_mask']],
            task_name='map_distance_from_forest_edge')

        # Build spatial index for gridded global model for closest 3 points
        LOGGER.info('Building spatial index for forest edge models.')
        build_spatial_index_task = task_graph.add_task(
            func=_build_spatial_index,
            args=(args['lulc_raster_path'], intermediate_dir,
                  args['tropical_forest_edge_carbon_model_vector_path'],
                  output_file_registry['spatial_index_pickle']),
            target_path_list=[output_file_registry['spatial_index_pickle']],
            task_name='build_spatial_index')

        # calculate the carbon edge effect on forests
        LOGGER.info('Calculating forest edge carbon')
        task_graph.add_task(
            func=_calculate_tropical_forest_edge_carbon_map,
            args=(output_file_registry['edge_distance'],
                  output_file_registry['spatial_index_pickle'],
                  int(args['n_nearest_model_points']),
                  float(args['biomass_to_carbon_conversion_factor']),
                  output_file_registry['tropical_forest_edge_carbon_map']),
            target_path_list=[
                output_file_registry['tropical_forest_edge_carbon_map']],
            task_name='calculate_forest_edge_carbon_map',
            dependent_task_list=[map_distance_task, build_spatial_index_task])

        # This is also a carbon stock
        carbon_maps.append(
            output_file_registry['tropical_forest_edge_carbon_map'])

    # combine maps into a single output
    LOGGER.info('combining carbon maps into single raster')

    carbon_maps_band_list = [(path, 1) for path in carbon_maps]

    # Join here since the raster calculation depends on the target datasets
    # from all the tasks above
    task_graph.join()

    combine_carbon_maps_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(carbon_maps_band_list, combine_carbon_maps,
              output_file_registry['carbon_map'], gdal.GDT_Float32,
              NODATA_VALUE),
        target_path_list=[output_file_registry['carbon_map']],
        task_name='combine_carbon_maps')

    # generate report (optional) by aoi if they exist
    if 'aoi_vector_path' in args and args['aoi_vector_path'] != '':
        LOGGER.info('aggregating carbon map by aoi')
        task_graph.add_task(
            func=_aggregate_carbon_map,
            args=(args['aoi_vector_path'], output_file_registry['carbon_map'],
                  output_file_registry['aggregated_result_vector']),
            target_path_list=[
                output_file_registry['aggregated_result_vector']],
            task_name='combine_carbon_maps',
            dependent_task_list=[combine_carbon_maps_task])

    # close taskgraph
    task_graph.close()
    task_graph.join()


def combine_carbon_maps(*carbon_maps):
    """Combine the carbon maps and leave nodata where all inputs were nodata.

    Args:
        *carbon_maps (array): arrays of carbon stocks stored in different pool
            types.

    Returns:
        result (array): an array consists of all the carbon stocks from
            different pool types.

    """
    result = numpy.zeros(carbon_maps[0].shape)
    nodata_mask = numpy.empty(carbon_maps[0].shape, dtype=bool)
    nodata_mask[:] = True
    for carbon_map in carbon_maps:
        valid_mask = ~utils.array_equals_nodata(carbon_map, NODATA_VALUE)
        nodata_mask &= ~valid_mask
        result[valid_mask] += carbon_map[valid_mask]
    result[nodata_mask] = NODATA_VALUE
    return result


def _aggregate_carbon_map(
        aoi_vector_path, carbon_map_path, target_aggregate_vector_path):
    """Helper function to aggregate carbon values for the given serviceshed.

    Generates a new shapefile that's a copy of 'aoi_vector_path' in
    'workspace_dir' with mean and sum values from the raster at
    'carbon_map_path'

    Args:
        aoi_vector_path (string): path to shapefile that will be used to
            aggregate raster at'carbon_map_path'.
        workspace_dir (string): path to a directory that function can copy
            the shapefile at aoi_vector_path into.
        carbon_map_path (string): path to raster that will be aggregated by
            the given serviceshed polygons
        target_aggregate_vector_path (string): path to an ESRI shapefile that
            will be created by this function as the aggregating output.

    Returns:
        None

    """
    aoi_vector = gdal.OpenEx(aoi_vector_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('ESRI Shapefile')

    if os.path.exists(target_aggregate_vector_path):
        os.remove(target_aggregate_vector_path)
    driver.CreateCopy(target_aggregate_vector_path, aoi_vector)
    aoi_vector = None

    # aggregate carbon stocks by the FID
    serviceshed_stats = pygeoprocessing.zonal_statistics(
        (carbon_map_path, 1), target_aggregate_vector_path)

    carbon_sum_field = ogr.FieldDefn('c_sum', ogr.OFTReal)
    carbon_sum_field.SetWidth(24)
    carbon_sum_field.SetPrecision(11)
    carbon_mean_field = ogr.FieldDefn('c_ha_mean', ogr.OFTReal)
    carbon_mean_field.SetWidth(24)
    carbon_mean_field.SetPrecision(11)

    target_aggregate_vector = gdal.OpenEx(
        target_aggregate_vector_path, gdal.OF_UPDATE)
    target_aggregate_layer = target_aggregate_vector.GetLayer()
    target_aggregate_layer.CreateField(carbon_sum_field)
    target_aggregate_layer.CreateField(carbon_mean_field)

    target_aggregate_layer.ResetReading()
    target_aggregate_layer.StartTransaction()

    for poly_feat in target_aggregate_layer:
        poly_fid = poly_feat.GetFID()
        poly_feat.SetField(
            'c_sum', serviceshed_stats[poly_fid]['sum'])
        # calculates mean pixel value per ha in for each feature in AOI
        poly_geom = poly_feat.GetGeometryRef()
        poly_area_ha = poly_geom.GetArea() / 1e4  # converts m^2 to hectare
        poly_geom = None
        poly_feat.SetField(
            'c_ha_mean', serviceshed_stats[poly_fid]['sum']/poly_area_ha)

        target_aggregate_layer.SetFeature(poly_feat)
    target_aggregate_layer.CommitTransaction()
    target_aggregate_layer, target_aggregate_vector = None, None


def _calculate_lulc_carbon_map(
        lulc_raster_path, biophysical_table_path, carbon_pool_type,
        ignore_tropical_type, compute_forest_edge_effects, carbon_map_path):
    """Calculates the carbon on the map from non-forest landcover types only.

    Args:
        lulc_raster_path (string): a filepath to the landcover map that
            contains integer landcover codes
        biophysical_table_path (string): a filepath to a csv table that indexes
            landcover codes to surface carbon, contains at least the fields
            'lucode' (landcover integer code), 'is_tropical_forest' (0 or 1
            depending on landcover code type), and 'c_above' (carbon density in
            terms of Mg/Ha)
        carbon_pool_type (string): a carbon mapping field in
            biophysical_table_path.  ex. 'c_above', 'c_below', ...
        ignore_tropical_type (boolean): if true, any landcover type whose
            'is_tropical_forest' field == 1 will be ignored for mapping the
            carbon pool type.
        compute_forest_edge_effects (boolean): if true the 'is_tropical_forest'
            header will be considered, if not, it is ignored
        carbon_map_path (string): a filepath to the output raster
            that will contain total mapped carbon per cell.

    Returns:
        None

    """
    # classify forest pixels from lulc
    biophysical_table = utils.build_lookup_from_csv(
        biophysical_table_path, 'lucode', to_lower=False)

    lucode_to_per_cell_carbon = {}
    cell_size = pygeoprocessing.get_raster_info(
        lulc_raster_path)['pixel_size']  # in meters
    cell_area_ha = abs(cell_size[0]) * abs(cell_size[1]) / 10000

    # Build a lookup table
    for lucode in biophysical_table:
        if compute_forest_edge_effects:
            is_tropical_forest = (
                int(biophysical_table[int(lucode)]['is_tropical_forest']))
        else:
            is_tropical_forest = 0
        if ignore_tropical_type and is_tropical_forest == 1:
            # if tropical forest above ground, lookup table is nodata
            lucode_to_per_cell_carbon[int(lucode)] = NODATA_VALUE
        else:
            try:
                lucode_to_per_cell_carbon[int(lucode)] = float(
                    biophysical_table[lucode][carbon_pool_type]) * cell_area_ha
            except ValueError:
                raise ValueError(
                    "Could not interpret carbon pool value as a number. "
                    f"lucode: {lucode}, pool_type: {carbon_pool_type}, "
                    f"value: {biophysical_table[lucode][carbon_pool_type]}")

    # map aboveground carbon from table to lulc that is not forest
    reclass_error_details = {
        'raster_name': 'LULC',
        'column_name': 'lucode',
        'table_name': 'Biophysical'}

    utils.reclassify_raster(
        (lulc_raster_path, 1), lucode_to_per_cell_carbon,
        carbon_map_path, gdal.GDT_Float32, NODATA_VALUE,
        reclass_error_details)


def _map_distance_from_tropical_forest_edge(
        base_lulc_raster_path, biophysical_table_path, edge_distance_path,
        target_non_forest_mask_path):
    """Generates a raster of forest edge distances.

    Generates a raster of forest edge distances where each pixel is the
    distance to the edge of the forest in meters.

    Args:
        base_lulc_raster_path (string): path to the landcover raster that
            contains integer landcover codes
        biophysical_table_path (string): path to a csv table that indexes
            landcover codes to forest type, contains at least the fields
            'lucode' (landcover integer code) and 'is_tropical_forest' (0 or 1
            depending on landcover code type)
        edge_distance_path (string): path to output raster where each pixel
            contains the euclidean pixel distance to nearest forest edges on
            all non-nodata values of base_lulc_raster_path
        target_non_forest_mask_path (string): path to the output non forest
            mask raster

    Returns:
        None

    """
    # Build a list of forest lucodes
    biophysical_table = utils.build_lookup_from_csv(
        biophysical_table_path, 'lucode', to_lower=False)
    forest_codes = [
        lucode for (lucode, ludata) in biophysical_table.items()
        if int(ludata['is_tropical_forest']) == 1]

    # Make a raster where 1 is non-forest landcover types and 0 is forest
    lulc_nodata = pygeoprocessing.get_raster_info(
        base_lulc_raster_path)['nodata']

    forest_mask_nodata = 255

    def mask_non_forest_op(lulc_array):
        """Convert forest lulc codes to 0.
        Args:
            lulc_array (numpy.ndarray): array representing a LULC raster where
                each forest LULC code is in `forest_codes`.
        Returns:
            numpy.ndarray with the same shape as lulc_array. All pixels are
                0 (forest), 1 (non-forest), or 255 (nodata).
        """
        non_forest_mask = ~numpy.isin(lulc_array, forest_codes)
        nodata_mask = lulc_array == lulc_nodata
        # where LULC has nodata, set value to nodata value (255)
        # where LULC has data, set to 0 if LULC is a forest type, 1 if it's not
        return numpy.where(nodata_mask, forest_mask_nodata, non_forest_mask)

    pygeoprocessing.raster_calculator(
        [(base_lulc_raster_path, 1)], mask_non_forest_op,
        target_non_forest_mask_path, gdal.GDT_Byte, forest_mask_nodata)

    # Do the distance transform on non-forest pixels
    # This is the distance from each pixel to the nearest pixel with value 1.
    #   - for forest pixels, this is the distance to the forest edge
    #   - for non-forest pixels, this is 0
    #   - for nodata pixels, distance is calculated but is meaningless
    pygeoprocessing.distance_transform_edt(
        (target_non_forest_mask_path, 1), edge_distance_path)

    # mask out the meaningless distance pixels so they don't affect the output
    lulc_raster = gdal.OpenEx(base_lulc_raster_path)
    lulc_band = lulc_raster.GetRasterBand(1)
    edge_distance_raster = gdal.OpenEx(edge_distance_path, gdal.GA_Update)
    edge_distance_band = edge_distance_raster.GetRasterBand(1)

    for offset_dict in pygeoprocessing.iterblocks((base_lulc_raster_path, 1), offset_only=True):
        # where LULC has nodata, overwrite edge distance with nodata value
        lulc_block = lulc_band.ReadAsArray(**offset_dict)
        distance_block = edge_distance_band.ReadAsArray(**offset_dict)
        nodata_mask = utils.array_equals_nodata(lulc_block, lulc_nodata)
        distance_block[nodata_mask] = lulc_nodata
        edge_distance_band.WriteArray(
            distance_block,
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])


def _build_spatial_index(
        base_raster_path, local_model_dir,
        tropical_forest_edge_carbon_model_vector_path,
        target_spatial_index_pickle_path):
    """Build a kd-tree index.

    Build a kd-tree index of the locally projected globally georeferenced
    carbon edge model parameters.

    Args:
        base_raster_path (string): path to a raster that is used to define the
            bounding box and projection of the local model.
        local_model_dir (string): path to a directory where we can write a
            shapefile of the locally projected global data model grid.
            Function will create a file called 'local_carbon_shape.shp' in
            that location and overwrite one if it exists.
        tropical_forest_edge_carbon_model_vector_path (string): a path to an
            OGR shapefile that has the parameters for the global carbon edge
            model. Each georeferenced feature should have fields 'theta1',
            'theta2', 'theta3', and 'method'
        spatial_index_pickle_path (string): path to the pickle file to store a
            tuple of:
                scipy.spatial.cKDTree (georeferenced locally projected model
                    points)
                theta_model_parameters (parallel Nx3 array of theta parameters)
                method_model_parameter (parallel N array of model numbers (1..3))

    Returns:
        None

    """
    # Reproject the global model into local coordinate system
    carbon_model_reproject_path = os.path.join(
        local_model_dir, 'local_carbon_shape.shp')
    lulc_projection_wkt = pygeoprocessing.get_raster_info(
        base_raster_path)['projection_wkt']
    pygeoprocessing.reproject_vector(
        tropical_forest_edge_carbon_model_vector_path, lulc_projection_wkt,
        carbon_model_reproject_path)

    model_vector = gdal.OpenEx(carbon_model_reproject_path)
    model_layer = model_vector.GetLayer()

    kd_points = []
    theta_model_parameters = []
    method_model_parameter = []

    # put all the polygons in the kd_tree because it's fast and simple
    for poly_feature in model_layer:
        poly_geom = poly_feature.GetGeometryRef()
        poly_centroid = poly_geom.Centroid()
        # put in row/col order since rasters are row/col indexed
        kd_points.append([poly_centroid.GetY(), poly_centroid.GetX()])

        theta_model_parameters.append([
            poly_feature.GetField(feature_id) for feature_id in
            ['theta1', 'theta2', 'theta3']])
        method_model_parameter.append(poly_feature.GetField('method'))

    method_model_parameter = numpy.array(
        method_model_parameter, dtype=numpy.int32)
    theta_model_parameters = numpy.array(
        theta_model_parameters, dtype=numpy.float32)

    LOGGER.info('Building kd_tree')
    kd_tree = scipy.spatial.cKDTree(kd_points)
    LOGGER.info(f'Done building kd_tree with {len(kd_points)} points')

    with open(target_spatial_index_pickle_path, 'wb') as picklefile:
        picklefile.write(
            pickle.dumps(
                (kd_tree, theta_model_parameters, method_model_parameter)))


def _calculate_tropical_forest_edge_carbon_map(
        edge_distance_path, spatial_index_pickle_path, n_nearest_model_points,
        biomass_to_carbon_conversion_factor,
        tropical_forest_edge_carbon_map_path):
    """Calculates the carbon on the forest pixels accounting for their global
    position with respect to precalculated edge carbon models.

    Args:
        edge_distance_path (string): path to the a raster where each pixel
            contains the pixel distance to forest edge.
        spatial_index_pickle_path (string): path to the pickle file that
            contains a tuple of:
                kd_tree (scipy.spatial.cKDTree): a kd-tree that has indexed the
                    valid model parameter points for fast nearest neighbor
                    calculations.
                theta_model_parameters (numpy.array Nx3): parallel array of
                    model theta parameters consistent with the order in which
                    points were inserted into 'kd_tree'
                method_model_parameter (numpy.array N): parallel array of
                    method numbers (1..3) consistent with the order in which
                    points were inserted into 'kd_tree'.
        n_nearest_model_points (int): number of nearest model points to search
            for.
        biomass_to_carbon_conversion_factor (float): number by which to
            multiply the biomass by to get carbon.
        tropical_forest_edge_carbon_map_path (string): a filepath to the output
            raster which will contain total carbon stocks per cell of forest
            type.

    Returns:
        None

    """
    # load spatial indices from pickle file
    # let d = number of precalculated model cells (2217 for sample data)
    #   kd_tree.data.shape: (d, 2)
    #   theta_model_parameters.shape: (d, 3)
    #   method_model_parameter.shape: (d,)
    kd_tree, theta_model_parameters, method_model_parameter = pickle.load(
        open(spatial_index_pickle_path, 'rb'))

    # create output raster and open band for writing
    # fill nodata, in case we skip entire memory blocks that are non-forest
    pygeoprocessing.new_raster_from_base(
        edge_distance_path, tropical_forest_edge_carbon_map_path,
        gdal.GDT_Float32, band_nodata_list=[NODATA_VALUE],
        fill_value_list=[NODATA_VALUE])
    edge_carbon_raster = gdal.OpenEx(
        tropical_forest_edge_carbon_map_path, gdal.GA_Update)
    edge_carbon_band = edge_carbon_raster.GetRasterBand(1)
    edge_carbon_geotransform = edge_carbon_raster.GetGeoTransform()

    # create edge distance band for memory block reading
    n_rows = edge_carbon_raster.RasterYSize
    n_cols = edge_carbon_raster.RasterXSize
    n_cells = n_rows * n_cols
    n_cells_processed = 0
    # timer to give updates per call
    last_time = time.time()

    cell_xsize, cell_ysize = pygeoprocessing.get_raster_info(
        edge_distance_path)['pixel_size']
    cell_size_km = (abs(cell_xsize) + abs(cell_ysize))/2 / 1000
    cell_area_ha = (abs(cell_xsize) * abs(cell_ysize)) / 10000

    # Loop memory block by memory block, calculating the forest edge carbon
    # for every forest pixel.
    for edge_distance_data, edge_distance_block in pygeoprocessing.iterblocks(
            (edge_distance_path, 1), largest_block=2**12):
        current_time = time.time()
        if current_time - last_time > 5:
            LOGGER.info('Carbon edge calculation approx. '
                        f'{n_cells_processed / n_cells * 100:.2f} complete')
            last_time = current_time
        n_cells_processed += (
            edge_distance_data['win_xsize'] * edge_distance_data['win_ysize'])
        # only forest pixels will have an edge distance > 0
        valid_edge_distance_mask = (edge_distance_block > 0)

        # if no valid forest pixels to calculate, skip to the next block
        if not valid_edge_distance_mask.any():
            continue

        # calculate local coordinates for each pixel so we can test for
        # distance to the nearest carbon model points
        col_range = numpy.linspace(
            edge_carbon_geotransform[0] +
            edge_carbon_geotransform[1] * edge_distance_data['xoff'],
            edge_carbon_geotransform[0] +
            edge_carbon_geotransform[1] * (
                edge_distance_data['xoff'] + edge_distance_data['win_xsize']),
            num=edge_distance_data['win_xsize'], endpoint=False)
        row_range = numpy.linspace(
            edge_carbon_geotransform[3] +
            edge_carbon_geotransform[5] * edge_distance_data['yoff'],
            edge_carbon_geotransform[3] +
            edge_carbon_geotransform[5] * (
                edge_distance_data['yoff'] + edge_distance_data['win_ysize']),
            num=edge_distance_data['win_ysize'], endpoint=False)
        col_coords, row_coords = numpy.meshgrid(col_range, row_range)

        # query nearest points for every point in the grid
        # workers=-1 means use all available CPUs
        coord_points = list(zip(
            row_coords[valid_edge_distance_mask].ravel(),
            col_coords[valid_edge_distance_mask].ravel()))
        # for each forest point x, for each of its k nearest neighbors
        # shape of distances and indexes: (x, k)
        distances, indexes = kd_tree.query(
            coord_points, k=n_nearest_model_points,
            distance_upper_bound=DISTANCE_UPPER_BOUND, workers=-1)

        if n_nearest_model_points == 1:
            distances = distances.reshape(distances.shape[0], 1)
            indexes = indexes.reshape(indexes.shape[0], 1)

        # 3 is for the 3 thetas in the carbon model. thetas shape: (x, k, 3)
        thetas = numpy.zeros((indexes.shape[0], indexes.shape[1], 3))
        valid_index_mask = (indexes != kd_tree.n)
        thetas[valid_index_mask] = theta_model_parameters[
            indexes[valid_index_mask]]

        # reshape to an N,nearest_points so we can multiply by thetas
        valid_edge_distances_km = numpy.repeat(
            edge_distance_block[valid_edge_distance_mask] * cell_size_km,
            n_nearest_model_points).reshape(-1, n_nearest_model_points)

        # For each forest pixel x, for each of its k nearest neighbors, the
        # chosen regression method (1, 2, or 3). model_index shape: (x, k)
        model_index = numpy.zeros(indexes.shape, dtype=numpy.int8)
        model_index[valid_index_mask] = (
            method_model_parameter[indexes[valid_index_mask]])

        # biomass shape: (x, k)
        biomass = numpy.zeros((indexes.shape[0], indexes.shape[1]),
                              dtype=numpy.float32)

        # mask shapes: (x, k)
        mask_1 = model_index == 1
        mask_2 = model_index == 2
        mask_3 = model_index == 3

        # exponential model
        # biomass_1 = t1 - t2 * exp(-t3 * edge_dist_km)
        biomass[mask_1] = (
            thetas[mask_1][:, 0] - thetas[mask_1][:, 1] * numpy.exp(
                -thetas[mask_1][:, 2] * valid_edge_distances_km[mask_1])
        ) * cell_area_ha

        # logarithmic model
        # biomass_2 = t1 + t2 * numpy.log(edge_dist_km)
        biomass[mask_2] = (
            thetas[mask_2][:, 0] + thetas[mask_2][:, 1] * numpy.log(
                valid_edge_distances_km[mask_2])) * cell_area_ha

        # linear regression
        # biomass_3 = t1 + t2 * edge_dist_km
        biomass[mask_3] = (
            thetas[mask_3][:, 0] + thetas[mask_3][:, 1] *
            valid_edge_distances_km[mask_3]) * cell_area_ha

        # reshape the array so that each set of points is in a separate
        # dimension, here distances are distances to each valid model
        # point, not distance to edge of forest
        weights = numpy.zeros(distances.shape)
        valid_distance_mask = (distances > 0) & (distances < numpy.inf)
        weights[valid_distance_mask] = (
            n_nearest_model_points / distances[valid_distance_mask])

        # Denominator is the sum of the weights per nearest point (axis 1)
        denom = numpy.sum(weights, axis=1)
        # To avoid a divide by 0
        valid_denom = denom != 0
        average_biomass = numpy.zeros(distances.shape[0])
        average_biomass[valid_denom] = (
            numpy.sum(weights[valid_denom] *
                      biomass[valid_denom], axis=1) / denom[valid_denom])

        # Ensure the result has nodata everywhere the distance was invalid
        result = numpy.full(edge_distance_block.shape, NODATA_VALUE,
                            dtype=numpy.float32)
        # convert biomass to carbon in this stage
        result[valid_edge_distance_mask] = (
            average_biomass * biomass_to_carbon_conversion_factor)
        edge_carbon_band.WriteArray(
            result, xoff=edge_distance_data['xoff'],
            yoff=edge_distance_data['yoff'])
    LOGGER.info('Carbon edge calculation 100.0% complete')


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
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
    validation_warnings = validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])

    invalid_keys = set([])
    for affected_keys, error_msg in validation_warnings:
        for key in affected_keys:
            invalid_keys.add(key)

    if ('pools_to_calculate' not in invalid_keys and
            'biophysical_table_path' not in invalid_keys):
        if args['pools_to_calculate'] == 'all':
            # other fields have already been checked by validate
            required_fields = ['c_above', 'c_below', 'c_soil', 'c_dead']
            error_msg = validation.check_csv(
                args['biophysical_table_path'],
                header_patterns=required_fields,
                axis=1)
            if error_msg:
                validation_warnings.append(
                    (['biophysical_table_path'], error_msg))

    return validation_warnings
