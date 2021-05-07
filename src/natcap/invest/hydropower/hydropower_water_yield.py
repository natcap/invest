"""InVEST Hydropower Water Yield model."""
import logging
import os
import math
import pickle

import numpy
from osgeo import gdal
from osgeo import ogr
import pygeoprocessing
import taskgraph

from .. import validation
from .. import utils

LOGGER = logging.getLogger(__name__)

ARGS_SPEC = {
    "model_name": "Hydropower Water Yield",
    "module": __name__,
    "userguide_html": "reservoirhydropowerproduction.html",
    "args_with_spatial_overlap": {
        "spatial_keys": ["depth_to_root_rest_layer_path",
                         "precipitation_path",
                         "pawc_path",
                         "eto_path",
                         "watersheds_path",
                         "sub_watersheds_path"],
        "different_projections_ok": False,
    },
    "args": {
        "workspace_dir": validation.WORKSPACE_SPEC,
        "results_suffix": validation.SUFFIX_SPEC,
        "n_workers": validation.N_WORKERS_SPEC,
        "lulc_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "A GDAL-supported raster file containing LULC code "
                "(expressed as integers) for each cell."),
            "name": "Land Use"
        },
        "depth_to_root_rest_layer_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "A GDAL-supported raster file containing an average root "
                "restricting layer depth value for each cell. The root "
                "restricting layer depth value should be in "
                "millimeters (mm)."),
            "name": "Depth To Root Restricting Layer"
        },
        "precipitation_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "A GDAL-supported raster file containing non-zero, average "
                "annual precipitation values for each cell. The "
                "precipitation values should be in millimeters "
                "(mm)."),
            "name": "Precipitation"
        },
        "pawc_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "A GDAL-supported raster file containing plant available "
                "water content values for each cell.  The plant available "
                "water content fraction should be a value between 0 "
                "and 1."),
            "name": "Plant Available Water Fraction"
        },
        "eto_path": {
            "type": "raster",
            "required": True,
            "validation_options": {
                "projected": True,
            },
            "about": (
                "A GDAL-supported raster file containing annual average "
                "reference evapotranspiration values for each cell.  The "
                "reference evapotranspiration values should be in "
                "millimeters (mm)."),
            "name": "Reference Evapotranspiration"
        },
        "watersheds_path": {
            "validation_options": {
                "required_fields": ['ws_id'],
                "projected": True,
            },
            "type": "vector",
            "required": True,
            "about": (
                "A GDAL-supported vector file containing one polygon per "
                "watershed.  Each polygon that represents a watershed is "
                "required to have a field 'ws_id' that is a unique integer "
                "which identifies that watershed."),
            "name": "Watersheds"
        },
        "sub_watersheds_path": {
            "validation_options": {
                "required_fields": ["subws_id"],
                "projected": True
            },
            "type": "vector",
            "required": False,
            "about": (
                "A GDAL-supported vector file with one polygon per "
                "sub-watershed within the main watersheds specified in the "
                "Watersheds shapefile.  Each polygon that represents a "
                "sub-watershed is required to have a field 'subws_id' that "
                "is a unique integer which identifies that sub-watershed."),
            "name": "Sub-Watersheds"
        },
        "biophysical_table_path": {
            "validation_options": {
                "required_fields": ["lucode", "root_depth", "Kc"],
            },
            "type": "csv",
            "required": True,
            "about": (
                "A CSV table of land use/land cover (LULC) classes, "
                "containing data on biophysical coefficients used in this "
                "model.  The following columns are required: "
                "'lucode' (integer), 'root_depth' (mm), 'Kc' (coefficient)."),
            "name": "Biophysical Table"
        },
        "seasonality_constant": {
            "validation_options": {
                "expression": "value > 0"
            },
            "type": "number",
            "required": True,
            "about": (
                "Floating point value on the order of 1 to 30 "
                "corresponding to the seasonal distribution of "
                "precipitation."),
            "name": "Z parameter"
        },
        "demand_table_path": {
            "validation_options": {
                "required_fields": ["lucode", "demand"],
            },
            "type": "csv",
            "required": False,
            "about": (
                "A CSV table of LULC classes, showing consumptive water use "
                "for each land-use/land-cover type.  The table requires two "
                "column fields: 'lucode' and 'demand'. The demand values "
                "should be the estimated average consumptive water use for "
                "each land-use/land- cover type.  Water use should be given "
                "in cubic meters per year for a pixel in the "
                "land-use/land-cover map.  NOTE: the accounting for pixel "
                "area is important since larger areas will consume more "
                "water for the same land-cover type."),
            "name": "Water Demand Table"
        },
        "valuation_table_path": {
            "validation_options": {
                "required_fields": ["ws_id", "efficiency", "fraction",
                                    "height", "kw_price", "cost", "time_span",
                                    "discount"],
            },
            "type": "csv",
            "required": False,
            "about": (
                "A CSV table of hydropower stations with associated model "
                "values.  The table should have the following column "
                "fields: 'ws_id', 'efficiency', 'fraction', 'height', "
                "'kw_price', 'cost', 'time_span', and 'discount'."),
            "name": "Hydropower Valuation Table"
        }
    }
}


def execute(args):
    """Annual Water Yield: Reservoir Hydropower Production.

    Executes the hydropower/water_yield model

    Args:
        args['workspace_dir'] (string): a path to the directory that will write
            output and other temporary files during calculation. (required)

        args['lulc_path'] (string): a path to a land use/land cover raster
            whose LULC indexes correspond to indexes in the biophysical table
            input. Used for determining soil retention and other biophysical
            properties of the landscape. (required)

        args['depth_to_root_rest_layer_path'] (string): a path to an input
            raster describing the depth of "good" soil before reaching this
            restrictive layer (required)

        args['precipitation_path'] (string): a path to an input raster
            describing the average annual precipitation value for each cell
            (mm) (required)

        args['pawc_path'] (string): a path to an input raster describing the
            plant available water content value for each cell. Plant Available
            Water Content fraction (PAWC) is the fraction of water that can be
            stored in the soil profile that is available for plants' use.
            PAWC is a fraction from 0 to 1 (required)

        args['eto_path'] (string): a path to an input raster describing the
            annual average evapotranspiration value for each cell. Potential
            evapotranspiration is the potential loss of water from soil by
            both evaporation from the soil and transpiration by healthy
            Alfalfa (or grass) if sufficient water is available (mm)
            (required)

        args['watersheds_path'] (string): a path to an input shapefile of the
            watersheds of interest as polygons. (required)

        args['sub_watersheds_path'] (string): a path to an input shapefile of
            the subwatersheds of interest that are contained in the
            ``args['watersheds_path']`` shape provided as input. (optional)

        args['biophysical_table_path'] (string): a path to an input CSV table
            of land use/land cover classes, containing data on biophysical
            coefficients such as root_depth (mm) and Kc, which are required.
            A column with header LULC_veg is also required which should
            have values of 1 or 0, 1 indicating a land cover type of
            vegetation, a 0 indicating non vegetation or wetland, water.
            NOTE: these data are attributes of each LULC class rather than
            attributes of individual cells in the raster map (required)

        args['seasonality_constant'] (float): floating point value between
            1 and 30 corresponding to the seasonal distribution of
            precipitation (required)

        args['results_suffix'] (string): a string that will be concatenated
            onto the end of file names (optional)

        args['demand_table_path'] (string): (optional) if a non-empty string,
            a path to an input CSV
            table of LULC classes, showing consumptive water use for each
            landuse / land-cover type (cubic meters per year) to calculate
            water scarcity.

        args['valuation_table_path'] (string): (optional) if a non-empty
            string, a path to an input CSV table of
            hydropower stations with the following fields to calculate
            valuation: 'ws_id', 'time_span', 'discount', 'efficiency', 
            'fraction', 'cost', 'height', 'kw_price'
            Required if ``calculate_valuation`` is True.

        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None

    """
    LOGGER.info('Validating arguments')
    invalid_parameters = validate(args)
    if invalid_parameters:
        raise ValueError("Invalid parameters passed: %s" % invalid_parameters)

    # valuation_params is passed to create_vector_output()
    # which computes valuation if valuation_params is not None.
    valuation_params = None
    if 'valuation_table_path' in args and args['valuation_table_path'] != '':
        LOGGER.info(
            'Checking that watersheds have entries for every `ws_id` in the '
            'valuation table.')
        # Open/read in valuation parameters from CSV file
        valuation_params = utils.build_lookup_from_csv(
            args['valuation_table_path'], 'ws_id')
        watershed_vector = gdal.OpenEx(
            args['watersheds_path'], gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()
        missing_ws_ids = []
        for watershed_feature in watershed_layer:
            watershed_ws_id = watershed_feature.GetField('ws_id')
            if watershed_ws_id not in valuation_params:
                missing_ws_ids.append(watershed_ws_id)
        watershed_feature = None
        watershed_layer = None
        watershed_vector = None
        if missing_ws_ids:
            raise ValueError(
                'The following `ws_id`s exist in the watershed vector file '
                'but are not found in the valuation table. Check your '
                'valuation table to see if they are missing: "%s"' % (
                    ', '.join(str(x) for x in sorted(missing_ws_ids))))

    # Construct folder paths
    workspace_dir = args['workspace_dir']
    output_dir = os.path.join(workspace_dir, 'output')
    per_pixel_output_dir = os.path.join(output_dir, 'per_pixel')
    intermediate_dir = os.path.join(workspace_dir, 'intermediate')
    pickle_dir = os.path.join(intermediate_dir, '_tmp_zonal_stats')
    utils.make_directories(
        [workspace_dir, output_dir, per_pixel_output_dir,
         intermediate_dir, pickle_dir])

    # Append a _ to the suffix if it's not empty and doesn't already have one
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    # Paths for targets of align_and_resize_raster_stack
    clipped_lulc_path = os.path.join(
        intermediate_dir, 'clipped_lulc%s.tif' % file_suffix)
    eto_path = os.path.join(intermediate_dir, 'eto%s.tif' % file_suffix)
    precip_path = os.path.join(intermediate_dir, 'precip%s.tif' % file_suffix)
    depth_to_root_rest_layer_path = os.path.join(
        intermediate_dir, 'depth_to_root_rest_layer%s.tif' % file_suffix)
    pawc_path = os.path.join(intermediate_dir, 'pawc%s.tif' % file_suffix)
    tmp_pet_path = os.path.join(intermediate_dir, 'pet%s.tif' % file_suffix)

    # Paths for output rasters
    fractp_path = os.path.join(
        per_pixel_output_dir, 'fractp%s.tif' % file_suffix)
    wyield_path = os.path.join(
        per_pixel_output_dir, 'wyield%s.tif' % file_suffix)
    aet_path = os.path.join(per_pixel_output_dir, 'aet%s.tif' % file_suffix)

    demand_path = os.path.join(intermediate_dir, 'demand%s.tif' % file_suffix)

    watersheds_path = args['watersheds_path']
    watershed_results_vector_path = os.path.join(
        output_dir, 'watershed_results_wyield%s.shp' % file_suffix)
    watershed_paths_list = [
        (watersheds_path, 'ws_id', watershed_results_vector_path)]

    sub_watersheds_path = None
    if 'sub_watersheds_path' in args and args['sub_watersheds_path'] != '':
        sub_watersheds_path = args['sub_watersheds_path']
        subwatershed_results_vector_path = os.path.join(
            output_dir, 'subwatershed_results_wyield%s.shp' % file_suffix)
        watershed_paths_list.append(
            (sub_watersheds_path, 'subws_id',
             subwatershed_results_vector_path))

    seasonality_constant = float(args['seasonality_constant'])

    # Initialize a TaskGraph
    work_token_dir = os.path.join(intermediate_dir, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # single process mode.
    graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    base_raster_path_list = [
        args['eto_path'],
        args['precipitation_path'],
        args['depth_to_root_rest_layer_path'],
        args['pawc_path'],
        args['lulc_path']]

    aligned_raster_path_list = [
        eto_path,
        precip_path,
        depth_to_root_rest_layer_path,
        pawc_path,
        clipped_lulc_path]

    target_pixel_size = pygeoprocessing.get_raster_info(
        args['lulc_path'])['pixel_size']
    align_raster_stack_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        args=(base_raster_path_list, aligned_raster_path_list,
              ['near'] * len(base_raster_path_list),
              target_pixel_size, 'intersection'),
        kwargs={'raster_align_index': 4,
                'base_vector_path_list': [watersheds_path]},
        target_path_list=aligned_raster_path_list,
        task_name='align_raster_stack')
    # Joining now since this task will always be the root node
    # and it's useful to have the raster info available.
    align_raster_stack_task.join()

    nodata_dict = {
        'out_nodata': -1.0,
        'precip': pygeoprocessing.get_raster_info(precip_path)['nodata'][0],
        'eto': pygeoprocessing.get_raster_info(eto_path)['nodata'][0],
        'depth_root': pygeoprocessing.get_raster_info(
            depth_to_root_rest_layer_path)['nodata'][0],
        'pawc': pygeoprocessing.get_raster_info(pawc_path)['nodata'][0],
        'lulc': pygeoprocessing.get_raster_info(clipped_lulc_path)['nodata'][0]}

    # Open/read in the csv file into a dictionary and add to arguments
    bio_dict = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'lucode', to_lower=True)
    bio_lucodes = set(bio_dict.keys())
    bio_lucodes.add(nodata_dict['lulc'])
    LOGGER.debug('bio_lucodes %s', bio_lucodes)

    if 'demand_table_path' in args and args['demand_table_path'] != '':
        demand_dict = utils.build_lookup_from_csv(
            args['demand_table_path'], 'lucode')
        demand_reclassify_dict = dict(
            [(lucode, demand_dict[lucode]['demand'])
             for lucode in demand_dict])
        demand_lucodes = set(demand_dict.keys())
        demand_lucodes.add(nodata_dict['lulc'])
        LOGGER.debug('demand_lucodes %s', demand_lucodes)
    else:
        demand_lucodes = None

    # Break the bio_dict into three separate dictionaries based on
    # Kc, root_depth, and LULC_veg fields to use for reclassifying
    Kc_dict = {}
    root_dict = {}
    vegetated_dict = {}

    for lulc_code in bio_dict:
        Kc_dict[lulc_code] = bio_dict[lulc_code]['kc']

        # Catch invalid LULC_veg values with an informative error.
        lulc_veg_value = bio_dict[lulc_code]['lulc_veg']
        try:
            vegetated_dict[lulc_code] = int(lulc_veg_value)
            if vegetated_dict[lulc_code] not in set([0, 1]):
                raise ValueError()
        except ValueError:
            # If the user provided an invalid LULC_veg value, raise an
            # informative error.
            raise ValueError('LULC_veg value must be either 1 or 0, not %s',
                             lulc_veg_value)

        # If LULC_veg value is 1 get root depth value
        if vegetated_dict[lulc_code] == 1.0:
            root_dict[lulc_code] = bio_dict[lulc_code]['root_depth']
        # If LULC_veg value is 0 then we do not care about root
        # depth value so will just substitute in a 1.0 . This
        # value will not end up being used.
        else:
            root_dict[lulc_code] = 1.0

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}
    # Create Kc raster from table values to use in future calculations
    LOGGER.info("Reclassifying temp_Kc raster")
    tmp_Kc_raster_path = os.path.join(intermediate_dir, 'kc_raster.tif')
    create_Kc_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((clipped_lulc_path, 1), Kc_dict, tmp_Kc_raster_path,
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[tmp_Kc_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_Kc_raster')

    # Create root raster from table values to use in future calculations
    LOGGER.info("Reclassifying tmp_root raster")
    tmp_root_raster_path = os.path.join(
        intermediate_dir, 'root_depth.tif')
    create_root_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((clipped_lulc_path, 1), root_dict, tmp_root_raster_path,
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[tmp_root_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_root_raster')

    # Create veg raster from table values to use in future calculations
    # of determining which AET equation to use
    LOGGER.info("Reclassifying tmp_veg raster")
    tmp_veg_raster_path = os.path.join(intermediate_dir, 'veg.tif')
    create_veg_raster_task = graph.add_task(
        func=utils.reclassify_raster,
        args=((clipped_lulc_path, 1), vegetated_dict, tmp_veg_raster_path,
              gdal.GDT_Float32, nodata_dict['out_nodata'],
              reclass_error_details),
        target_path_list=[tmp_veg_raster_path],
        dependent_task_list=[align_raster_stack_task],
        task_name='create_veg_raster')

    dependent_tasks_for_watersheds_list = []

    LOGGER.info('Calculate PET from Ref Evap times Kc')
    calculate_pet_task = graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(eto_path, 1), (tmp_Kc_raster_path, 1),
               (nodata_dict['eto'], 'raw'),
               (nodata_dict['out_nodata'], 'raw')],
              pet_op, tmp_pet_path, gdal.GDT_Float32,
              nodata_dict['out_nodata']),
        target_path_list=[tmp_pet_path],
        dependent_task_list=[create_Kc_raster_task],
        task_name='calculate_pet')
    dependent_tasks_for_watersheds_list.append(calculate_pet_task)

    # List of rasters to pass into the vectorized fractp operation
    raster_list = [
        tmp_Kc_raster_path, eto_path, precip_path, tmp_root_raster_path,
        depth_to_root_rest_layer_path, pawc_path, tmp_veg_raster_path]

    LOGGER.debug('Performing fractp operation')
    calculate_fractp_task = graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(x, 1) for x in raster_list]
              + [(nodata_dict, 'raw'), (seasonality_constant, 'raw')],
              fractp_op, fractp_path, gdal.GDT_Float32,
              nodata_dict['out_nodata']),
        target_path_list=[fractp_path],
        dependent_task_list=[
            create_Kc_raster_task, create_veg_raster_task,
            create_root_raster_task, align_raster_stack_task],
        task_name='calculate_fractp')

    LOGGER.info('Performing wyield operation')
    calculate_wyield_task = graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(fractp_path, 1), (precip_path, 1),
               (nodata_dict['precip'], 'raw'),
               (nodata_dict['out_nodata'], 'raw')],
              wyield_op, wyield_path, gdal.GDT_Float32,
              nodata_dict['out_nodata']),
        target_path_list=[wyield_path],
        dependent_task_list=[calculate_fractp_task, align_raster_stack_task],
        task_name='calculate_wyield')
    dependent_tasks_for_watersheds_list.append(calculate_wyield_task)

    LOGGER.debug('Performing aet operation')
    calculate_aet_task = graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(fractp_path, 1), (precip_path, 1),
               (nodata_dict['precip'], 'raw'),
               (nodata_dict['out_nodata'], 'raw')],
              aet_op, aet_path, gdal.GDT_Float32, nodata_dict['out_nodata']),
        target_path_list=[aet_path],
        dependent_task_list=[
            calculate_fractp_task, create_veg_raster_task,
            align_raster_stack_task],
        task_name='calculate_aet')
    dependent_tasks_for_watersheds_list.append(calculate_aet_task)

    # list of rasters that will always be summarized with zonal stats
    raster_names_paths_list = [
        ('precip_mn', precip_path),
        ('PET_mn', tmp_pet_path),
        ('AET_mn', aet_path),
        ('wyield_mn', wyield_path)]

    if 'demand_table_path' in args and args['demand_table_path'] != '':
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Demand'}
        # Create demand raster from table values to use in future calculations
        create_demand_raster_task = graph.add_task(
            func=utils.reclassify_raster,
            args=((clipped_lulc_path, 1), demand_reclassify_dict, demand_path,
                  gdal.GDT_Float32, nodata_dict['out_nodata'],
                  reclass_error_details),
            target_path_list=[demand_path],
            dependent_task_list=[align_raster_stack_task],
            task_name='create_demand_raster')
        dependent_tasks_for_watersheds_list.append(create_demand_raster_task)
        raster_names_paths_list.append(('demand', demand_path))

    # Aggregate results to watershed polygons, and do the optional
    # scarcity and valuation calculations.
    for base_ws_path, ws_id_name, target_ws_path in watershed_paths_list:

        zonal_stats_task_list = []
        zonal_stats_pickle_list = []

        # Do zonal stats with the input shapefiles provided by the user
        # and store results dictionaries in pickles
        for key_name, rast_path in raster_names_paths_list:
            target_stats_pickle = os.path.join(
                pickle_dir,
                '%s_%s%s.pickle' % (ws_id_name, key_name, file_suffix))
            zonal_stats_pickle_list.append((target_stats_pickle, key_name))
            zonal_stats_task_list.append(graph.add_task(
                func=zonal_stats_tofile,
                args=(base_ws_path, rast_path, target_stats_pickle),
                target_path_list=[target_stats_pickle],
                dependent_task_list=dependent_tasks_for_watersheds_list,
                task_name='%s_%s_zonalstats' % (ws_id_name, key_name)))

        # Create copies of the input shapefiles in the output workspace.
        # Add the zonal stats data to the attribute tables.
        # Compute optional scarcity and valuation
        create_output_vector_task = graph.add_task(
            func=create_vector_output,
            args=(base_ws_path, target_ws_path, ws_id_name,
                  zonal_stats_pickle_list, valuation_params),
            target_path_list=[target_ws_path],
            dependent_task_list=zonal_stats_task_list,
            task_name='create_%s_vector_output' % ws_id_name)

        # Export a CSV with all the fields present in the output vector
        target_basename = os.path.splitext(target_ws_path)[0]
        target_csv_path = target_basename + '.csv'
        create_output_table_task = graph.add_task(
            func=convert_vector_to_csv,
            args=(target_ws_path, target_csv_path),
            target_path_list=[target_csv_path],
            dependent_task_list=[create_output_vector_task],
            task_name='create_%s_table_output' % ws_id_name)

    graph.join()


def create_vector_output(
        base_vector_path, target_vector_path, ws_id_name,
        stats_path_list, valuation_params):
    """Create the main vector outputs of this model.

    Join results of zonal stats to copies of the watershed shapefiles.
    Also do optional scarcity and valuation calculations.

    Args:
        base_vector_path (string): Path to a watershed shapefile provided in
            the args dictionary.
        target_vector_path (string): Path where base_vector_path will be copied
            to in the output workspace.
        ws_id_name (string): Either 'ws_id' or 'subws_id', which are required
            names of a unique ID field in the watershed and subwatershed
            shapefiles, respectively. Used to determine if the polygons
            represent watersheds or subwatersheds.
        stats_path_list (list): List of file paths to pickles storing the zonal
            stats results.
        valuation_params (dict): The dictionary built from
            args['valuation_table_path']. Or None if valuation table was not
            provided.

    Returns:
        None

    """
    esri_shapefile_driver = gdal.GetDriverByName('ESRI Shapefile')
    watershed_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    esri_shapefile_driver.CreateCopy(target_vector_path, watershed_vector)
    watershed_vector = None

    for pickle_path, key_name in stats_path_list:
        with open(pickle_path, 'rb') as picklefile:
            ws_stats_dict = pickle.load(picklefile)

            if key_name == 'wyield_mn':
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, key_name, 'mean')
                # Also create and populate 'wyield_vol' field, which
                # relies on 'wyield_mn' already present in attribute table
                compute_water_yield_volume(target_vector_path)

            # consum_* variables rely on 'wyield_*' fields present,
            # so this would fail if somehow 'demand' comes before 'wyield_mn'
            # in key_names. The order is hardcoded in raster_names_paths_list.
            elif key_name == 'demand':
                # Add aggregated consumption to sheds shapefiles
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, 'consum_vol', 'sum')

                # Add aggregated consumption means to sheds shapefiles
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, 'consum_mn', 'mean')
                compute_rsupply_volume(target_vector_path)

            else:
                _add_zonal_stats_dict_to_shape(
                    target_vector_path, ws_stats_dict, key_name, 'mean')

    if valuation_params:
        # only do valuation for watersheds, not subwatersheds
        if ws_id_name == 'ws_id':
            compute_watershed_valuation(target_vector_path, valuation_params)


def convert_vector_to_csv(base_vector_path, target_csv_path):
    """Create a CSV with all the fields present in vector attribute table.

    Args:
        base_vector_path (string):
            Path to the watershed shapefile in the output workspace.
        target_csv_path (string):
            Path to a CSV to create in the output workspace.

    Returns:
        None

    """
    watershed_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    csv_driver = gdal.GetDriverByName('CSV')
    _ = csv_driver.CreateCopy(target_csv_path, watershed_vector)


def zonal_stats_tofile(base_vector_path, raster_path, target_stats_pickle):
    """Calculate zonal statistics for watersheds and write results to a file.

    Args:
        base_vector_path (string): Path to the watershed shapefile in the
            output workspace.
        raster_path (string): Path to raster to aggregate.
        target_stats_pickle (string): Path to pickle file to store dictionary
            returned by zonal stats.

    Returns:
        None

    """
    ws_stats_dict = pygeoprocessing.zonal_statistics(
        (raster_path, 1), base_vector_path, ignore_nodata=True)
    with open(target_stats_pickle, 'wb') as picklefile:
        picklefile.write(pickle.dumps(ws_stats_dict))


def aet_op(fractp, precip, precip_nodata, output_nodata):
    """Compute actual evapotranspiration values.

    Args:
        fractp (numpy.ndarray float): fractp raster values.
        precip (numpy.ndarray): precipitation raster values (mm).
        precip_nodata (float): nodata value from the precip raster.
        output_nodata (float): nodata value assigned to output of
            raster_calculator.

    Returns:
        numpy.ndarray of actual evapotranspiration values (mm).

    """
    result = numpy.empty_like(fractp)
    result[:] = output_nodata
    # checking if fractp >= 0 because it's a value that's between 0 and 1
    # and the nodata value is a large negative number.
    valid_mask = fractp >= 0
    if precip_nodata is not None:
        valid_mask &= ~numpy.isclose(precip, precip_nodata)
    result[valid_mask] = fractp[valid_mask] * precip[valid_mask]
    return result


def wyield_op(fractp, precip, precip_nodata, output_nodata):
    """Calculate water yield.

    Args:
        fractp (numpy.ndarray float): fractp raster values.
        precip (numpy.ndarray): precipitation raster values (mm).
        precip_nodata (float): nodata value from the precip raster.
        output_nodata (float): nodata value assigned to output of
            raster_calculator.

    Returns:
        numpy.ndarray of water yield value (mm).

    """
    result = numpy.empty_like(fractp)
    result[:] = output_nodata
    # output_nodata is defined above, should never be None
    valid_mask = ~numpy.isclose(fractp, output_nodata)
    if precip_nodata is not None:
        valid_mask &= ~numpy.isclose(precip, precip_nodata)
    result[valid_mask] = (1.0 - fractp[valid_mask]) * precip[valid_mask]
    return result


def fractp_op(
        Kc, eto, precip, root, soil, pawc, veg,
        nodata_dict, seasonality_constant):
    """Calculate actual evapotranspiration fraction of precipitation.

    Args:
        Kc (numpy.ndarray): Kc (plant evapotranspiration
          coefficient) raster values
        eto (numpy.ndarray): potential evapotranspiration raster
          values (mm)
        precip (numpy.ndarray): precipitation raster values (mm)
        root (numpy.ndarray): root depth (maximum root depth for
           vegetated land use classes) raster values (mm)
        soil (numpy.ndarray): depth to root restricted layer raster
            values (mm)
        pawc (numpy.ndarray): plant available water content raster
           values
        veg (numpy.ndarray): 1 or 0 where 1 depicts the land type as
            vegetation and 0 depicts the land type as non
            vegetation (wetlands, urban, water, etc...). If 1 use
            regular AET equation if 0 use: AET = Kc * ETo
        nodata_dict (dict): stores nodata values keyed by raster names
        seasonality_constant (float): floating point value between
            1 and 30 corresponding to the seasonal distribution of
            precipitation.

    Returns:
        numpy.ndarray (float) of actual evapotranspiration as fraction
            of precipitation.

    """
    # Kc, root, & veg were created by reclassify_raster, which set nodata
    # to out_nodata. All others are products of align_and_resize_raster_stack
    # and retain their original nodata values.
    # out_nodata is defined above and should never be None.
    valid_mask = (
        ~numpy.isclose(Kc, nodata_dict['out_nodata']) &
        ~numpy.isclose(root, nodata_dict['out_nodata']) &
        ~numpy.isclose(veg, nodata_dict['out_nodata']) &
        ~numpy.isclose(precip, 0.0))
    if nodata_dict['eto'] is not None:
        valid_mask &= ~numpy.isclose(eto, nodata_dict['eto'])
    if nodata_dict['precip'] is not None:
        valid_mask &= ~numpy.isclose(precip, nodata_dict['precip'])
    if nodata_dict['depth_root'] is not None:
        valid_mask &= ~numpy.isclose(soil, nodata_dict['depth_root'])
    if nodata_dict['pawc'] is not None:
        valid_mask &= ~numpy.isclose(pawc, nodata_dict['pawc'])

    # Compute Budyko Dryness index
    # Use the original AET equation if the land cover type is vegetation
    # If not vegetation (wetlands, urban, water, etc...) use
    # Alternative equation Kc * Eto
    phi = (Kc[valid_mask] * eto[valid_mask]) / precip[valid_mask]
    pet = Kc[valid_mask] * eto[valid_mask]

    # Calculate plant available water content (mm) using the minimum
    # of soil depth and root depth
    awc = numpy.where(
        root[valid_mask] < soil[valid_mask], root[valid_mask],
        soil[valid_mask]) * pawc[valid_mask]
    climate_w = (
        (awc / precip[valid_mask]) * seasonality_constant) + 1.25
    # Capping to 5.0 to set to upper limit if exceeded
    climate_w[climate_w > 5.0] = 5.0

    # Compute evapotranspiration partition of the water balance
    aet_p = (
        1.0 + (pet / precip[valid_mask])) - (
            (1.0 + (pet / precip[valid_mask]) ** climate_w) ** (
                1.0 / climate_w))

    # We take the minimum of the following values (phi, aet_p)
    # to determine the evapotranspiration partition of the
    # water balance (see users guide)
    veg_result = numpy.where(phi < aet_p, phi, aet_p)
    # Take the minimum of precip and Kc * ETo to avoid x / p > 1.0
    nonveg_result = Kc[valid_mask] * eto[valid_mask]
    nonveg_mask = precip[valid_mask] < Kc[valid_mask] * eto[valid_mask]
    nonveg_result[nonveg_mask] = precip[valid_mask][nonveg_mask]
    nonveg_result_fract = nonveg_result / precip[valid_mask]

    # If veg is 1.0 use the result for vegetated areas else use result
    # for non veg areas
    result = numpy.where(
        veg[valid_mask] == 1.0,
        veg_result, nonveg_result_fract)

    fractp = numpy.empty(valid_mask.shape, dtype=numpy.float32)
    fractp[:] = nodata_dict['out_nodata']
    fractp[valid_mask] = result
    return fractp


def pet_op(eto_pix, Kc_pix, eto_nodata, output_nodata):
    """Calculate the plant potential evapotranspiration.

    Args:
        eto_pix (numpy.ndarray): a numpy array of ETo
        Kc_pix (numpy.ndarray): a numpy array of  Kc coefficient
        precip_nodata (float): nodata value from the precip raster
        output_nodata (float): nodata value assigned to output of
            raster_calculator

    Returns:
        numpy.ndarray of potential evapotranspiration (mm)

    """
    result = numpy.empty(eto_pix.shape, dtype=numpy.float32)
    result[:] = output_nodata

    valid_mask = ~numpy.isclose(Kc_pix, output_nodata)
    if eto_nodata is not None:
        valid_mask &= ~numpy.isclose(eto_pix, eto_nodata)
    result[valid_mask] = eto_pix[valid_mask] * Kc_pix[valid_mask]
    return result


def compute_watershed_valuation(watershed_results_vector_path, val_dict):
    """Compute net present value and energy for the watersheds.

    Args:
        watershed_results_vector_path (string):
            Path to an OGR shapefile for the watershed results.
            Where the results will be added.
        val_dict (dict): a python dictionary that has all the valuation
            parameters for each watershed.

    Returns:
        None.

    """
    ws_ds = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    ws_layer = ws_ds.GetLayer()

    # The field names for the new attributes
    energy_field = 'hp_energy'
    npv_field = 'hp_val'

    # Add the new fields to the shapefile
    for new_field in [energy_field, npv_field]:
        field_defn = ogr.FieldDefn(new_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        ws_layer.CreateField(field_defn)

    ws_layer.ResetReading()
    # Iterate over the number of features (polygons)
    for ws_feat in ws_layer:
        # Get the watershed ID to index into the valuation parameter dictionary
        # Since we only allow valuation on watersheds (not subwatersheds)
        # it's okay to hardcode 'ws_id' here.
        ws_id = ws_feat.GetField('ws_id')
        # Get the rsupply volume for the watershed
        rsupply_vl = ws_feat.GetField('rsupply_vl')

        # there won't be a rsupply_vl value if the polygon feature only
        # covers nodata raster values, so check before doing math.
        if rsupply_vl is not None:
            # Get the valuation parameters for watershed 'ws_id'
            val_row = val_dict[ws_id]

            # Compute hydropower energy production (KWH)
            # This is from the equation given in the Users' Guide
            energy = (
                val_row['efficiency'] * val_row['fraction'] *
                val_row['height'] * rsupply_vl * 0.00272)

            dsum = 0.
            # Divide by 100 because it is input at a percent and we need
            # decimal value
            disc = val_row['discount'] / 100.0
            # To calculate the summation of the discount rate term over the life
            # span of the dam we can use a geometric series
            ratio = 1. / (1. + disc)
            if ratio != 1.:
                dsum = (1. - math.pow(ratio, val_row['time_span'])) / (1. - ratio)

            npv = ((val_row['kw_price'] * energy) - val_row['cost']) * dsum

            # Get the volume field index and add value
            ws_feat.SetField(energy_field, energy)
            ws_feat.SetField(npv_field, npv)

            ws_layer.SetFeature(ws_feat)


def compute_rsupply_volume(watershed_results_vector_path):
    """Calculate the total realized water supply volume.

    And the mean realized water supply volume per pixel for the given sheds.
    Output units in cubic meters and cubic meters per pixel respectively.

    Args:
        watershed_results_vector_path (string): a path to a vector that
            contains fields 'wyield_vol' and 'wyield_mn'.

    Returns:
        None.

    """
    ws_ds = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    ws_layer = ws_ds.GetLayer()

    # The field names for the new attributes
    rsupply_vol_name = 'rsupply_vl'
    rsupply_mn_name = 'rsupply_mn'

    # Add the new fields to the shapefile
    for new_field in [rsupply_vol_name, rsupply_mn_name]:
        field_defn = ogr.FieldDefn(new_field, ogr.OFTReal)
        field_defn.SetWidth(24)
        field_defn.SetPrecision(11)
        ws_layer.CreateField(field_defn)

    ws_layer.ResetReading()
    # Iterate over the number of features (polygons)
    for ws_feat in ws_layer:
        # Get mean and volume water yield values
        wyield_mn = ws_feat.GetField('wyield_mn')
        wyield = ws_feat.GetField('wyield_vol')

        # Get water demand/consumption values
        consump_vol = ws_feat.GetField('consum_vol')
        consump_mn = ws_feat.GetField('consum_mn')

        # Calculate realized supply
        # these values won't exist if the polygon feature only
        # covers nodata raster values, so check before doing math.
        if wyield_mn is not None and consump_mn is not None:
            rsupply_vol = wyield - consump_vol
            rsupply_mn = wyield_mn - consump_mn

            # Set values for the new rsupply fields
            ws_feat.SetField(rsupply_vol_name, rsupply_vol)
            ws_feat.SetField(rsupply_mn_name, rsupply_mn)

            ws_layer.SetFeature(ws_feat)


def compute_water_yield_volume(watershed_results_vector_path):
    """Calculate the water yield volume per sub-watershed or watershed.

    Results are added to a 'wyield_vol' field in
    `watershed_results_vector_path`. Units are cubic meters.

    Args:
        watershed_results_vector_path (str): Path to a sub-watershed
            or watershed vector. This vector's features should have a
            'wyield_mn' attribute.

    Returns:
        None.

    """
    shape = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    layer = shape.GetLayer()

    # The field names for the new attributes
    vol_name = 'wyield_vol'

    # Add the new field to the shapefile
    field_defn = ogr.FieldDefn(vol_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)

    layer.ResetReading()
    # Iterate over the number of features (polygons) and compute volume
    for feat in layer:
        wyield_mn = feat.GetField('wyield_mn')
        # there won't be a wyield_mn value if the polygon feature only
        # covers nodata raster values, so check before doing math.
        if wyield_mn is not None:
            geom = feat.GetGeometryRef()
            # Calculate water yield volume,
            # 1000 is for converting the mm of wyield to meters
            vol = wyield_mn * geom.Area() / 1000.0
            # Get the volume field index and add value
            feat.SetField(vol_name, vol)

            layer.SetFeature(feat)


def _add_zonal_stats_dict_to_shape(
        watershed_results_vector_path,
        stats_map, field_name, aggregate_field_id):
    """Add a new field to a shapefile with values from a dictionary.

    Args:
        watershed_results_vector_path (string): a path to a vector whose FIDs
            correspond with the keys in `stats_map`.
        stats_map (dict): a dictionary in the format generated by
            pygeoprocessing.zonal_statistics that contains at least the key
            value of `aggregate_field_id` per feature id.
        field_name (str): a string for the name of the new field to add to
            the target vector.
        aggregate_field_id (string): one of 'min' 'max' 'sum' 'mean' 'count'
            or 'nodata_count' as defined by pygeoprocessing.zonal_statistics.

    Returns:
        None

    """
    vector = gdal.OpenEx(
        watershed_results_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    layer = vector.GetLayer()

    # Create the new field
    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    field_defn.SetWidth(24)
    field_defn.SetPrecision(11)
    layer.CreateField(field_defn)

    # Get the number of features (polygons) and iterate through each
    layer.ResetReading()
    for feature in layer:
        feature_fid = feature.GetFID()

        # Using the unique feature ID, index into the
        # dictionary to get the corresponding value
        # only write a value if zonal stats found valid pixels in the polygon:
        if stats_map[feature_fid]['count'] > 0:
            if aggregate_field_id == 'mean':
                field_val = float(
                    stats_map[feature_fid]['sum']) / stats_map[feature_fid]['count']
            else:
                field_val = float(stats_map[feature_fid][aggregate_field_id])

            # Set the value for the new field
            feature.SetField(field_name, field_val)

            layer.SetFeature(feature)


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
    return validation.validate(
        args, ARGS_SPEC['args'], ARGS_SPEC['args_with_spatial_overlap'])
