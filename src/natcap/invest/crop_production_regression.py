"""InVEST Crop Production Percentile Model."""
import collections
import os
import logging

import numpy
from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import taskgraph

from .spec_utils import u
from . import spec_utils
from . import utils
from . import validation
from .model_metadata import MODEL_METADATA
from . import gettext


LOGGER = logging.getLogger(__name__)

CROPS = {
    "barley": {"description": gettext("barley")},
    "maize": {"description": gettext("maize")},
    "oilpalm": {"description": gettext("oil palm")},
    "potato": {"description": gettext("potato")},
    "rice": {"description": gettext("rice")},
    "soybean": {"description": gettext("soybean")},
    "sugarbeet": {"description": gettext("sugar beet")},
    "sugarcane": {"description": gettext("sugarcane")},
    "wheat": {"description": gettext("wheat")}
}


ARGS_SPEC = {
    "model_name": MODEL_METADATA["crop_production_regression"].model_title,
    "pyname": MODEL_METADATA["crop_production_regression"].pyname,
    "userguide": MODEL_METADATA["crop_production_regression"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["landcover_raster_path", "aggregate_polygon_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "landcover_raster_path": {
            **spec_utils.LULC,
            "projected": True,
            "projection_units": u.meter
        },
        "landcover_to_crop_table_path": {
            "type": "csv",
            "columns": {
                "lucode": {"type": "integer"},
                "crop_name": {
                    "type": "option_string",
                    "options": CROPS
                }
            },
            "about": gettext(
                "A table that maps each LULC code from the LULC map to one of "
                "the 10 canonical crop names representing the crop grown in "
                "that LULC class."),
            "name": gettext("LULC to crop table")
        },
        "fertilization_rate_table_path": {
            "type": "csv",
            "columns": {
                "crop_name": {
                    "type": "option_string",
                    "options": CROPS,
                    "about": gettext("One of the supported crop types.")
                },
                **{f"{nutrient}_rate": {
                    "type": "number",
                    "units": u.kilogram/u.hectare,
                    "about": f"Rate of {nutrient} application for the crop."
                } for nutrient in ["nitrogen", "phosphorus", "potassium"]}
            },
            "about": gettext(
                "A table that maps crops to fertilizer application rates."),
            "name": gettext("fertilization rate table")
        },
        "aggregate_polygon_path": {
            **spec_utils.AOI,
            "required": False
        },
        "model_data_path": {
            "type": "directory",
            "contents": {
                "climate_regression_yield_tables": {
                    "type": "directory",
                    "contents": {
                        "[CROP]_regression_yield_table.csv": {
                            "type": "csv",
                            "columns": {
                                'climate_bin': {"type": "integer"},
                                'yield_ceiling': {
                                    "type": "number",
                                    "units": u.metric_ton/u.hectare
                                },
                                'b_nut':  {"type": "number", "units": u.none},
                                'b_k2o':  {"type": "number", "units": u.none},
                                'c_n':    {"type": "number", "units": u.none},
                                'c_p2o5': {"type": "number", "units": u.none},
                                'c_k2o':  {"type": "number", "units": u.none}
                            }
                        }
                    }
                },
                "crop_nutrient.csv": {
                    "type": "csv",
                    "columns": {
                        nutrient: {
                            "type": "number",
                            "units": units
                        } for nutrient, units in {
                            "protein":     u.gram/u.hectogram,
                            "lipid":       u.gram/u.hectogram,       # total lipid
                            "energy":      u.kilojoule/u.hectogram,
                            "ca":          u.milligram/u.hectogram,  # calcium
                            "fe":          u.milligram/u.hectogram,  # iron
                            "mg":          u.milligram/u.hectogram,  # magnesium
                            "ph":          u.milligram/u.hectogram,  # phosphorus
                            "k":           u.milligram/u.hectogram,  # potassium
                            "na":          u.milligram/u.hectogram,  # sodium
                            "zn":          u.milligram/u.hectogram,  # zinc
                            "cu":          u.milligram/u.hectogram,  # copper
                            "fl":          u.microgram/u.hectogram,  # fluoride
                            "mn":          u.milligram/u.hectogram,  # manganese
                            "se":          u.microgram/u.hectogram,  # selenium
                            "vita":        u.IU/u.hectogram,         # vitamin A
                            "betac":       u.microgram/u.hectogram,  # beta carotene
                            "alphac":      u.microgram/u.hectogram,  # alpha carotene
                            "vite":        u.milligram/u.hectogram,  # vitamin e
                            "crypto":      u.microgram/u.hectogram,  # cryptoxanthin
                            "lycopene":    u.microgram/u.hectogram,  # lycopene
                            "lutein":      u.microgram/u.hectogram,  # lutein + zeaxanthin
                            "betaT":       u.milligram/u.hectogram,  # beta tocopherol
                            "gammaT":      u.milligram/u.hectogram,  # gamma tocopherol
                            "deltaT":      u.milligram/u.hectogram,  # delta tocopherol
                            "vitc":        u.milligram/u.hectogram,  # vitamin C
                            "thiamin":     u.milligram/u.hectogram,
                            "riboflavin":  u.milligram/u.hectogram,
                            "niacin":      u.milligram/u.hectogram,
                            "pantothenic": u.milligram/u.hectogram,  # pantothenic acid
                            "vitb6":       u.milligram/u.hectogram,  # vitamin B6
                            "folate":      u.microgram/u.hectogram,
                            "vitb12":      u.microgram/u.hectogram,  # vitamin B12
                            "vitk":        u.microgram/u.hectogram,  # vitamin K
                        }.items()
                    }
                }
            },
            "about": gettext("The Crop Production datasets provided with the model."),
            "name": gettext("model data")
        }
    }
}

_INTERMEDIATE_OUTPUT_DIR = 'intermediate_output'

_REGRESSION_TABLE_PATTERN = os.path.join(
    'climate_regression_yield_tables', '%s_regression_yield_table.csv')

_EXPECTED_REGRESSION_TABLE_HEADERS = [
    'climate_bin', 'yield_ceiling', 'b_nut', 'b_k2o', 'c_n', 'c_p2o5', 'c_k2o']

# crop_name, yield_regression_id, file_suffix
_COARSE_YIELD_REGRESSION_PARAMETER_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_%s_coarse_regression_parameter%s.tif')

# crop_name, yield_regression_id
_INTERPOLATED_YIELD_REGRESSION_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_%s_interpolated_regression_parameter%s.tif')

# crop_id, file_suffix
_NITROGEN_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_nitrogen_yield%s.tif')

# crop_id, file_suffix
_PHOSPHORUS_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_phosphorus_yield%s.tif')

# crop_id, file_suffix
_POTASSIUM_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_potassium_yield%s.tif')

# file suffix
_CLIPPED_NITROGEN_RATE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'nitrogen_rate%s.tif')

# file suffix
_CLIPPED_PHOSPHORUS_RATE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'phosphorus_rate%s.tif')

# file suffix
_CLIPPED_POTASSIUM_RATE_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'potassium_rate%s.tif')

# file suffix
_CLIPPED_IRRIGATION_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'irrigation_mask%s.tif')

# crop_name, file_suffix
_CROP_PRODUCTION_FILE_PATTERN = os.path.join(
    '.', '%s_regression_production%s.tif')

# crop_name, file_suffix
_N_REQ_RF_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_n_req_rf_%s.tif')

# crop_name, file_suffix
_P_REQ_RF_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_p_req_rf_%s.tif')

# crop_name, file_suffix
_K_REQ_RF_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_k_req_rf_%s.tif')


_GLOBAL_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    'observed_yield', '%s_yield_map.tif')  # crop_name
_EXTENDED_CLIMATE_BIN_FILE_PATTERN = os.path.join(
    'extended_climate_bin_maps', 'extendedclimatebins%s.tif')  # crop_name

# crop_name, file_suffix
_CLIPPED_CLIMATE_BIN_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR,
    'clipped_%s_climate_bin_map%s.tif')

# crop_name, file_suffix
_CLIPPED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_clipped_observed_yield%s.tif')

# crop_name, file_suffix
_ZEROED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_zeroed_observed_yield%s.tif')

# crop_name, file_suffix
_INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, '%s_interpolated_observed_yield%s.tif')

# crop_name, file_suffix
_OBSERVED_PRODUCTION_FILE_PATTERN = os.path.join(
    '.', '%s_observed_production%s.tif')

# file_suffix
_AGGREGATE_VECTOR_FILE_PATTERN = os.path.join(
    _INTERMEDIATE_OUTPUT_DIR, 'aggregate_vector%s.shp')

# file_suffix
_AGGREGATE_TABLE_FILE_PATTERN = os.path.join(
    '.', 'aggregate_results%s.csv')

_EXPECTED_NUTRIENT_TABLE_HEADERS = [
    'Protein', 'Lipid', 'Energy', 'Ca', 'Fe', 'Mg', 'Ph', 'K', 'Na', 'Zn',
    'Cu', 'Fl', 'Mn', 'Se', 'VitA', 'betaC', 'alphaC', 'VitE', 'Crypto',
    'Lycopene', 'Lutein', 'betaT', 'gammaT', 'deltaT', 'VitC', 'Thiamin',
    'Riboflavin', 'Niacin', 'Pantothenic', 'VitB6', 'Folate', 'VitB12',
    'VitK']
_EXPECTED_LUCODE_TABLE_HEADER = 'lucode'
_NODATA_YIELD = -1


def execute(args):
    """Crop Production Regression.

    This model will take a landcover (crop cover?), N, P, and K map and
    produce modeled yields, and a nutrient table.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['landcover_raster_path'] (string): path to landcover raster
        args['landcover_to_crop_table_path'] (string): path to a table that
            converts landcover types to crop names that has two headers:

            * lucode: integer value corresponding to a landcover code in
              `args['landcover_raster_path']`.
            * crop_name: a string that must match one of the crops in
              args['model_data_path']/climate_regression_yield_tables/[cropname]_*
              A ValueError is raised if strings don't match.

        args['fertilization_rate_table_path'] (string): path to CSV table
            that contains fertilization rates for the crops in the simulation,
            though it can contain additional crops not used in the simulation.
            The headers must be 'crop_name', 'nitrogen_rate',
            'phosphorus_rate', and 'potassium_rate', where 'crop_name' is the
            name string used to identify crops in the
            'landcover_to_crop_table_path', and rates are in units kg/Ha.
        args['aggregate_polygon_path'] (string): path to polygon vector
            that will be used to aggregate crop yields and total nutrient
            value. (optional, if value is None, then skipped)
        args['model_data_path'] (string): path to the InVEST Crop Production
            global data directory.  This model expects that the following
            directories are subdirectories of this path:

            * climate_bin_maps (contains [cropname]_climate_bin.tif files)
            * climate_percentile_yield (contains
              [cropname]_percentile_yield_table.csv files)

            Please see the InVEST user's guide chapter on crop production for
            details about how to download these data.

    Returns:
        None.

    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories([
        output_dir, os.path.join(output_dir, _INTERMEDIATE_OUTPUT_DIR)])

    # Initialize a TaskGraph
    work_token_dir = os.path.join(
        output_dir, _INTERMEDIATE_OUTPUT_DIR, '_taskgraph_working_dir')
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Single process mode.
    task_graph = taskgraph.TaskGraph(work_token_dir, n_workers)
    dependent_task_list = []

    LOGGER.info(
        "Checking if the landcover raster is missing lucodes")
    crop_to_landcover_table = utils.build_lookup_from_csv(
        args['landcover_to_crop_table_path'], 'crop_name', to_lower=True)

    crop_to_fertlization_rate_table = utils.build_lookup_from_csv(
        args['fertilization_rate_table_path'], 'crop_name', to_lower=True)

    crop_lucodes = [
        x[_EXPECTED_LUCODE_TABLE_HEADER]
        for x in crop_to_landcover_table.values()]

    unique_lucodes = numpy.array([])
    for _, lu_band_data in pygeoprocessing.iterblocks(
            (args['landcover_raster_path'], 1)):
        unique_block = numpy.unique(lu_band_data)
        unique_lucodes = numpy.unique(numpy.concatenate(
            (unique_lucodes, unique_block)))

    missing_lucodes = set(crop_lucodes).difference(
        set(unique_lucodes))
    if len(missing_lucodes) > 0:
        LOGGER.warning(
            "The following lucodes are in the landcover to crop table but "
            "aren't in the landcover raster: %s", missing_lucodes)

    LOGGER.info("Checking that crops correspond to known types.")
    for crop_name in crop_to_landcover_table:
        crop_lucode = crop_to_landcover_table[crop_name][
            _EXPECTED_LUCODE_TABLE_HEADER]
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'],
            _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)
        if not os.path.exists(crop_climate_bin_raster_path):
            raise ValueError(
                "Expected climate bin map called %s for crop %s "
                "specified in %s", crop_climate_bin_raster_path, crop_name,
                args['landcover_to_crop_table_path'])

    landcover_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    pixel_area_ha = numpy.product([
        abs(x) for x in landcover_raster_info['pixel_size']]) / 10000
    landcover_nodata = landcover_raster_info['nodata'][0]
    if landcover_nodata is None:
        LOGGER.warning(
            "%s does not have nodata value defined; "
            "assuming all pixel values are valid"
            % args['landcover_raster_path'])

    # Calculate lat/lng bounding box for landcover map
    wgs84srs = osr.SpatialReference()
    wgs84srs.ImportFromEPSG(4326)  # EPSG4326 is WGS84 lat/lng
    landcover_wgs84_bounding_box = pygeoprocessing.transform_bounding_box(
        landcover_raster_info['bounding_box'],
        landcover_raster_info['projection_wkt'], wgs84srs.ExportToWkt(),
        edge_samples=11)

    crop_lucode = None
    observed_yield_nodata = None

    for crop_name in crop_to_landcover_table:
        crop_lucode = crop_to_landcover_table[crop_name][
            _EXPECTED_LUCODE_TABLE_HEADER]
        LOGGER.info("Processing crop %s", crop_name)
        crop_climate_bin_raster_path = os.path.join(
            args['model_data_path'],
            _EXTENDED_CLIMATE_BIN_FILE_PATTERN % crop_name)

        LOGGER.info(
            "Clipping global climate bin raster to landcover bounding box.")
        clipped_climate_bin_raster_path = os.path.join(
            output_dir, _CLIPPED_CLIMATE_BIN_FILE_PATTERN % (
                crop_name, file_suffix))
        crop_climate_bin_raster_info = pygeoprocessing.get_raster_info(
            crop_climate_bin_raster_path)
        crop_climate_bin_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(crop_climate_bin_raster_path,
                  crop_climate_bin_raster_info['pixel_size'],
                  clipped_climate_bin_raster_path, 'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[clipped_climate_bin_raster_path],
            task_name='crop_climate_bin')
        dependent_task_list.append(crop_climate_bin_task)

        crop_regression_table_path = os.path.join(
            args['model_data_path'], _REGRESSION_TABLE_PATTERN % crop_name)

        crop_regression_table = utils.build_lookup_from_csv(
            crop_regression_table_path, 'climate_bin', to_lower=True)
        for bin_id in crop_regression_table:
            for header in _EXPECTED_REGRESSION_TABLE_HEADERS:
                if crop_regression_table[bin_id][header.lower()] == '':
                    crop_regression_table[bin_id][header.lower()] = 0

        yield_regression_headers = [
            x for x in list(crop_regression_table.values())[0]
            if x != 'climate_bin']

        reclassify_error_details = {
            'raster_name': f'{crop_name} Climate Bin',
            'column_name': 'climate_bin',
            'table_name': f'Climate {crop_name} Regression Yield'}
        regression_parameter_raster_path_lookup = {}
        for yield_regression_id in yield_regression_headers:
            # there are extra headers in that table
            if yield_regression_id not in _EXPECTED_REGRESSION_TABLE_HEADERS:
                continue
            LOGGER.info("Map %s to climate bins.", yield_regression_id)
            regression_parameter_raster_path_lookup[yield_regression_id] = (
                os.path.join(
                    output_dir,
                    _INTERPOLATED_YIELD_REGRESSION_FILE_PATTERN % (
                        crop_name, yield_regression_id, file_suffix)))
            bin_to_regression_value = dict([
                (bin_id,
                 crop_regression_table[bin_id][yield_regression_id])
                for bin_id in crop_regression_table])
            # reclassify nodata to a valid value of 0
            # we're assuming that the crop doesn't exist where there is no data
            # this is more likely than assuming the crop does exist, esp.
            # in the context of the provided climate bins map
            bin_to_regression_value[
                crop_climate_bin_raster_info['nodata'][0]] = 0
            coarse_regression_parameter_raster_path = os.path.join(
                output_dir,
                _COARSE_YIELD_REGRESSION_PARAMETER_FILE_PATTERN % (
                    crop_name, yield_regression_id, file_suffix))
            create_coarse_regression_parameter_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=((clipped_climate_bin_raster_path, 1),
                      bin_to_regression_value,
                      coarse_regression_parameter_raster_path,
                      gdal.GDT_Float32, _NODATA_YIELD,
                      reclassify_error_details),
                target_path_list=[coarse_regression_parameter_raster_path],
                dependent_task_list=[crop_climate_bin_task],
                task_name='create_coarse_regression_parameter_%s_%s' % (
                    crop_name, yield_regression_id))
            dependent_task_list.append(create_coarse_regression_parameter_task)

            LOGGER.info(
                "Interpolate %s %s parameter to landcover resolution.",
                crop_name, yield_regression_id)
            create_interpolated_parameter_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(coarse_regression_parameter_raster_path,
                      landcover_raster_info['pixel_size'],
                      regression_parameter_raster_path_lookup[yield_regression_id],
                      'cubicspline'),
                kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                        'target_bb': landcover_raster_info['bounding_box']},
                target_path_list=[
                    regression_parameter_raster_path_lookup[yield_regression_id]],
                dependent_task_list=[
                    create_coarse_regression_parameter_task],
                task_name='create_interpolated_parameter_%s_%s' % (
                    crop_name, yield_regression_id))
            dependent_task_list.append(create_interpolated_parameter_task)

        LOGGER.info('Calc nitrogen yield')
        nitrogen_yield_raster_path = os.path.join(
            output_dir, _NITROGEN_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))
        calc_nitrogen_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(regression_parameter_raster_path_lookup['yield_ceiling'], 1),
                   (regression_parameter_raster_path_lookup['b_nut'], 1),
                   (regression_parameter_raster_path_lookup['c_n'], 1),
                   (args['landcover_raster_path'], 1),
                   (crop_to_fertlization_rate_table[crop_name]
                    ['nitrogen_rate'], 'raw'),
                   (crop_lucode, 'raw'), (pixel_area_ha, 'raw')],
                  _x_yield_op,
                  nitrogen_yield_raster_path, gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[nitrogen_yield_raster_path],
            dependent_task_list=dependent_task_list,
            task_name='calculate_nitrogen_yield_%s' % crop_name)

        LOGGER.info('Calc phosphorus yield')
        phosphorus_yield_raster_path = os.path.join(
            output_dir, _PHOSPHORUS_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))
        calc_phosphorus_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(regression_parameter_raster_path_lookup['yield_ceiling'], 1),
                   (regression_parameter_raster_path_lookup['b_nut'], 1),
                   (regression_parameter_raster_path_lookup['c_p2o5'], 1),
                   (args['landcover_raster_path'], 1),
                   (crop_to_fertlization_rate_table[crop_name]
                    ['phosphorus_rate'], 'raw'),
                   (crop_lucode, 'raw'), (pixel_area_ha, 'raw')],
                  _x_yield_op,
                  phosphorus_yield_raster_path, gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[phosphorus_yield_raster_path],
            dependent_task_list=dependent_task_list,
            task_name='calculate_phosphorus_yield_%s' % crop_name)

        LOGGER.info('Calc potassium yield')
        potassium_yield_raster_path = os.path.join(
            output_dir, _POTASSIUM_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))
        calc_potassium_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(regression_parameter_raster_path_lookup['yield_ceiling'], 1),
                   (regression_parameter_raster_path_lookup['b_k2o'], 1),
                   (regression_parameter_raster_path_lookup['c_k2o'], 1),
                   (args['landcover_raster_path'], 1),
                   (crop_to_fertlization_rate_table[crop_name]
                    ['potassium_rate'], 'raw'),
                   (crop_lucode, 'raw'), (pixel_area_ha, 'raw')],
                  _x_yield_op,
                  potassium_yield_raster_path, gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[potassium_yield_raster_path],
            dependent_task_list=dependent_task_list,
            task_name='calculate_potassium_yield_%s' % crop_name)

        dependent_task_list.extend((
            calc_nitrogen_yield_task,
            calc_phosphorus_yield_task,
            calc_potassium_yield_task))

        LOGGER.info('Calc the min of N, K, and P')
        crop_production_raster_path = os.path.join(
            output_dir, _CROP_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))

        calc_min_NKP_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(nitrogen_yield_raster_path, 1),
                   (phosphorus_yield_raster_path, 1),
                   (potassium_yield_raster_path, 1)],
                  _min_op, crop_production_raster_path,
                  gdal.GDT_Float32, _NODATA_YIELD),
            target_path_list=[crop_production_raster_path],
            dependent_task_list=dependent_task_list,
            task_name='calc_min_of_NKP')
        dependent_task_list.append(calc_min_NKP_task)

        LOGGER.info("Calculate observed yield for %s", crop_name)
        global_observed_yield_raster_path = os.path.join(
            args['model_data_path'],
            _GLOBAL_OBSERVED_YIELD_FILE_PATTERN % crop_name)
        global_observed_yield_raster_info = (
            pygeoprocessing.get_raster_info(
                global_observed_yield_raster_path))
        clipped_observed_yield_raster_path = os.path.join(
            output_dir, _CLIPPED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))
        clip_global_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(global_observed_yield_raster_path,
                  global_observed_yield_raster_info['pixel_size'],
                  clipped_observed_yield_raster_path, 'near'),
            kwargs={'target_bb': landcover_wgs84_bounding_box},
            target_path_list=[clipped_observed_yield_raster_path],
            task_name='clip_global_observed_yield_%s_' % crop_name)
        dependent_task_list.append(clip_global_observed_yield_task)

        observed_yield_nodata = (
            global_observed_yield_raster_info['nodata'][0])

        zeroed_observed_yield_raster_path = os.path.join(
            output_dir, _ZEROED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))

        nodata_to_zero_for_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(clipped_observed_yield_raster_path, 1),
                   (observed_yield_nodata, 'raw')],
                  _zero_observed_yield_op, zeroed_observed_yield_raster_path,
                  gdal.GDT_Float32, observed_yield_nodata),
            target_path_list=[zeroed_observed_yield_raster_path],
            dependent_task_list=[clip_global_observed_yield_task],
            task_name='nodata_to_zero_for_observed_yield_%s_' % crop_name)
        dependent_task_list.append(nodata_to_zero_for_observed_yield_task)

        interpolated_observed_yield_raster_path = os.path.join(
            output_dir, _INTERPOLATED_OBSERVED_YIELD_FILE_PATTERN % (
                crop_name, file_suffix))

        LOGGER.info(
            "Interpolating observed %s raster to landcover.", crop_name)
        interpolate_observed_yield_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(zeroed_observed_yield_raster_path,
                  landcover_raster_info['pixel_size'],
                  interpolated_observed_yield_raster_path, 'cubicspline'),
            kwargs={'target_projection_wkt': landcover_raster_info['projection_wkt'],
                    'target_bb': landcover_raster_info['bounding_box']},
            target_path_list=[interpolated_observed_yield_raster_path],
            dependent_task_list=[nodata_to_zero_for_observed_yield_task],
            task_name='interpolate_observed_yield_to_lulc_%s' % crop_name)
        dependent_task_list.append(interpolate_observed_yield_task)

        observed_production_raster_path = os.path.join(
            output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))
        calculate_observed_production_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(args['landcover_raster_path'], 1),
                   (interpolated_observed_yield_raster_path, 1),
                   (observed_yield_nodata, 'raw'), (landcover_nodata, 'raw'),
                   (crop_lucode, 'raw'), (pixel_area_ha, 'raw')],
                  _mask_observed_yield_op, observed_production_raster_path,
                  gdal.GDT_Float32, observed_yield_nodata),
            target_path_list=[observed_production_raster_path],
            dependent_task_list=[interpolate_observed_yield_task],
            task_name='calculate_observed_production_%s' % crop_name)
        dependent_task_list.append(calculate_observed_production_task)

    # both 'crop_nutrient.csv' and 'crop' are known data/header values for
    # this model data.
    nutrient_table = utils.build_lookup_from_csv(
        os.path.join(args['model_data_path'], 'crop_nutrient.csv'),
        'crop', to_lower=False)

    LOGGER.info("Generating report table")
    result_table_path = os.path.join(
        output_dir, 'result_table%s.csv' % file_suffix)
    _ = task_graph.add_task(
        func=tabulate_regression_results,
        args=(nutrient_table,
              crop_to_landcover_table, pixel_area_ha,
              args['landcover_raster_path'], landcover_nodata,
              output_dir, file_suffix, result_table_path),
        target_path_list=[result_table_path],
        dependent_task_list=dependent_task_list,
        task_name='tabulate_results')

    if ('aggregate_polygon_path' in args and
            args['aggregate_polygon_path'] not in ['', None]):
        LOGGER.info("aggregating result over query polygon")
        # reproject polygon to LULC's projection
        target_aggregate_vector_path = os.path.join(
            output_dir, _AGGREGATE_VECTOR_FILE_PATTERN % (file_suffix))
        aggregate_results_table_path = os.path.join(
            output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)
        _ = task_graph.add_task(
            func=aggregate_regression_results_to_polygons,
            args=(args['aggregate_polygon_path'],
                  target_aggregate_vector_path,
                  landcover_raster_info['projection_wkt'],
                  crop_to_landcover_table, nutrient_table,
                  output_dir, file_suffix,
                  aggregate_results_table_path),
            target_path_list=[target_aggregate_vector_path,
                              aggregate_results_table_path],
            dependent_task_list=dependent_task_list,
            task_name='aggregate_results_to_polygons')

    task_graph.close()
    task_graph.join()


def _x_yield_op(
        y_max, b_x, c_x, lulc_array, fert_rate, crop_lucode, pixel_area_ha):
    """Calc generalized yield op, Ymax*(1-b_NP*exp(-cN * N_GC)).

    The regression model has identical mathematical equations for
    the nitrogen, phosphorus, and potassium.  The only difference is
    the scalars in the equation (fertilization rate and pixel area).
    """
    result = numpy.empty(b_x.shape, dtype=numpy.float32)
    result[:] = _NODATA_YIELD
    valid_mask = (
        ~utils.array_equals_nodata(y_max,  _NODATA_YIELD) &
        ~utils.array_equals_nodata(b_x, _NODATA_YIELD) &
        ~utils.array_equals_nodata(c_x, _NODATA_YIELD) &
        (lulc_array == crop_lucode))
    result[valid_mask] = pixel_area_ha * y_max[valid_mask] * (
        1 - b_x[valid_mask] * numpy.exp(
            -c_x[valid_mask] * fert_rate))

    return result


def _min_op(y_n, y_p, y_k):
    """Calculate the min of the three inputs and multiply by Ymax."""
    result = numpy.empty(y_n.shape, dtype=numpy.float32)
    result[:] = _NODATA_YIELD
    valid_mask = (
        ~utils.array_equals_nodata(y_n, _NODATA_YIELD) &
        ~utils.array_equals_nodata(y_k, _NODATA_YIELD) &
        ~utils.array_equals_nodata(y_p, _NODATA_YIELD))
    result[valid_mask] = (
        numpy.min(
            [y_n[valid_mask], y_k[valid_mask], y_p[valid_mask]],
            axis=0))
    return result


def _zero_observed_yield_op(observed_yield_array, observed_yield_nodata):
    """Reclassify observed_yield nodata to zero.

    Args:
        observed_yield_array (numpy.ndarray): raster values
        observed_yield_nodata (float): raster nodata value

    Returns:
        numpy.ndarray with observed yield values

    """
    result = numpy.empty(
        observed_yield_array.shape, dtype=numpy.float32)
    result[:] = 0
    valid_mask = slice(None)
    if observed_yield_nodata is not None:
        valid_mask = ~utils.array_equals_nodata(
            observed_yield_array, observed_yield_nodata)
    result[valid_mask] = observed_yield_array[valid_mask]
    return result


def _mask_observed_yield_op(
        lulc_array, observed_yield_array, observed_yield_nodata,
        landcover_nodata, crop_lucode, pixel_area_ha):
    """Mask total observed yield to crop lulc type.

    Args:
        lulc_array (numpy.ndarray): landcover raster values
        observed_yield_array (numpy.ndarray): yield raster values
        observed_yield_nodata (float): yield raster nodata value
        landcover_nodata (float): landcover raster nodata value
        crop_lucode (int): code used to mask in the current crop
        pixel_area_ha (float): area of lulc raster cells (hectares)

    Returns:
        numpy.ndarray with float values of yields masked to crop_lucode

    """
    result = numpy.empty(lulc_array.shape, dtype=numpy.float32)
    if landcover_nodata is not None:
        result[:] = observed_yield_nodata
        valid_mask = ~utils.array_equals_nodata(lulc_array, landcover_nodata)
        result[valid_mask] = 0
    else:
        result[:] = 0
    lulc_mask = lulc_array == crop_lucode
    result[lulc_mask] = (
        observed_yield_array[lulc_mask] * pixel_area_ha)
    return result


def tabulate_regression_results(
        nutrient_table,
        crop_to_landcover_table, pixel_area_ha, landcover_raster_path,
        landcover_nodata, output_dir, file_suffix, target_table_path):
    """Write table with total yield and nutrient results by crop.

    This function includes all the operations that write to results_table.csv.

    Args:
        nutrient_table (dict): a lookup of nutrient values by crop in the
            form of nutrient_table[<crop>][<nutrient>].
        crop_to_landcover_table (dict): landcover codes keyed by crop names
        pixel_area_ha (float): area of lulc raster cells (hectares)
        landcover_raster_path (string): path to landcover raster
        landcover_nodata (float): landcover raster nodata value
        output_dir (string): the file path to the output workspace.
        file_suffix (string): string to append to any output filenames.
        target_table_path (string): path to 'result_table.csv' in the output
            workspace

    Returns:
        None

    """
    nutrient_headers = [
        nutrient_id + '_' + mode
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
        for mode in ['modeled', 'observed']]
    with open(target_table_path, 'w') as result_table:
        result_table.write(
            'crop,area (ha),' + 'production_observed,production_modeled,' +
            ','.join(nutrient_headers) + '\n')
        for crop_name in sorted(crop_to_landcover_table):
            result_table.write(crop_name)
            production_lookup = {}
            production_pixel_count = 0
            yield_sum = 0
            observed_production_raster_path = os.path.join(
                output_dir,
                _OBSERVED_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))

            LOGGER.info(
                "Calculating production area and summing observed yield.")
            observed_yield_nodata = pygeoprocessing.get_raster_info(
                observed_production_raster_path)['nodata'][0]
            for _, yield_block in pygeoprocessing.iterblocks(
                    (observed_production_raster_path, 1)):

                # make a valid mask showing which pixels are not nodata
                # if nodata value undefined, assume all pixels are valid
                valid_mask = numpy.full(yield_block.shape, True)
                if observed_yield_nodata is not None:
                    valid_mask = ~utils.array_equals_nodata(
                        yield_block, observed_yield_nodata)
                production_pixel_count += numpy.count_nonzero(
                    valid_mask & (yield_block > 0.0))
                yield_sum += numpy.sum(yield_block[valid_mask])
            production_area = production_pixel_count * pixel_area_ha
            production_lookup['observed'] = yield_sum
            result_table.write(',%f' % production_area)
            result_table.write(",%f" % yield_sum)

            crop_production_raster_path = os.path.join(
                output_dir, _CROP_PRODUCTION_FILE_PATTERN % (
                    crop_name, file_suffix))
            yield_sum = 0
            for _, yield_block in pygeoprocessing.iterblocks(
                    (crop_production_raster_path, 1)):
                yield_sum += numpy.sum(
                    # _NODATA_YIELD will always have a value (defined above)
                    yield_block[~utils.array_equals_nodata(
                        yield_block, _NODATA_YIELD)])
            production_lookup['modeled'] = yield_sum
            result_table.write(",%f" % yield_sum)

            # convert 100g to Mg and fraction left over from refuse
            nutrient_factor = 1e4 * (
                1 - nutrient_table[crop_name]['Percentrefuse'] / 100)
            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                total_nutrient = (
                    nutrient_factor *
                    production_lookup['modeled'] *
                    nutrient_table[crop_name][nutrient_id])
                result_table.write(",%f" % (total_nutrient))
                result_table.write(
                    ",%f" % (
                        nutrient_factor *
                        production_lookup['observed'] *
                        nutrient_table[crop_name][nutrient_id]))
            result_table.write('\n')

        total_area = 0
        for _, band_values in pygeoprocessing.iterblocks(
                (landcover_raster_path, 1)):
            if landcover_nodata is not None:
                total_area += numpy.count_nonzero(
                    ~utils.array_equals_nodata(band_values, landcover_nodata))
            else:
                total_area += band_values.size
        result_table.write(
            '\n,total area (both crop and non-crop)\n,%f\n' % (
                total_area * pixel_area_ha))


def aggregate_regression_results_to_polygons(
        base_aggregate_vector_path, target_aggregate_vector_path,
        landcover_raster_projection, crop_to_landcover_table,
        nutrient_table, output_dir, file_suffix,
        target_aggregate_table_path):
    """Write table with aggregate results of yield and nutrient values.

    Use zonal statistics to summarize total observed and interpolated
    production and nutrient information for each polygon in
    base_aggregate_vector_path.

    Args:
        base_aggregate_vector_path (string): path to polygon vector
        target_aggregate_vector_path (string):
            path to re-projected copy of polygon vector
        landcover_raster_projection (string): a WKT projection string
        crop_to_landcover_table (dict): landcover codes keyed by crop names
        nutrient_table (dict): a lookup of nutrient values by crop in the
            form of nutrient_table[<crop>][<nutrient>].
        output_dir (string): the file path to the output workspace.
        file_suffix (string): string to append to any output filenames.
        target_aggregate_table_path (string): path to 'aggregate_results.csv'
            in the output workspace

    Returns:
        None

    """
    pygeoprocessing.reproject_vector(
        base_aggregate_vector_path,
        landcover_raster_projection,
        target_aggregate_vector_path,
        driver_name='ESRI Shapefile')

    # loop over every crop and query with pgp function
    total_yield_lookup = {}
    total_nutrient_table = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(
            float)))
    for crop_name in crop_to_landcover_table:
        # convert 100g to Mg and fraction left over from refuse
        nutrient_factor = 1e4 * (
            1 - nutrient_table[crop_name]['Percentrefuse'] / 100)
        LOGGER.info(
            "Calculating zonal stats for %s", crop_name)
        crop_production_raster_path = os.path.join(
            output_dir, _CROP_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))
        total_yield_lookup['%s_modeled' % crop_name] = (
            pygeoprocessing.zonal_statistics(
                (crop_production_raster_path, 1),
                target_aggregate_vector_path))

        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
            for fid_index in total_yield_lookup['%s_modeled' % crop_name]:
                total_nutrient_table[nutrient_id][
                    'modeled'][fid_index] += (
                        nutrient_factor *
                        total_yield_lookup['%s_modeled' % crop_name][
                            fid_index]['sum'] *
                        nutrient_table[crop_name][nutrient_id])

        # process observed
        observed_yield_path = os.path.join(
            output_dir, _OBSERVED_PRODUCTION_FILE_PATTERN % (
                crop_name, file_suffix))
        total_yield_lookup['%s_observed' % crop_name] = (
            pygeoprocessing.zonal_statistics(
                (observed_yield_path, 1),
                target_aggregate_vector_path))
        for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
            for fid_index in total_yield_lookup[
                    '%s_observed' % crop_name]:
                total_nutrient_table[
                    nutrient_id]['observed'][fid_index] += (
                        nutrient_factor *
                        total_yield_lookup[
                            '%s_observed' % crop_name][fid_index]['sum'] *
                        nutrient_table[crop_name][nutrient_id])

    # report everything to a table
    aggregate_table_path = os.path.join(
        output_dir, _AGGREGATE_TABLE_FILE_PATTERN % file_suffix)
    with open(aggregate_table_path, 'w') as aggregate_table:
        # write header
        aggregate_table.write('FID,')
        aggregate_table.write(','.join(sorted(total_yield_lookup)) + ',')
        aggregate_table.write(
            ','.join([
                '%s_%s' % (nutrient_id, model_type)
                for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS
                for model_type in sorted(
                    list(total_nutrient_table.values())[0])]))
        aggregate_table.write('\n')

        # iterate by polygon index
        for id_index in list(total_yield_lookup.values())[0]:
            aggregate_table.write('%s,' % id_index)
            aggregate_table.write(','.join([
                str(total_yield_lookup[yield_header][id_index]['sum'])
                for yield_header in sorted(total_yield_lookup)]))

            for nutrient_id in _EXPECTED_NUTRIENT_TABLE_HEADERS:
                for model_type in sorted(
                        list(total_nutrient_table.values())[0]):
                    aggregate_table.write(
                        ',%s' % total_nutrient_table[
                            nutrient_id][model_type][id_index])
            aggregate_table.write('\n')


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
