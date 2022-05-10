"""InVEST Seasonal Water Yield Model."""
import fractions
import logging
import os
import re
import warnings

import numpy
import pygeoprocessing
import pygeoprocessing.routing
import scipy.special
import taskgraph
from osgeo import gdal
from osgeo import ogr

from .. import gettext
from .. import spec_utils
from .. import utils
from .. import validation
from ..model_metadata import MODEL_METADATA
from ..spec_utils import u
from . import seasonal_water_yield_core

gdal.SetCacheMax(2**26)

LOGGER = logging.getLogger(__name__)

TARGET_NODATA = -1
N_MONTHS = 12
MONTH_ID_TO_LABEL = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec']

ARGS_SPEC = {
    "model_name": MODEL_METADATA["seasonal_water_yield"].model_title,
    "pyname": MODEL_METADATA["seasonal_water_yield"].pyname,
    "userguide": MODEL_METADATA["seasonal_water_yield"].userguide,
    "args_with_spatial_overlap": {
        "spatial_keys": ["dem_raster_path", "lulc_raster_path",
                         "soil_group_path", "aoi_path", "l_path",
                         "climate_zone_raster_path"],
        "different_projections_ok": True,
    },
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "threshold_flow_accumulation": spec_utils.THRESHOLD_FLOW_ACCUMULATION,
        "et0_dir": {
            "type": "directory",
            "contents": {
                # monthly et0 maps, each file ending in a number 1-12
                "[MONTH]": {
                    **spec_utils.ET0,
                    "about": gettext(
                        "Twelve files, one for each month. File names must "
                        "end with the month number (1-12). For example, "
                        "the filenames 'et0_1.tif' "
                        "'evapotranspiration1.tif' are both valid for the "
                        "month of January."),
                },
            },
            "required": "not user_defined_local_recharge",
            "about": gettext(
                "Directory containing maps of reference evapotranspiration "
                "for each month. Only .tif files should be in this folder "
                "(no .tfw, .xml, etc files)."),
            "name": gettext("ET0 directory")
        },
        "precip_dir": {
            "type": "directory",
            "contents": {
                # monthly precipitation maps, each file ending in a number 1-12
                "[MONTH]": {
                    **spec_utils.PRECIP,
                    "about": gettext(
                        "Twelve files, one for each month. File names must "
                        "end with the month number (1-12). For example, "
                        "the filenames 'precip_1.tif' and 'precip1.tif' are "
                        "both valid names for the month of January."),
                },
            },
            "required": "not user_defined_local_recharge",
            "about": gettext(
                "Directory containing maps of monthly precipitation for each "
                "month. Only .tif files should be in this folder (no .tfw, "
                ".xml, etc files)."),
            "name": gettext("precipitation directory")
        },
        "dem_raster_path": {
            **spec_utils.DEM,
            "projected": True
        },
        "lulc_raster_path": {
            **spec_utils.LULC,
            "projected": True,
            "about": gettext(
                f"{spec_utils.LULC['about']} All values in this raster MUST "
                "have corresponding entries in the Biophysical Table.")
        },
        "soil_group_path": {
            **spec_utils.SOIL_GROUP,
            "projected": True,
            "required": "not user_defined_local_recharge"
        },
        "aoi_path": {
            **spec_utils.AOI,
            "projected": True
        },
        "biophysical_table_path": {
            "type": "csv",
            "columns": {
                "lucode": {
                    "type": "integer",
                    "about": gettext("LULC code matching those in the LULC raster.")},
                "cn_[SOIL_GROUP]": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "Curve number values for each combination of soil "
                        "group and LULC class. Replace [SOIL_GROUP] with each "
                        "soil group code A, B, C, D so that there is one "
                        "column for each soil group. Curve number values must "
                        "be greater than 0.")
                },
                "kc_[MONTH]": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "Crop/vegetation coefficient (Kc) values for this "
                        "LULC class in each month. Replace [MONTH] with the "
                        "numbers 1 to 12 so that there is one column for each "
                        "month.")
                }
            },
            "about": gettext(
                "A table mapping each LULC code to biophysical properties of "
                "the corresponding LULC class. All values in the LULC raster "
                "must have corresponding entries in this table."),
            "name": gettext("biophysical table")
        },
        "rain_events_table_path": {
            "type": "csv",
            "columns": {
                "month": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "Values are the numbers 1-12 corresponding to each "
                        "month, January (1) through December (12).")
                },
                "events": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("The number of rain events in that month.")
                }
            },
            "required": (
                "(not user_defined_local_recharge) & (not "
                "user_defined_climate_zones)"),
            "about": gettext(
                "A table containing the number of rain events for each month. "
                "Required if neither User-Defined Local Recharge nor User-"
                "Defined Climate Zones is selected."),
            "name": gettext("rain events table")
        },
        "alpha_m": {
            "type": "freestyle_string",
            "required": "not monthly_alpha",
            "about": gettext(
                "The proportion of upslope annual available local recharge "
                "that is available in each month. Required if Use Monthly "
                "Alpha Table is not selected."),
            "name": gettext("alpha_m parameter")
        },
        "beta_i": {
            "type": "ratio",
            "about": gettext(
                "The proportion of the upgradient subsidy that is available "
                "for downgradient evapotranspiration."),
            "name": gettext("beta_i parameter")
        },
        "gamma": {
            "type": "ratio",
            "about": gettext(
                "The proportion of pixel local recharge that is available to "
                "downgradient pixels."),
            "name": gettext("gamma parameter")
        },
        "user_defined_local_recharge": {
            "type": "boolean",
            "about": gettext(
                "Use user-defined local recharge data instead of calculating "
                "local recharge from the other provided data."),
            "name": gettext("user-defined recharge layer (advanced)")
        },
        "l_path": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": u.millimeter
            }},
            "required": "user_defined_local_recharge",
            "projected": True,
            "about": gettext(
                "Map of local recharge data. Required if User-Defined Local "
                "Recharge is selected."),
            "name": gettext("local recharge")
        },
        "user_defined_climate_zones": {
            "type": "boolean",
            "about": gettext(
                "Use user-defined climate zone data in lieu of a global rain "
                "events table."),
            "name": gettext("climate zones (advanced)")
        },
        "climate_zone_table_path": {
            "type": "csv",
            "columns": {
                "cz_id": {
                    "type": "integer",
                    "about": gettext(
                        "Climate zone ID numbers, corresponding to the values "
                        "in the Climate Zones map.")},
                "[MONTH]": {  # jan, feb, mar, etc.
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "The number of rain events that occur in each month "
                        "in this climate zone. Replace [MONTH] with the month "
                        "abbreviations: jan, feb, mar, apr, may, jun, jul, "
                        "aug, sep, oct, nov, dec, so that there is a column "
                        "for each month.")}
            },
            "required": "user_defined_climate_zones",
            "about": gettext(
                "Table of monthly precipitation events for each climate zone. "
                "Required if User-Defined Climate Zones is selected."),
            "name": gettext("climate zone table")
        },
        "climate_zone_raster_path": {
            "type": "raster",
            "bands": {1: {"type": "integer"}},
            "required": "user_defined_climate_zones",
            "projected": True,
            "about": gettext(
                "Map of climate zones. All values in this raster must have "
                "corresponding entries in the Climate Zone Table."),
            "name": gettext("climate zone map")
        },
        "monthly_alpha": {
            "type": "boolean",
            "about": gettext(
                "Use montly alpha values instead of a single value for the "
                "whole year."),
            "name": gettext("use monthly alpha table (advanced)")
        },
        "monthly_alpha_path": {
            "type": "csv",
            "columns": {
                "month": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext(
                        "Values are the numbers 1-12 corresponding to each "
                        "month.")
                },
                "alpha": {
                    "type": "number",
                    "units": u.none,
                    "about": gettext("The alpha value for that month.")
                }
            },
            "required": "monthly_alpha",
            "about": gettext(
                "Table of alpha values for each month. "
                "Required if Use Monthly Alpha Table is selected."),
            "name": gettext("monthly alpha table")
        }
    }
}


_OUTPUT_BASE_FILES = {
    'aggregate_vector_path': 'aggregated_results_swy.shp',
    'annual_precip_path': 'P.tif',
    'cn_path': 'CN.tif',
    'l_avail_path': 'L_avail.tif',
    'l_path': 'L.tif',
    'l_sum_path': 'L_sum.tif',
    'l_sum_avail_path': 'L_sum_avail.tif',
    'qf_path': 'QF.tif',
    'b_sum_path': 'B_sum.tif',
    'b_path': 'B.tif',
    'vri_path': 'Vri.tif',
}

_INTERMEDIATE_BASE_FILES = {
    'aet_path': 'aet.tif',
    'aetm_path_list': ['aetm_%d.tif' % (x+1) for x in range(N_MONTHS)],
    'flow_dir_mfd_path': 'flow_dir_mfd.tif',
    'qfm_path_list': ['qf_%d.tif' % (x+1) for x in range(N_MONTHS)],
    'stream_path': 'stream.tif',
}

_TMP_BASE_FILES = {
    'outflow_direction_path': 'outflow_direction.tif',
    'outflow_weights_path': 'outflow_weights.tif',
    'kc_path': 'kc.tif',
    'si_path': 'Si.tif',
    'lulc_aligned_path': 'lulc_aligned.tif',
    'dem_aligned_path': 'dem_aligned.tif',
    'dem_pit_filled_path': 'pit_filled_dem.tif',
    'loss_path': 'loss.tif',
    'zero_absorption_source_path': 'zero_absorption.tif',
    'soil_group_aligned_path': 'soil_group_aligned.tif',
    'flow_accum_path': 'flow_accum.tif',
    'precip_path_aligned_list': ['prcp_a%d.tif' % x for x in range(N_MONTHS)],
    'n_events_path_list': ['n_events%d.tif' % x for x in range(N_MONTHS)],
    'et0_path_aligned_list': ['et0_a%d.tif' % x for x in range(N_MONTHS)],
    'kc_path_list': ['kc_%d.tif' % x for x in range(N_MONTHS)],
    'l_aligned_path': 'l_aligned.tif',
    'cz_aligned_raster_path': 'cz_aligned.tif',
    'l_sum_pre_clamp': 'l_sum_pre_clamp.tif'
}


def execute(args):
    """Seasonal Water Yield.

    Args:
        args['workspace_dir'] (string): output directory for intermediate,
            temporary, and final files
        args['results_suffix'] (string): (optional) string to append to any
            output files
        args['threshold_flow_accumulation'] (number): used when classifying
            stream pixels from the DEM by thresholding the number of upslope
            cells that must flow into a cell before it's considered
            part of a stream.
        args['et0_dir'] (string): required if
            args['user_defined_local_recharge'] is False.  Path to a directory
            that contains rasters of monthly reference evapotranspiration;
            units in mm.
        args['precip_dir'] (string): required if
            args['user_defined_local_recharge'] is False. A path to a directory
            that contains rasters of monthly precipitation; units in mm.
        args['dem_raster_path'] (string): a path to a digital elevation raster
        args['lulc_raster_path'] (string): a path to a land cover raster used
            to classify biophysical properties of pixels.
        args['soil_group_path'] (string): required if
            args['user_defined_local_recharge'] is  False. A path to a raster
            indicating SCS soil groups where integer values are mapped to soil
            types
        args['aoi_path'] (string): path to a vector that indicates the area
            over which the model should be run, as well as the area in which to
            aggregate over when calculating the output Qb.
        args['biophysical_table_path'] (string): path to a CSV table that maps
            landcover codes paired with soil group types to curve numbers as
            well as Kc values.  Headers must include 'lucode', 'CN_A', 'CN_B',
            'CN_C', 'CN_D', 'Kc_1', 'Kc_2', 'Kc_3', 'Kc_4', 'Kc_5', 'Kc_6',
            'Kc_7', 'Kc_8', 'Kc_9', 'Kc_10', 'Kc_11', 'Kc_12'.
        args['rain_events_table_path'] (string): Not required if
            args['user_defined_local_recharge'] is True or
            args['user_defined_climate_zones'] is True.  Path to a CSV table
            that has headers 'month' (1-12) and 'events' (int >= 0) that
            indicates the number of rain events per month
        args['alpha_m'] (float or string): required if args['monthly_alpha'] is
            false.  Is the proportion of upslope annual available local
            recharge that is available in month m.
        args['beta_i'] (float or string): is the fraction of the upgradient
            subsidy that is available for downgradient evapotranspiration.
        args['gamma'] (float or string): is the fraction of pixel local
            recharge that is available to downgradient pixels.
        args['user_defined_local_recharge'] (boolean): if True, indicates user
            will provide pre-defined local recharge raster layer
        args['l_path'] (string): required if
            args['user_defined_local_recharge'] is True.  If provided pixels
            indicate the amount of local recharge; units in mm.
        args['user_defined_climate_zones'] (boolean): if True, user provides
            a climate zone rain events table and a climate zone raster map in
            lieu of a global rain events table.
        args['climate_zone_table_path'] (string): required if
            args['user_defined_climate_zones'] is True. Contains monthly
            precipitation events per climate zone.  Fields must be
            "cz_id", "jan", "feb", "mar", "apr", "may", "jun", "jul",
            "aug", "sep", "oct", "nov", "dec".
        args['climate_zone_raster_path'] (string): required if
            args['user_defined_climate_zones'] is True, pixel values correspond
            to the "cz_id" values defined in args['climate_zone_table_path']
        args['monthly_alpha'] (boolean): if True, use the alpha
        args['monthly_alpha_path'] (string): required if args['monthly_alpha']
            is True. A CSV file.
        args['n_workers'] (int): (optional) indicates the number of processes
            to devote to potential parallel task execution. A value < 0 will
            use a single process, 0 will be non-blocking scheduling but
            single process, and >= 1 will make additional processes for
            parallel execution.

    Returns:
        None.
    """
    # This upgrades warnings to exceptions across this model.
    # I found this useful to catch all kinds of weird inputs to the model
    # during debugging and think it makes sense to have in production of this
    # model too.
    try:
        warnings.filterwarnings('error')
        _execute(args)
    finally:
        warnings.resetwarnings()


def _execute(args):
    """Execute the seasonal water yield model.

    Args:
        See the parameters for
        `natcap.invest.seasonal_water_yield.seasonal_wateryield.execute`.

    Returns:
        None
    """
    LOGGER.info('prepare and test inputs for common errors')

    # fail early on a missing required rain events table
    if (not args['user_defined_local_recharge'] and
            not args['user_defined_climate_zones']):
        rain_events_lookup = (
            utils.build_lookup_from_csv(
                args['rain_events_table_path'], 'month'))

    biophysical_table = utils.build_lookup_from_csv(
        args['biophysical_table_path'], 'lucode')

    bad_value_list = []
    for lucode, value in biophysical_table.items():
        for biophysical_id in ['cn_a', 'cn_b', 'cn_c', 'cn_d'] + [
                'kc_%d' % (month_index+1) for month_index in range(N_MONTHS)]:
            try:
                _ = float(value[biophysical_id])
            except ValueError:
                bad_value_list.append(
                    (biophysical_id, lucode, value[biophysical_id]))

    if bad_value_list:
        raise ValueError(
            'biophysical_table at %s seems to have the following incorrect '
            'values (expecting all floating point numbers): %s' % (
                args['biophysical_table_path'], ','.join(
                    ['%s(lucode %d): "%s"' % (
                        lucode, biophysical_id, bad_value)
                     for lucode, biophysical_id, bad_value in
                        bad_value_list])))

    if args['monthly_alpha']:
        # parse out the alpha lookup table of the form (month_id: alpha_val)
        alpha_month_map = dict(
            (key, val['alpha']) for key, val in
            utils.build_lookup_from_csv(
                args['monthly_alpha_path'], 'month').items())
    else:
        # make all 12 entries equal to args['alpha_m']
        alpha_m = float(fractions.Fraction(args['alpha_m']))
        alpha_month_map = dict(
            (month_index+1, alpha_m) for month_index in range(N_MONTHS))

    beta_i = float(fractions.Fraction(args['beta_i']))
    gamma = float(fractions.Fraction(args['gamma']))
    threshold_flow_accumulation = float(args['threshold_flow_accumulation'])
    pixel_size = pygeoprocessing.get_raster_info(
        args['dem_raster_path'])['pixel_size']
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    cache_dir = os.path.join(args['workspace_dir'], 'cache_dir')
    output_dir = args['workspace_dir']
    utils.make_directories([intermediate_output_dir, cache_dir, output_dir])

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(
        cache_dir, n_workers, reporting_interval=5.0)

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, cache_dir)], file_suffix)

    LOGGER.info('Checking that the AOI is not the output aggregate vector')
    if (os.path.normpath(args['aoi_path']) ==
            os.path.normpath(file_registry['aggregate_vector_path'])):
        raise ValueError(
            "The input AOI is the same as the output aggregate vector, "
            "please choose a different workspace or move the AOI file "
            "out of the current workspace %s" %
            file_registry['aggregate_vector_path'])

    LOGGER.info('Aligning and clipping dataset list')
    input_align_list = [args['lulc_raster_path'], args['dem_raster_path']]
    output_align_list = [
        file_registry['lulc_aligned_path'], file_registry['dem_aligned_path']]
    if not args['user_defined_local_recharge']:
        precip_path_list = []
        et0_path_list = []

        et0_dir_list = [
            os.path.join(args['et0_dir'], f) for f in os.listdir(
                args['et0_dir'])]
        precip_dir_list = [
            os.path.join(args['precip_dir'], f) for f in os.listdir(
                args['precip_dir'])]

        for month_index in range(1, N_MONTHS + 1):
            month_file_match = re.compile(r'.*[^\d]%d\.[^.]+$' % month_index)

            for data_type, dir_list, path_list in [
                    ('et0', et0_dir_list, et0_path_list),
                    ('Precip', precip_dir_list, precip_path_list)]:
                file_list = [
                    month_file_path for month_file_path in dir_list
                    if month_file_match.match(month_file_path)]
                if len(file_list) == 0:
                    raise ValueError(
                        "No %s found for month %d" % (data_type, month_index))
                if len(file_list) > 1:
                    raise ValueError(
                        "Ambiguous set of files found for month %d: %s" %
                        (month_index, file_list))
                path_list.append(file_list[0])

        input_align_list = (
            precip_path_list + [args['soil_group_path']] + et0_path_list +
            input_align_list)
        output_align_list = (
            file_registry['precip_path_aligned_list'] +
            [file_registry['soil_group_aligned_path']] +
            file_registry['et0_path_aligned_list'] + output_align_list)

    align_index = len(input_align_list) - 1  # this aligns with the DEM
    if args['user_defined_local_recharge']:
        input_align_list.append(args['l_path'])
        output_align_list.append(file_registry['l_aligned_path'])
    elif args['user_defined_climate_zones']:
        input_align_list.append(args['climate_zone_raster_path'])
        output_align_list.append(
            file_registry['cz_aligned_raster_path'])
    interpolate_list = ['near'] * len(input_align_list)

    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            input_align_list, output_align_list, interpolate_list,
            pixel_size, 'intersection'),
        kwargs={
            'base_vector_path_list': (args['aoi_path'],),
            'raster_align_index': align_index},
        target_path_list=output_align_list,
        task_name='align rasters')

    fill_pit_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (file_registry['dem_aligned_path'], 1),
            file_registry['dem_pit_filled_path']),
        kwargs={'working_dir': cache_dir},
        target_path_list=[file_registry['dem_pit_filled_path']],
        dependent_task_list=[align_task],
        task_name='fill dem pits')

    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=(
            (file_registry['dem_pit_filled_path'], 1),
            file_registry['flow_dir_mfd_path']),
        kwargs={'working_dir': cache_dir},
        target_path_list=[file_registry['flow_dir_mfd_path']],
        dependent_task_list=[fill_pit_task],
        task_name='flow dir mfd')

    flow_accum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=(
            (file_registry['flow_dir_mfd_path'], 1),
            file_registry['flow_accum_path']),
        target_path_list=[file_registry['flow_accum_path']],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum task')

    stream_threshold_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_streams_mfd,
        args=(
            (file_registry['flow_accum_path'], 1),
            (file_registry['flow_dir_mfd_path'], 1),
            threshold_flow_accumulation,
            file_registry['stream_path']),
        target_path_list=[file_registry['stream_path']],
        dependent_task_list=[flow_accum_task],
        task_name='stream threshold')

    LOGGER.info('quick flow')
    if args['user_defined_local_recharge']:
        file_registry['l_path'] = file_registry['l_aligned_path']

        l_avail_task = task_graph.add_task(
            func=_calculate_l_avail,
            args=(
                file_registry['l_path'], gamma,
                file_registry['l_avail_path']),
            target_path_list=[file_registry['l_avail_path']],
            dependent_task_list=[align_task],
            task_name='l avail task')
    else:
        # user didn't predefine local recharge so calculate it
        LOGGER.info('loading number of monthly events')
        reclassify_n_events_task_list = []
        reclass_error_details = {
            'raster_name': 'Climate Zone', 'column_name': 'cz_id',
            'table_name': 'Climate Zone'}
        for month_id in range(N_MONTHS):
            if args['user_defined_climate_zones']:
                cz_rain_events_lookup = (
                    utils.build_lookup_from_csv(
                        args['climate_zone_table_path'], 'cz_id'))
                month_label = MONTH_ID_TO_LABEL[month_id]
                climate_zone_rain_events_month = dict([
                    (cz_id, cz_rain_events_lookup[cz_id][month_label]) for
                    cz_id in cz_rain_events_lookup])
                n_events_nodata = -1
                n_events_task = task_graph.add_task(
                    func=utils.reclassify_raster,
                    args=(
                        (file_registry['cz_aligned_raster_path'], 1),
                        climate_zone_rain_events_month,
                        file_registry['n_events_path_list'][month_id],
                        gdal.GDT_Float32, n_events_nodata,
                        reclass_error_details),
                    target_path_list=[
                        file_registry['n_events_path_list'][month_id]],
                    dependent_task_list=[align_task],
                    task_name='n_events for month %d' % month_id)
                reclassify_n_events_task_list.append(n_events_task)
            else:
                # rain_events_lookup defined near entry point of execute
                n_events = rain_events_lookup[month_id+1]['events']
                n_events_task = task_graph.add_task(
                    func=pygeoprocessing.new_raster_from_base,
                    args=(
                        file_registry['dem_aligned_path'],
                        file_registry['n_events_path_list'][month_id],
                        gdal.GDT_Float32, [TARGET_NODATA]),
                    kwargs={'fill_value_list': (n_events,)},
                    target_path_list=[
                        file_registry['n_events_path_list'][month_id]],
                    dependent_task_list=[align_task],
                    task_name=(
                        'n_events as a constant raster month %d' % month_id))
                reclassify_n_events_task_list.append(n_events_task)

        curve_number_task = task_graph.add_task(
            func=_calculate_curve_number_raster,
            args=(
                file_registry['lulc_aligned_path'],
                file_registry['soil_group_aligned_path'],
                biophysical_table, file_registry['cn_path']),
            target_path_list=[file_registry['cn_path']],
            dependent_task_list=[align_task],
            task_name='calculate curve number')

        si_task = task_graph.add_task(
            func=_calculate_si_raster,
            args=(
                file_registry['cn_path'], file_registry['stream_path'],
                file_registry['si_path']),
            target_path_list=[file_registry['si_path']],
            dependent_task_list=[curve_number_task, stream_threshold_task],
            task_name='calculate Si raster')

        quick_flow_task_list = []
        for month_index in range(N_MONTHS):
            LOGGER.info('calculate quick flow for month %d', month_index+1)
            monthly_quick_flow_task = task_graph.add_task(
                func=_calculate_monthly_quick_flow,
                args=(
                    file_registry['precip_path_aligned_list'][month_index],
                    file_registry['lulc_aligned_path'],
                    file_registry['cn_path'],
                    file_registry['n_events_path_list'][month_index],
                    file_registry['stream_path'],
                    file_registry['si_path'],
                    file_registry['qfm_path_list'][month_index]),
                target_path_list=[
                    file_registry['qfm_path_list'][month_index]],
                dependent_task_list=[
                    align_task, reclassify_n_events_task_list[month_index],
                    si_task, stream_threshold_task],
                task_name='calculate quick flow for month %d' % (
                    month_index+1))
            quick_flow_task_list.append(monthly_quick_flow_task)

        qf_task = task_graph.add_task(
            func=_calculate_annual_qfi,
            args=(file_registry['qfm_path_list'], file_registry['qf_path']),
            target_path_list=[file_registry['qf_path']],
            dependent_task_list=quick_flow_task_list,
            task_name='calculate QFi')

        LOGGER.info('calculate local recharge')
        kc_task_list = []
        reclass_error_details = {
            'raster_name': 'LULC', 'column_name': 'lucode',
            'table_name': 'Biophysical'}
        for month_index in range(N_MONTHS):
            kc_lookup = dict([
                (lucode, biophysical_table[lucode]['kc_%d' % (month_index+1)])
                for lucode in biophysical_table])
            kc_nodata = -1  # a reasonable nodata value
            kc_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=(
                    (file_registry['lulc_aligned_path'], 1), kc_lookup,
                    file_registry['kc_path_list'][month_index],
                    gdal.GDT_Float32, kc_nodata, reclass_error_details),
                target_path_list=[file_registry['kc_path_list'][month_index]],
                dependent_task_list=[align_task],
                task_name='classify kc month %d' % month_index)
            kc_task_list.append(kc_task)

        # call through to a cython function that does the necessary routing
        # between AET and L.sum.avail in equation [7], [4], and [3]
        calculate_local_recharge_task = task_graph.add_task(
            func=seasonal_water_yield_core.calculate_local_recharge,
            args=(
                file_registry['precip_path_aligned_list'],
                file_registry['et0_path_aligned_list'],
                file_registry['qfm_path_list'],
                file_registry['flow_dir_mfd_path'],
                file_registry['kc_path_list'],
                alpha_month_map,
                beta_i, gamma, file_registry['stream_path'],
                file_registry['l_path'],
                file_registry['l_avail_path'],
                file_registry['l_sum_avail_path'],
                file_registry['aet_path'],
                file_registry['annual_precip_path']),
            target_path_list=[
                file_registry['l_path'],
                file_registry['l_avail_path'],
                file_registry['l_sum_avail_path'],
                file_registry['aet_path'],
                file_registry['annual_precip_path'],
            ],
            dependent_task_list=[
                align_task, flow_dir_task, stream_threshold_task,
                fill_pit_task, qf_task] + quick_flow_task_list,
            task_name='calculate local recharge')

    # calculate Qb as the sum of local_recharge_avail over the AOI, Eq [9]
    if args['user_defined_local_recharge']:
        vri_dependent_task_list = [l_avail_task]
    else:
        vri_dependent_task_list = [calculate_local_recharge_task]

    vri_task = task_graph.add_task(
        func=_calculate_vri,
        args=(file_registry['l_path'], file_registry['vri_path']),
        target_path_list=[file_registry['vri_path']],
        dependent_task_list=vri_dependent_task_list,
        task_name='calculate vri')

    aggregate_recharge_task = task_graph.add_task(
        func=_aggregate_recharge,
        args=(
            args['aoi_path'], file_registry['l_path'],
            file_registry['vri_path'],
            file_registry['aggregate_vector_path']),
        target_path_list=[file_registry['aggregate_vector_path']],
        dependent_task_list=[vri_task],
        task_name='aggregate recharge')

    LOGGER.info('calculate L_sum')  # Eq. [12]
    l_sum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=(
            (file_registry['flow_dir_mfd_path'], 1),
            file_registry['l_sum_path']),
        kwargs={'weight_raster_path_band': (file_registry['l_path'], 1)},
        target_path_list=[file_registry['l_sum_path']],
        dependent_task_list=vri_dependent_task_list + [
            fill_pit_task, flow_dir_task, stream_threshold_task],
        task_name='calculate l sum')

    if args['user_defined_local_recharge']:
        b_sum_dependent_task_list = [l_avail_task]
    else:
        b_sum_dependent_task_list = [calculate_local_recharge_task]

    b_sum_task = task_graph.add_task(
        func=seasonal_water_yield_core.route_baseflow_sum,
        args=(
            file_registry['flow_dir_mfd_path'],
            file_registry['l_path'],
            file_registry['l_avail_path'],
            file_registry['l_sum_path'],
            file_registry['stream_path'],
            file_registry['b_path'],
            file_registry['b_sum_path']),

        target_path_list=[
            file_registry['b_sum_path'], file_registry['b_path']],
        dependent_task_list=b_sum_dependent_task_list + [l_sum_task],
        task_name='calculate B_sum')

    task_graph.close()
    task_graph.join()

    LOGGER.info('  (\\w/)  SWY Complete!')
    LOGGER.info('  (..  \\ ')
    LOGGER.info(' _/  )  \\______')
    LOGGER.info('(oo /\'\\        )`,')
    LOGGER.info(' `--\' (v  __( / ||')
    LOGGER.info('       |||  ||| ||')
    LOGGER.info('      //_| //_|')


def _calculate_vri(l_path, target_vri_path):
    """Calculate VRI as li_array / qb_sum.

    Args:
        l_path (str): path to L raster.
        target_vri_path (str): path to output Vri raster.

    Returns:
        None.

    """
    qb_sum = 0.0
    qb_valid_count = 0
    l_nodata = pygeoprocessing.get_raster_info(l_path)['nodata'][0]

    for _, block in pygeoprocessing.iterblocks((l_path, 1)):
        valid_mask = (
            ~utils.array_equals_nodata(block, l_nodata) &
            (~numpy.isinf(block)))
        qb_sum += numpy.sum(block[valid_mask])
        qb_valid_count += numpy.count_nonzero(valid_mask)
    li_nodata = pygeoprocessing.get_raster_info(l_path)['nodata'][0]

    def vri_op(li_array):
        """Calculate vri index [Eq 10]."""
        result = numpy.empty_like(li_array)
        result[:] = li_nodata
        if qb_sum > 0:
            valid_mask = ~utils.array_equals_nodata(li_array, li_nodata)
            try:
                result[valid_mask] = li_array[valid_mask] / qb_sum
            except RuntimeWarning:
                LOGGER.exception(qb_sum)
                raise
        return result
    pygeoprocessing.raster_calculator(
        [(l_path, 1)], vri_op, target_vri_path, gdal.GDT_Float32,
        li_nodata)


def _calculate_annual_qfi(qfm_path_list, target_qf_path):
    """Calculate annual quickflow.

    Args:
        qfm_path_list (list): list of monthly quickflow raster paths.
        target_qf_path (str): path to target annual quickflow raster.

    Returns:
        None.

    """
    qf_nodata = -1

    def qfi_sum_op(*qf_values):
        """Sum the monthly qfis."""
        qf_sum = numpy.zeros(qf_values[0].shape)
        valid_mask = ~utils.array_equals_nodata(qf_values[0], qf_nodata)
        valid_qf_sum = qf_sum[valid_mask]
        for index in range(len(qf_values)):
            valid_qf_sum += qf_values[index][valid_mask]
        qf_sum[:] = qf_nodata
        qf_sum[valid_mask] = valid_qf_sum
        return qf_sum

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in qfm_path_list],
        qfi_sum_op, target_qf_path, gdal.GDT_Float32, qf_nodata)


def _calculate_monthly_quick_flow(
        precip_path, lulc_raster_path, cn_path, n_events_raster_path,
        stream_path, si_path, qf_monthly_path):
    """Calculate quick flow for a month.

    Args:
        precip_path (string): path to file that correspond to monthly
            precipitation
        lulc_raster_path (string): path to landcover raster
        cn_path (string): path to curve number raster
        n_events_raster_path (string): a path to a raster where each pixel
            indicates the number of rain events.
        stream_path (string): path to stream mask raster where 1 indicates a
            stream pixel, 0 is a non-stream but otherwise valid area from the
            original DEM, and nodata indicates areas outside the valid DEM.
        si_path (string): path to raster that has potential maximum retention
        qf_monthly_path_list (list of string): list of paths to output monthly
            rasters.

    Returns:
        None
    """
    si_nodata = pygeoprocessing.get_raster_info(si_path)['nodata'][0]

    qf_nodata = -1
    p_nodata = pygeoprocessing.get_raster_info(precip_path)['nodata'][0]
    n_events_nodata = pygeoprocessing.get_raster_info(
        n_events_raster_path)['nodata'][0]
    stream_nodata = pygeoprocessing.get_raster_info(stream_path)['nodata'][0]

    def qf_op(p_im, s_i, n_events, stream_array):
        """Calculate quick flow as in Eq [1] in user's guide.

        Args:
            p_im (numpy.array): precipitation at pixel i on month m
            s_i (numpy.array): factor that is 1000/CN_i - 10
                (Equation 1b from user's guide)
            n_events (numpy.array): number of rain events on the pixel
            stream_mask (numpy.array): 1 if stream, otherwise not a stream
                pixel.

        Returns:
            quick flow (numpy.array)

        """
        # s_i is an intermediate output which will always have a defined
        # nodata value
        valid_mask = ((p_im != 0.0) &
                      (stream_array != 1) &
                      (n_events > 0) &
                      ~utils.array_equals_nodata(s_i, si_nodata))
        if p_nodata is not None:
            valid_mask &= ~utils.array_equals_nodata(p_im, p_nodata)
        if n_events_nodata is not None:
            valid_mask &= ~utils.array_equals_nodata(n_events, n_events_nodata)
        # stream_nodata is the only input that carry over nodata values from
        # the aligned DEM.
        if stream_nodata is not None:
            valid_mask &= ~utils.array_equals_nodata(
                stream_array, stream_nodata)

        valid_n_events = n_events[valid_mask]
        valid_si = s_i[valid_mask]

        # a_im is the mean rain depth on a rainy day at pixel i on month m
        # the 25.4 converts inches to mm since Si is in inches
        a_im = numpy.empty(valid_n_events.shape)
        a_im = p_im[valid_mask] / (valid_n_events * 25.4)
        qf_im = numpy.empty(p_im.shape)
        qf_im[:] = qf_nodata

        # Precompute the last two terms in quickflow so we can handle a
        # numerical instability when s_i is large and/or a_im is small
        # on large valid_si/a_im this number will be zero and the latter
        # exponent will also be zero because of a divide by zero. rather than
        # raise that numerical warning, just handle it manually
        E1 = scipy.special.expn(1, valid_si / a_im)
        E1[valid_si == 0] = 0
        nonzero_e1_mask = E1 != 0
        exp_result = numpy.zeros(valid_si.shape)
        exp_result[nonzero_e1_mask] = numpy.exp(
            (0.8 * valid_si[nonzero_e1_mask]) / a_im[nonzero_e1_mask] +
            numpy.log(E1[nonzero_e1_mask]))

        # qf_im is the quickflow at pixel i on month m Eq. [1]
        qf_im[valid_mask] = (25.4 * valid_n_events * (
            (a_im - valid_si) * numpy.exp(-0.2 * valid_si / a_im) +
            valid_si ** 2 / a_im * exp_result))

        # if precip is 0, then QF should be zero
        qf_im[(p_im == 0) | (n_events == 0)] = 0.0
        # if we're on a stream, set quickflow to the precipitation
        valid_stream_precip_mask = stream_array == 1
        if p_nodata is not None:
            valid_stream_precip_mask &= ~utils.array_equals_nodata(
                p_im, p_nodata)
        qf_im[valid_stream_precip_mask] = p_im[valid_stream_precip_mask]

        # this handles some user cases where they don't have data defined on
        # their landcover raster. It otherwise crashes later with some NaNs.
        # more intermediate outputs with nodata values guaranteed to be defined
        qf_im[utils.array_equals_nodata(qf_im, qf_nodata) &
              ~utils.array_equals_nodata(stream_array, stream_nodata)] = 0.0
        return qf_im

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in [
            precip_path, si_path, n_events_raster_path, stream_path]], qf_op,
        qf_monthly_path, gdal.GDT_Float32, qf_nodata)


def _calculate_curve_number_raster(
        lulc_raster_path, soil_group_path, biophysical_table, cn_path):
    """Calculate the CN raster from the landcover and soil group rasters.

    Args:
        lulc_raster_path (string): path to landcover raster
        soil_group_path (string): path to raster indicating soil group where
            pixel values are in [1,2,3,4]
        biophysical_table (dict): maps landcover IDs to dictionaries that
            contain at least the keys 'cn_a', 'cn_b', 'cn_c', 'cn_d', that
            map to the curve numbers for that landcover and soil type.
        cn_path (string): path to output curve number raster to be output
            which will be the dimensions of the intersection of
            `lulc_raster_path` and `soil_group_path` the cell size of
            `lulc_raster_path`.

    Returns:
        None
    """
    soil_nodata = pygeoprocessing.get_raster_info(
        soil_group_path)['nodata'][0]
    map_soil_type_to_header = {
        1: 'cn_a',
        2: 'cn_b',
        3: 'cn_c',
        4: 'cn_d',
    }
    # curve numbers are always positive so -1 a good nodata choice
    cn_nodata = -1
    lulc_to_soil = {}
    lulc_nodata = pygeoprocessing.get_raster_info(
        lulc_raster_path)['nodata'][0]

    lucodes = list(biophysical_table)
    if lulc_nodata is not None:
        lucodes.append(lulc_nodata)

    for soil_id, soil_column in map_soil_type_to_header.items():
        lulc_to_soil[soil_id] = {
            'lulc_values': [],
            'cn_values': []
        }

        for lucode in sorted(lucodes):
            if lucode != lulc_nodata:
                lulc_to_soil[soil_id]['cn_values'].append(
                    biophysical_table[lucode][soil_column])
                lulc_to_soil[soil_id]['lulc_values'].append(lucode)
            else:
                # handle the lulc nodata with cn nodata
                lulc_to_soil[soil_id]['lulc_values'].append(lulc_nodata)
                lulc_to_soil[soil_id]['cn_values'].append(cn_nodata)

        # Making the landcover array a float32 in case the user provides a
        # float landcover map like Kate did.
        lulc_to_soil[soil_id]['lulc_values'] = (
            numpy.array(lulc_to_soil[soil_id]['lulc_values'],
                        dtype=numpy.float32))
        lulc_to_soil[soil_id]['cn_values'] = (
            numpy.array(lulc_to_soil[soil_id]['cn_values'],
                        dtype=numpy.float32))

    # Use set of table lucodes in cn_op
    lucodes_set = set(lucodes)

    def cn_op(lulc_array, soil_group_array):
        """Map lulc code and soil to a curve number."""
        cn_result = numpy.empty(lulc_array.shape)
        cn_result[:] = cn_nodata

        # if lulc_array value not in lulc_to_soil[soil_group_id]['lulc_values']
        # then numpy.digitize will not bin properly and cause an IndexError
        # during the reshaping call
        lulc_unique = set(numpy.unique(lulc_array))
        if not lulc_unique.issubset(lucodes_set):
            # cast to list to conform with similar error messages in InVEST
            missing_lulc_values = sorted(lulc_unique.difference(lucodes_set))
            error_message = (
                "Values in the LULC raster were found that are not"
                " represented under the 'lucode' key column of the"
                " Biophysical table. The missing values found in the LULC"
                f" raster but not the table are: {missing_lulc_values}.")
            raise ValueError(error_message)

        for soil_group_id in numpy.unique(soil_group_array):
            if soil_group_id == soil_nodata:
                continue
            current_soil_mask = (soil_group_array == soil_group_id)
            index = numpy.digitize(
                lulc_array.ravel(),
                lulc_to_soil[soil_group_id]['lulc_values'], right=True)
            cn_values = (
                lulc_to_soil[soil_group_id]['cn_values'][index]).reshape(
                    lulc_array.shape)
            cn_result[current_soil_mask] = cn_values[current_soil_mask]
        return cn_result

    cn_nodata = -1
    pygeoprocessing.raster_calculator(
        [(lulc_raster_path, 1), (soil_group_path, 1)], cn_op, cn_path,
        gdal.GDT_Float32, cn_nodata)


def _calculate_si_raster(cn_path, stream_path, si_path):
    """Calculate the S factor of the quickflow equation [1].

    Args:
        cn_path (string): path to curve number raster
        stream_path (string): path to a stream raster (0, 1)
        si_path (string): path to output s_i raster

    Returns:
        None
    """
    si_nodata = -1
    cn_nodata = pygeoprocessing.get_raster_info(cn_path)['nodata'][0]

    def si_op(ci_factor, stream_mask):
        """Calculate si factor."""
        valid_mask = (
            ~utils.array_equals_nodata(ci_factor, cn_nodata) &
            (ci_factor > 0))
        si_array = numpy.empty(ci_factor.shape)
        si_array[:] = si_nodata
        # multiply by the stream mask != 1 so we get 0s on the stream and
        # unaffected results everywhere else
        si_array[valid_mask] = (
            (1000.0 / ci_factor[valid_mask] - 10) * (
                stream_mask[valid_mask] != 1))
        return si_array

    pygeoprocessing.raster_calculator(
        [(cn_path, 1), (stream_path, 1)], si_op, si_path, gdal.GDT_Float32,
        si_nodata)


def _aggregate_recharge(
        aoi_path, l_path, vri_path, aggregate_vector_path):
    """Aggregate recharge values for the provided watersheds/AOIs.

    Generates a new shapefile that's a copy of 'aoi_path' in sum values from L
    and Vri.

    Args:
        aoi_path (string): path to shapefile that will be used to
            aggregate rasters
        l_path (string): path to (L) local recharge raster
        vri_path (string): path to Vri raster
        aggregate_vector_path (string): path to shapefile that will be created
            by this function as the aggregating output.  will contain fields
            'l_sum' and 'vri_sum' per original feature in `aoi_path`.  If this
            file exists on disk prior to the call it is overwritten with
            the result of this call.

    Returns:
        None
    """
    if os.path.exists(aggregate_vector_path):
        LOGGER.warning(
            '%s exists, deleting and writing new output',
            aggregate_vector_path)
        os.remove(aggregate_vector_path)

    original_aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)

    driver = gdal.GetDriverByName('ESRI Shapefile')
    driver.CreateCopy(aggregate_vector_path, original_aoi_vector)
    gdal.Dataset.__swig_destroy__(original_aoi_vector)
    original_aoi_vector = None
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, 1)
    aggregate_layer = aggregate_vector.GetLayer()

    for raster_path, aggregate_field_id, op_type in [
            (l_path, 'qb', 'mean'), (vri_path, 'vri_sum', 'sum')]:

        # aggregate carbon stocks by the new ID field
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregate_vector_path)

        aggregate_field = ogr.FieldDefn(aggregate_field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)

        aggregate_layer.ResetReading()
        for poly_index, poly_feat in enumerate(aggregate_layer):
            if op_type == 'mean':
                pixel_count = aggregate_stats[poly_index]['count']
                if pixel_count != 0:
                    value = (aggregate_stats[poly_index]['sum'] / pixel_count)
                else:
                    LOGGER.warning(
                        "no coverage for polygon %s", ', '.join(
                            [str(poly_feat.GetField(_)) for _ in range(
                                poly_feat.GetFieldCount())]))
                    value = 0.0
            elif op_type == 'sum':
                value = aggregate_stats[poly_index]['sum']
            poly_feat.SetField(aggregate_field_id, float(value))
            aggregate_layer.SetFeature(poly_feat)

    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


def _calculate_l_avail(l_path, gamma, target_l_avail_path):
    """l avail = l * gamma."""
    li_nodata = pygeoprocessing.get_raster_info(l_path)['nodata'][0]

    def l_avail_op(l_array):
        """Calculate equation [8] L_avail = min(gamma*L, L)."""
        result = numpy.empty(l_array.shape)
        result[:] = li_nodata
        valid_mask = ~utils.array_equals_nodata(l_array, li_nodata)
        result[valid_mask] = numpy.min(numpy.stack(
            (gamma*l_array[valid_mask], l_array[valid_mask])), axis=0)
        return result

    pygeoprocessing.raster_calculator(
        [(l_path, 1)], l_avail_op, target_l_avail_path, gdal.GDT_Float32,
        li_nodata)


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
    return validation.validate(args, ARGS_SPEC['args'],
                               ARGS_SPEC['args_with_spatial_overlap'])
