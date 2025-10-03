"""Urban Cooling Model."""
import copy
import logging
import math
import os
import pickle
import shutil
import tempfile
import time

import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import rtree
import shapely.prepared
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)
TARGET_NODATA = -1
_LOGGING_PERIOD = 5

MODEL_SPEC = spec.ModelSpec(
    model_id="urban_cooling_model",
    model_title=gettext("Urban Cooling"),
    userguide="urban_cooling_model.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("ucm",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_raster_path", "ref_eto_raster_path",
         "aoi_vector_path", "biophysical_table_path"],
        ["t_ref", "uhi_max", "t_air_average_radius",
         "green_area_cooling_distance", "cc_method"],
        ["do_energy_valuation", "building_vector_path", "energy_consumption_table_path"],
        ["do_productivity_valuation", "avg_rel_humidity"],
        ["cc_weight_shade", "cc_weight_albedo", "cc_weight_eti"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="lulc_raster_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of LULC for the area of interest. All values in this raster must"
                " have corresponding entries in the Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.SingleBandRasterInput(
            id="ref_eto_raster_path",
            name=gettext("reference evapotranspiration"),
            about=gettext("Map of reference evapotranspiration values."),
            data_type=float,
            units=u.millimeter
        ),
        spec.AOI.model_copy(update=dict(id="aoi_vector_path")),
        spec.CSVInput(
            id="biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "A table mapping each LULC code to biophysical data for that LULC class."
                " All values in the LULC raster must have corresponding entries in this"
                " table."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.NumberInput(
                    id="kc",
                    about=gettext("Crop coefficient for this LULC class."),
                    units=u.none
                ),
                spec.BooleanInput(
                    id="green_area",
                    about=gettext(
                        "Enter 1 to indicate that the LULC is considered a green area."
                        " Enter 0 to indicate that the LULC is not considered a green"
                        " area."
                    )
                ),
                spec.RatioInput(
                    id="shade",
                    about=(
                        "The proportion of area in this LULC class that is covered by"
                        " tree canopy at least 2 meters high. Required if the 'factors'"
                        " option is selected for the Cooling Capacity Calculation Method."
                    ),
                    required="cc_method == 'factors'",
                    units=None
                ),
                spec.RatioInput(
                    id="albedo",
                    about=(
                        "The proportion of solar radiation that is directly reflected by"
                        " this LULC class. Required if the 'factors' option is selected"
                        " for the Cooling Capacity Calculation Method."
                    ),
                    required="cc_method == 'factors'",
                    units=None
                ),
                spec.RatioInput(
                    id="building_intensity",
                    about=(
                        "The ratio of building floor area to footprint area, with all"
                        " values in this column normalized between 0 and 1. Required if"
                        " the 'intensity' option is selected for the Cooling Capacity"
                        " Calculation Method."
                    ),
                    required="cc_method == 'intensity'",
                    units=None
                )
            ],
            index_col="lucode"
        ),
        spec.NumberInput(
            id="green_area_cooling_distance",
            name=gettext("maximum cooling distance"),
            about=gettext(
                "Distance over which green areas larger than 2 hectares have a cooling"
                " effect."
            ),
            units=u.meter,
            expression="value >= 0"
        ),
        spec.NumberInput(
            id="t_air_average_radius",
            name=gettext("air blending distance"),
            about=gettext(
                "Radius over which to average air temperatures to account for air mixing."
            ),
            units=u.meter,
            expression="value >= 0"
        ),
        spec.NumberInput(
            id="t_ref",
            name=gettext("reference air temperature"),
            about=gettext(
                "Air temperature in a rural reference area where the urban heat island"
                " effect is not observed."
            ),
            units=u.degree_Celsius
        ),
        spec.NumberInput(
            id="uhi_max",
            name=gettext("UHI effect"),
            about=gettext(
                "The magnitude of the urban heat island effect, i.e., the difference"
                " between the rural reference temperature and the maximum temperature"
                " observed in the city. This model is designed for cases where"
                " UHI is positive, meaning the urban air temperature is greater"
                " than the rural reference temperature."
            ),
            units=u.degree_Celsius,
            expression="value >= 0"
        ),
        spec.BooleanInput(
            id="do_energy_valuation",
            name=gettext("run energy savings valuation"),
            about=gettext("Run the energy savings valuation model.")
        ),
        spec.BooleanInput(
            id="do_productivity_valuation",
            name=gettext("run work productivity valuation"),
            about=gettext("Run the work productivity valuation model.")
        ),
        spec.PercentInput(
            id="avg_rel_humidity",
            name=gettext("average relative humidity"),
            about=gettext(
                "The average relative humidity over the time period of interest. Required"
                " if Run Work Productivity Valuation is selected."
            ),
            required="do_productivity_valuation",
            allowed="do_productivity_valuation",
            units=None,
            expression="0 <= value <= 100"
        ),
        spec.VectorInput(
            id="building_vector_path",
            name=gettext("buildings"),
            about=gettext(
                "A map of built infrastructure footprints. Required if Run Energy Savings"
                " Valuation is selected."
            ),
            required="do_energy_valuation",
            allowed="do_energy_valuation",
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.IntegerInput(
                    id="type",
                    about=gettext(
                        "Code indicating the building type. These codes must match those"
                        " in the Energy Consumption Table."
                    )
                )
            ],
            projected=None
        ),
        spec.CSVInput(
            id="energy_consumption_table_path",
            name=gettext("energy consumption table"),
            about=gettext(
                "A table of energy consumption data for each building type. Required if"
                " Run Energy Savings Valuation is selected."
            ),
            required="do_energy_valuation",
            allowed="do_energy_valuation",
            columns=[
                spec.IntegerInput(
                    id="type",
                    about=gettext(
                        "Building type codes matching those in the Buildings vector."
                    )
                ),
                spec.NumberInput(
                    id="consumption",
                    about=gettext(
                        "Energy consumption by footprint area for this building type."
                    ),
                    units=u.kilowatt_hour / u.degree_Celsius / u.meter**2
                ),
                spec.NumberInput(
                    id="cost",
                    about=gettext(
                        "The cost of electricity for this building type. If this column"
                        " is provided, the energy savings outputs will be in the this"
                        " currency unit rather than kWh."
                    ),
                    required=False,
                    units=u.currency / u.kilowatt_hour
                )
            ],
            index_col="type"
        ),
        spec.OptionStringInput(
            id="cc_method",
            name=gettext("cooling capacity calculation method"),
            about=gettext("The air temperature predictor method to use."),
            options=[
                spec.Option(
                    key="factors",
                    about=(
                        "Use the weighted shade, albedo, and ETI factors as a temperature"
                        " predictor (for daytime temperatures).")),
                spec.Option(
                    key="intensity",
                    about=(
                        "Use building intensity as a temperature predictor (for nighttime"
                        " temperatures)."))
            ]
        ),
        spec.RatioInput(
            id="cc_weight_shade",
            name=gettext("shade weight"),
            about=gettext(
                "The relative weight to apply to shade when calculating the cooling"
                " capacity index. If not provided, defaults to 0.6."
            ),
            required=False,
            units=None
        ),
        spec.RatioInput(
            id="cc_weight_albedo",
            name=gettext("albedo weight"),
            about=gettext(
                "The relative weight to apply to albedo when calculating the cooling"
                " capacity index. If not provided, defaults to 0.2."
            ),
            required=False,
            units=None
        ),
        spec.RatioInput(
            id="cc_weight_eti",
            name=gettext("evapotranspiration weight"),
            about=gettext(
                "The relative weight to apply to ETI when calculating the cooling"
                " capacity index. If not provided, defaults to 0.2."
            ),
            required=False,
            units=None
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="hm",
            path="hm.tif",
            about=gettext("Map of heat mitigation index."),
            data_type=float,
            units=None
        ),
        spec.VectorOutput(
            id="uhi_results",
            path="uhi_results.shp",
            about=gettext(
                "A copy of the input Area of Interest vector with additional fields."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="avg_cc", about=gettext("Average CC value"), units=u.none
                ),
                spec.NumberOutput(
                    id="avg_tmp_v",
                    about=gettext("Average temperature value"),
                    units=u.degree_Celsius
                ),
                spec.NumberOutput(
                    id="avg_tmp_an",
                    about=gettext("Average temperature anomaly"),
                    units=u.degree_Celsius
                ),
                spec.NumberOutput(
                    id="avd_eng_cn",
                    about=gettext(
                        "Avoided energy consumption (kWh or $ if optional energy cost"
                        " input column was provided in the Energy Consumption Table)."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="avg_wbgt_v",
                    about=gettext("Average wet bulb globe temperature."),
                    units=u.degree_Celsius
                ),
                spec.PercentOutput(
                    id="avg_ltls_v", about=gettext("Average light work productivity loss")
                ),
                spec.PercentOutput(
                    id="avg_hvls_v", about=gettext("Average heavy work productivity loss")
                )
            ]
        ),
        spec.VectorOutput(
            id="buildings_with_stats",
            path="buildings_with_stats.shp",
            about=gettext(
                "A copy of the input vector “Building Footprints” with additional fields."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="energy_sav",
                    about=gettext(
                        "Energy savings value (kWh or currency if optional energy cost"
                        " input column was provided in the Energy Consumption Table)."
                        " Savings are relative to a theoretical scenario where the city"
                        " contains NO natural areas nor green spaces; where CC = 0 for"
                        " all LULC classes."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="mean_t_air",
                    about=gettext(
                        "Average temperature value in building. Calculated from the mean"
                        " T_air pixel value under this building geometry."
                    ),
                    units=u.degree_Celsius
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="cc",
            path="intermediate/cc.tif",
            about=gettext("Map of cooling capacity"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="cc_park",
            path="intermediate/cc_park.tif",
            about=gettext("Map of cooling capacity decayed by proximity to greenspace"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="cc_masked_green_areas",
            path="intermediate/cc_masked_green_areas.tif",
            about=gettext("Cooling capacity map masked by non-green areas"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="t_air",
            path="intermediate/T_air.tif",
            about=gettext("Map of air temperature with air mixing."),
            data_type=float,
            units=u.degree_Celsius
        ),
        spec.SingleBandRasterOutput(
            id="t_air_nomix",
            path="intermediate/T_air_nomix.tif",
            about=gettext("Map of air temperature without air mixing."),
            data_type=float,
            units=u.degree_Celsius
        ),
        spec.SingleBandRasterOutput(
            id="eti",
            path="intermediate/eti.tif",
            about=gettext("Map of the evapotranspiration index."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="wbgt",
            path="intermediate/wbgt.tif",
            about=gettext("Map of wet bulb globe temperature."),
            data_type=float,
            units=u.degree_Celsius
        ),
        spec.VectorOutput(
            id="reprojected_aoi",
            path="intermediate/reprojected_aoi.shp",
            about=gettext(
                "The Area of Interest vector reprojected to the spatial reference"
                " of the LULC."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.VectorOutput(
            id="reprojected_buildings",
            path="intermediate/reprojected_buildings.shp",
            about=gettext(
                "The buildings vector reprojected to the spatial reference of the"
                " LULC."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="albedo",
            path="intermediate/albedo.tif",
            about=gettext("Map of albedo."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="area_kernel",
            path="intermediate/area_kernel.tif",
            about=gettext("Area kernel for green area convolution."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="green_area_sum",
            path="intermediate/green_area_sum.tif",
            about=gettext(
                "Map of green area pixels within a search distance around each"
                " pixel."
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="kc",
            path="intermediate/kc.tif",
            about=gettext("Map of crop coefficient."),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="green_area",
            path="intermediate/green_area.tif",
            about=gettext("Map of green area."),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="building_intensity",
            path="intermediate/building_intensity.tif",
            about=gettext("Map of building intensity."),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="lulc",
            path="intermediate/lulc.tif",
            about=gettext("Map of land use/land cover."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="ref_eto",
            path="intermediate/ref_eto.tif",
            about=gettext(
                "Map of reference evapotranspiration reprojected and aligned to"
                " the intersection of the AOI, ET0, and LULC."
            ),
            data_type=float,
            units=u.millimeter
        ),
        spec.SingleBandRasterOutput(
            id="shade",
            path="intermediate/shade.tif",
            about=gettext("Map of shade."),
            data_type=float,
            units=None
        ),
        spec.FileOutput(
            id="cc_ref_aoi_stats",
            path="intermediate/cc_ref_aoi_stats.pickle",
            about=gettext("Cooling capacity zonal statistics for aoi.")
        ),
        spec.FileOutput(
            id="t_air_aoi_stats",
            path="intermediate/t_air_aoi_stats.pickle",
            about=gettext("Air temperature zonal statistics for aoi.")
        ),
        spec.SingleBandRasterOutput(
            id="light_work_loss_percent",
            path="intermediate/light_work_loss_percent.tif",
            about=gettext("Map of percent work productivity loss for light work."),
            data_type=float,
            units=u.percent
        ),
        spec.SingleBandRasterOutput(
            id="heavy_work_loss_percent",
            path="intermediate/heavy_work_loss_percent.tif",
            about=gettext("Map of percent work productivity loss for heavy work."),
            data_type=float,
            units=u.percent
        ),
        spec.FileOutput(
            id="wbgt_stats_pickle",
            path="intermediate/wbgt_stats.pickle",
            about=gettext("Pickled zonal statistics for WBGT.")
        ),
        spec.FileOutput(
            id="light_loss_stats_pickle",
            path="intermediate/light_loss_stats.pickle",
            about=gettext("Pickled zonal statistics for light work loss.")
        ),
        spec.FileOutput(
            id="heavy_loss_stats_pickle",
            path="intermediate/heavy_loss_stats.pickle",
            about=gettext("Pickled zonal statistics for heavy work loss.")
        ),
        spec.FileOutput(
            id="t_air_stats_pickle",
            path="intermediate/t_air_stats.pickle",
            about=gettext("Pickled zonal statistics for air temperature.")
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Urban Cooling.

    Args:
        args['workspace_dir'] (str): path to target output directory.
        args['results_suffix'] (string): (optional) string to append to any
            output file names
        args['t_ref'] (str/float): reference air temperature.
        args['lulc_raster_path'] (str): path to landcover raster.  This raster
            must be in a linearly-projected CRS.
        args['ref_eto_raster_path'] (str): path to evapotranspiration raster.
        args['aoi_vector_path'] (str): path to desired AOI.
        args['biophysical_table_path'] (str): table to map landcover codes to
            Shade, Kc, and Albedo values. Must contain the fields 'lucode',
            'kc', and 'green_area'.  If ``args['cc_method'] == 'factors'``,
            then this table must also contain the fields 'shade' and
            'albedo'.  If ``args['cc_method'] == 'intensity'``, then this
            table must also contain the field 'building_intensity'.
        args['green_area_cooling_distance'] (float): Distance (in m) over
            which large green areas (> 2 ha) will have a cooling effect.
        args['t_air_average_radius'] (float): radius of the averaging filter
            for turning T_air_nomix into T_air.
        args['uhi_max'] (float): Magnitude of the UHI effect.
        args['do_energy_valuation'] (bool): if True, calculate energy savings
            valuation for buildings.
        args['do_productivity_valuation'] (bool): if True, calculate work
            productivity valuation based on humidity and temperature.
        args['avg_rel_humidity'] (float): (optional, depends on
            'do_productivity_valuation') Average relative humidity (0-100%).
        args['building_vector_path']: (str) (optional, depends on
            'do_energy_valuation') path to a vector of building footprints that
            contains at least the field 'type'.
        args['energy_consumption_table_path'] (str): (optional, depends on
            'do_energy_valuation') path to a table that maps building types to
            energy consumption. Must contain at least the fields 'type' and
            'consumption'.
        args['cc_method'] (str): Either "intensity" or "factors".  If
            "intensity", then the "building_intensity" column must be
            present in the biophysical table.  If "factors", then
            ``args['cc_weight_shade']``, ``args['cc_weight_albedo']``,
            ``args['cc_weight_eti']`` may be set to alternative weights
            if desired.
        args['cc_weight_shade'] (str/float): floating point number
            representing the relative weight to apply to shade when
            calculating the cooling index. Default: 0.6
        args['cc_weight_albedo'] (str/float): floating point number
            representing the relative weight to apply to albedo when
            calculating the cooling index. Default: 0.2
        args['cc_weight_eti'] (str/float): floating point number
            representing the relative weight to apply to ETI when
            calculating the cooling index. Default: 0.2


    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    LOGGER.info('Starting Urban Cooling Model')
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(
        args['biophysical_table_path'])

    # Use default weights for shade, albedo, eti if the user didn't provide
    # weights.
    if args['cc_weight_shade'] is None:
        cc_weight_shade_raw = 0.6
    else:
        cc_weight_shade_raw = args['cc_weight_shade']

    if args['cc_weight_albedo'] is None:
        cc_weight_albedo_raw = 0.2
    else:
        cc_weight_albedo_raw = args['cc_weight_albedo']

    if args['cc_weight_eti'] is None:
        cc_weight_eti_raw = 0.2
    else:
        cc_weight_eti_raw = args['cc_weight_eti']

    cc_weight_sum = sum(
        (cc_weight_shade_raw, cc_weight_albedo_raw, cc_weight_eti_raw))
    cc_weight_shade = cc_weight_shade_raw / cc_weight_sum
    cc_weight_albedo = cc_weight_albedo_raw / cc_weight_sum
    cc_weight_eti = cc_weight_eti_raw / cc_weight_sum

    # align all the input rasters.
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_raster_path'])
    # ensure raster has square pixels by picking the smallest dimension
    cell_size = numpy.min(numpy.abs(lulc_raster_info['pixel_size']))

    # Reproject and align inputs to the intersection of the AOI, ET0 and LULC,
    # with target raster sizes matching those of the LULC.
    aligned_raster_path_list = [file_registry['lulc'], file_registry['ref_eto']]
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=([args['lulc_raster_path'], args['ref_eto_raster_path']],
              aligned_raster_path_list,
              ['mode', 'cubicspline'],
              (cell_size, -cell_size),
              'intersection'),
        kwargs={
            'base_vector_path_list': [args['aoi_vector_path']],
            'raster_align_index': 0,
            'target_projection_wkt': lulc_raster_info['projection_wkt']},
        target_path_list=aligned_raster_path_list,
        task_name='align rasters')

    task_path_prop_map = {}
    reclassification_props = ('kc', 'green_area')
    if args['cc_method'] == 'factors':
        reclassification_props += ('shade', 'albedo')
    else:
        reclassification_props += ('building_intensity',)

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}
    for prop in reclassification_props:
        prop_task = task_graph.add_task(
            func=utils.reclassify_raster,
            args=(
                (file_registry['lulc'], 1),
                biophysical_df[prop].to_dict(), file_registry[prop],
                gdal.GDT_Float32, TARGET_NODATA, reclass_error_details),
            target_path_list=[file_registry[prop]],
            dependent_task_list=[align_task],
            task_name=f'reclassify to {prop}')
        task_path_prop_map[prop] = (prop_task, file_registry[prop])

    align_task.join()

    if args['cc_method'] == 'factors':
        LOGGER.info('Calculating Cooling Coefficient from factors')
        # Evapotranspiration index (Equation #1)
        ref_eto_raster = gdal.OpenEx(file_registry['ref_eto'],
                                     gdal.OF_RASTER)
        ref_eto_band = ref_eto_raster.GetRasterBand(1)
        _, ref_eto_max, _, _ = ref_eto_band.GetStatistics(0, 1)
        ref_eto_max = numpy.round(ref_eto_max, decimals=9)
        ref_eto_band = None
        ref_eto_raster = None

        eto_nodata = pygeoprocessing.get_raster_info(
            args['ref_eto_raster_path'])['nodata'][0]
        eti_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(task_path_prop_map['kc'][1], 1), (TARGET_NODATA, 'raw'),
                 (file_registry['ref_eto'], 1), (eto_nodata, 'raw'),
                 (ref_eto_max, 'raw'), (TARGET_NODATA, 'raw')],
                calc_eti_op, file_registry['eti'], gdal.GDT_Float32, TARGET_NODATA),
            target_path_list=[file_registry['eti']],
            dependent_task_list=[task_path_prop_map['kc'][0]],
            task_name='calculate eti')

        cc_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([(task_path_prop_map['shade'][1], 1),
                   (task_path_prop_map['albedo'][1], 1),
                   (file_registry['eti'], 1),
                   (cc_weight_shade, 'raw'),
                   (cc_weight_albedo, 'raw'),
                   (cc_weight_eti, 'raw')],
                  calc_cc_op_factors, file_registry['cc'],
                  gdal.GDT_Float32, TARGET_NODATA),
            target_path_list=[file_registry['cc']],
            dependent_task_list=[
                task_path_prop_map['shade'][0],
                task_path_prop_map['albedo'][0],
                eti_task],
            task_name='calculate cc index (weighted factors)')
    else:
        # args['cc_method'] must be 'intensity', so we use a modified CC
        # function.
        LOGGER.info('Calculating Cooling Coefficient using '
                    'building intensity')
        cc_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=calc_cc_op_intensity,
                rasters=[task_path_prop_map['building_intensity'][1]],
                target_path=file_registry['cc'],
                target_nodata=TARGET_NODATA),
            target_path_list=[file_registry['cc']],
            dependent_task_list=[
                task_path_prop_map['building_intensity'][0]],
            task_name='calculate cc index (intensity)')

    green_area_cc_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(task_path_prop_map['green_area'][1], 1),
             (file_registry['cc'], 1)],
            mask_cc_green_areas_op,
            file_registry['cc_masked_green_areas'],
            gdal.GDT_Float32, TARGET_NODATA),
        target_path_list=[file_registry['cc_masked_green_areas']],
        dependent_task_list=[task_path_prop_map['green_area'][0], cc_task],
        task_name='Compute green area cooling effect')

    green_area_decay_kernel_distance = int(numpy.round(
        args['green_area_cooling_distance'] / cell_size))
    cc_park_task = task_graph.add_task(
        func=convolve_2d_by_exponential,
        args=(
            green_area_decay_kernel_distance,
            file_registry['cc_masked_green_areas'],
            file_registry['cc_park']),
        target_path_list=[file_registry['cc_park']],
        dependent_task_list=[green_area_cc_task],
        task_name='calculate CC park')

    area_kernel_task = task_graph.add_task(
        func=pygeoprocessing.kernels.dichotomous_kernel,
        kwargs=dict(
            target_kernel_path=file_registry['area_kernel'],
            max_distance=green_area_decay_kernel_distance,
            normalize=False),
        target_path_list=[file_registry['area_kernel']],
        task_name='area kernel')

    green_area_sum_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=(
            (task_path_prop_map['green_area'][1], 1),
            (file_registry['area_kernel'], 1),
            file_registry['green_area_sum']),
        kwargs={
            'working_dir': args['workspace_dir'],
            'ignore_nodata_and_edges': True},
        target_path_list=[file_registry['green_area_sum']],
        dependent_task_list=[
            task_path_prop_map['green_area'][0],
            area_kernel_task],
        task_name='calculate green area')

    # Compute Heat Mitigation (HM) index.
    #
    # convert 2 hectares to number of pixels
    green_area_threshold = 2e4 / cell_size**2
    hm_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (file_registry['cc'], 1),
            (file_registry['green_area_sum'], 1),
            (file_registry['cc_park'], 1),
            (green_area_threshold, 'raw')
        ], hm_op, file_registry['hm'], gdal.GDT_Float32, TARGET_NODATA),
        target_path_list=[file_registry['hm']],
        dependent_task_list=[cc_task, green_area_sum_task, cc_park_task],
        task_name='calculate HM index')

    t_air_nomix_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([(args['t_ref'], 'raw'),
               (file_registry['hm'], 1),
               (args['uhi_max'], 'raw')],
              calc_t_air_nomix_op, file_registry['t_air_nomix'], gdal.GDT_Float32,
              TARGET_NODATA),
        target_path_list=[file_registry['t_air_nomix']],
        dependent_task_list=[hm_task, align_task],
        task_name='calculate T air nomix')

    decay_kernel_distance = int(numpy.round(
        args['t_air_average_radius'] / cell_size))
    t_air_task = task_graph.add_task(
        func=convolve_2d_by_exponential,
        args=(
            decay_kernel_distance,
            file_registry['t_air_nomix'],
            file_registry['t_air']),
        target_path_list=[file_registry['t_air']],
        dependent_task_list=[t_air_nomix_task],
        task_name='calculate T air')

    intermediate_uhi_result_vector_task = task_graph.add_task(
        func=pygeoprocessing.reproject_vector,
        args=(
            args['aoi_vector_path'], lulc_raster_info['projection_wkt'],
            file_registry['reprojected_aoi']),
        kwargs={'driver_name': 'ESRI Shapefile'},
        target_path_list=[file_registry['reprojected_aoi']],
        task_name='reproject and label aoi')

    _ = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            file_registry['reprojected_aoi'],
            file_registry['cc'],
            file_registry['cc_ref_aoi_stats']),
        target_path_list=[file_registry['cc_ref_aoi_stats']],
        dependent_task_list=[cc_task, intermediate_uhi_result_vector_task],
        task_name='pickle cc ref stats')

    _ = task_graph.add_task(
        func=pickle_zonal_stats,
        args=(
            file_registry['reprojected_aoi'],
            file_registry['t_air'], file_registry['t_air_aoi_stats']),
        target_path_list=[file_registry['t_air_aoi_stats']],
        dependent_task_list=[t_air_task, intermediate_uhi_result_vector_task],
        task_name='pickle t-air over stats over AOI')

    wbgt_stats_pickle_path = None
    light_loss_stats_pickle_path = None
    heavy_loss_stats_pickle_path = None
    energy_consumption_vector_path = None
    if args['do_productivity_valuation']:
        LOGGER.info('Starting work productivity valuation')
        wbgt_task = task_graph.add_task(
            func=calculate_wbgt,
            args=(
                args['avg_rel_humidity'], file_registry['t_air'],
                file_registry['wbgt']),
            target_path_list=[file_registry['wbgt']],
            dependent_task_list=[t_air_task],
            task_name='vapor pressure')

        loss_task_path_map = {}
        for loss_type, temp_map, loss_raster_path in [
                # Breaks here are described in the UG chapter and are the
                # result of a literature review.
                ('light', [31.5, 32.0, 32.5], file_registry['light_work_loss_percent']),
                ('heavy', [27.5, 29.5, 31.5], file_registry['heavy_work_loss_percent'])]:
            work_loss_task = task_graph.add_task(
                func=map_work_loss,
                args=(temp_map, file_registry['wbgt'], loss_raster_path),
                target_path_list=[loss_raster_path],
                dependent_task_list=[wbgt_task],
                task_name=f'work loss: {os.path.basename(loss_raster_path)}')
            loss_task_path_map[loss_type] = (work_loss_task, loss_raster_path)

        wbgt_stats_pickle_path = file_registry['wbgt_stats_pickle']
        _ = task_graph.add_task(
            func=pickle_zonal_stats,
            args=(
                file_registry['reprojected_aoi'],
                file_registry['wbgt'], wbgt_stats_pickle_path),
            target_path_list=[wbgt_stats_pickle_path],
            dependent_task_list=[
                wbgt_task, intermediate_uhi_result_vector_task],
            task_name='pickle WBgt stats')
        light_loss_stats_pickle_path = file_registry['light_loss_stats_pickle']
        _ = task_graph.add_task(
            func=pickle_zonal_stats,
            args=(
                file_registry['reprojected_aoi'],
                loss_task_path_map['light'][1], light_loss_stats_pickle_path),
            target_path_list=[light_loss_stats_pickle_path],
            dependent_task_list=[
                loss_task_path_map['light'][0],
                intermediate_uhi_result_vector_task],
            task_name='pickle light_loss stats')

        heavy_loss_stats_pickle_path = file_registry['heavy_loss_stats_pickle']
        _ = task_graph.add_task(
            func=pickle_zonal_stats,
            args=(
                file_registry['reprojected_aoi'],
                loss_task_path_map['heavy'][1], heavy_loss_stats_pickle_path),
            target_path_list=[heavy_loss_stats_pickle_path],
            dependent_task_list=[
                loss_task_path_map['heavy'][0],
                intermediate_uhi_result_vector_task],
            task_name='pickle heavy_loss stats')

    if bool(args['do_energy_valuation']):
        LOGGER.info('Starting energy savings valuation')
        intermediate_building_vector_task = task_graph.add_task(
            func=pygeoprocessing.reproject_vector,
            args=(
                args['building_vector_path'],
                lulc_raster_info['projection_wkt'],
                file_registry['reprojected_buildings']),
            kwargs={'driver_name': 'ESRI Shapefile'},
            target_path_list=[file_registry['reprojected_buildings']],
            task_name='reproject building vector')

        pickle_t_air_task = task_graph.add_task(
            func=pickle_zonal_stats,
            args=(
                file_registry['reprojected_buildings'],
                file_registry['t_air'], file_registry['t_air_stats_pickle']),
            target_path_list=[file_registry['t_air_stats_pickle']],
            dependent_task_list=[
                t_air_task, intermediate_building_vector_task],
            task_name='pickle t-air stats over buildings')

        energy_consumption_vector_path = file_registry['buildings_with_stats']
        _ = task_graph.add_task(
            func=calculate_energy_savings,
            args=(
                file_registry['t_air_stats_pickle'], args['t_ref'],
                args['uhi_max'], args['energy_consumption_table_path'],
                file_registry['reprojected_buildings'],
                file_registry['buildings_with_stats']),
            target_path_list=[file_registry['buildings_with_stats']],
            dependent_task_list=[
                pickle_t_air_task, intermediate_building_vector_task],
            task_name='calculate energy savings task')

    # final reporting can't be done until everything else is complete so
    # stop here
    task_graph.join()

    _ = task_graph.add_task(
        func=calculate_uhi_result_vector,
        args=(
            file_registry['reprojected_aoi'],
            args['t_ref'],
            file_registry['t_air_aoi_stats'],
            file_registry['cc_ref_aoi_stats'],
            wbgt_stats_pickle_path,
            light_loss_stats_pickle_path,
            heavy_loss_stats_pickle_path,
            energy_consumption_vector_path,
            file_registry['uhi_results']),
        target_path_list=[file_registry['uhi_results']],
        task_name='calculate uhi results')

    task_graph.close()
    task_graph.join()
    LOGGER.info('Urban Cooling Model complete.')
    return file_registry.registry


def calculate_uhi_result_vector(
        base_aoi_path, t_ref_val, t_air_stats_pickle_path,
        cc_stats_pickle_path,
        wbgt_stats_pickle_path,
        light_loss_stats_pickle_path,
        heavy_loss_stats_pickle_path,
        energy_consumption_vector_path, target_uhi_vector_path):
    """Summarize UHI results.

    Output vector will have fields with attributes summarizing:
        * average cc value
        * average temperature value
        * average temperature anomaly
        * avoided energy consumption

    Args:
        base_aoi_path (str): path to AOI vector.
        t_ref_val (float): reference temperature.
        wbgt_stats_pickle_path (str): path to pickled zonal stats for wbgt.
            Can be None if no valuation occurred.
        light_loss_stats_pickle_path (str): path to pickled zonal stats for
            light work loss. Can be None if no valuation occurred.
        heavy_loss_stats_pickle_path (str): path to pickled zonal stats for
            heavy work loss. Can be None if no valuation occurred.
        energy_consumption_vector_path (str): path to vector that contains
            building footprints with the field 'energy_sav'. Can be None
            if no valuation occurred.
        target_uhi_vector_path (str): path to UHI vector created for result.
            Will contain the fields:

                * avg_cc
                * avg_tmp_an
                * avd_eng_cn
                * average WBGT
                * average light loss work
                * average heavy loss work

    Returns:
        None.

    """
    LOGGER.info("Calculate UHI summary results for "
                f"{os.path.basename(target_uhi_vector_path)}")

    LOGGER.info("Loading t_air_stats")
    with open(t_air_stats_pickle_path, 'rb') as t_air_stats_pickle_file:
        t_air_stats = pickle.load(t_air_stats_pickle_file)
    LOGGER.info("Loading cc_stats")
    with open(cc_stats_pickle_path, 'rb') as cc_stats_pickle_file:
        cc_stats = pickle.load(cc_stats_pickle_file)

    wbgt_stats = None
    if wbgt_stats_pickle_path:
        LOGGER.info("Loading wbgt_stats")
        with open(wbgt_stats_pickle_path, 'rb') as wbgt_stats_pickle_file:
            wbgt_stats = pickle.load(wbgt_stats_pickle_file)

    light_loss_stats = None
    if light_loss_stats_pickle_path:
        LOGGER.info("Loading light_loss_stats")
        with open(light_loss_stats_pickle_path, 'rb') as (
                light_loss_stats_pickle_file):
            light_loss_stats = pickle.load(light_loss_stats_pickle_file)

    heavy_loss_stats = None
    if heavy_loss_stats_pickle_path:
        LOGGER.info("Loading heavy_loss_stats")
        with open(heavy_loss_stats_pickle_path, 'rb') as (
                heavy_loss_stats_pickle_file):
            heavy_loss_stats = pickle.load(heavy_loss_stats_pickle_file)

    base_aoi_vector = gdal.OpenEx(base_aoi_path, gdal.OF_VECTOR)
    shapefile_driver = gdal.GetDriverByName('ESRI Shapefile')
    try:
        # Can't make a shapefile on top of an existing one
        os.remove(target_uhi_vector_path)
    except FileNotFoundError:
        pass

    LOGGER.info(f"Creating {os.path.basename(target_uhi_vector_path)}")
    shapefile_driver.CreateCopy(
        target_uhi_vector_path, base_aoi_vector)
    base_aoi_vector = None
    target_uhi_vector = gdal.OpenEx(
        target_uhi_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_uhi_layer = target_uhi_vector.GetLayer()

    for field_id in [
            'avg_cc', 'avg_tmp_v', 'avg_tmp_an', 'avd_eng_cn', 'avg_wbgt_v',
            'avg_ltls_v', 'avg_hvls_v']:
        target_uhi_layer.CreateField(ogr.FieldDefn(field_id, ogr.OFTReal))

    # I don't really like having two of the same conditions (one here and one
    # in the for feature in target_uhi_layer loop), but if the user has
    # multiple AOI features, we shouldn't have to rebuild the buildings spatial
    # index every time.
    if energy_consumption_vector_path:
        energy_consumption_vector = gdal.OpenEx(
            energy_consumption_vector_path, gdal.OF_VECTOR)
        energy_consumption_layer = energy_consumption_vector.GetLayer()

        LOGGER.info('Parsing building footprint geometry')
        building_shapely_polygon_lookup = dict()
        for poly_feat in energy_consumption_layer:
            geom = poly_feat.GetGeometryRef()
            if geom:
                building_shapely_polygon_lookup[poly_feat.GetFID()] = (
                    shapely.wkb.loads(bytes(geom.ExportToWkb())))

        LOGGER.info("Constructing building footprint spatial index")
        poly_rtree_index = rtree.index.Index(
            [(poly_fid, poly.bounds, None)
             for poly_fid, poly in
             building_shapely_polygon_lookup.items()])

    target_uhi_layer.StartTransaction()
    for feature in target_uhi_layer:
        feature_id = feature.GetFID()
        if cc_stats[feature_id]['count'] > 0:
            mean_cc = (
                cc_stats[feature_id]['sum'] / cc_stats[feature_id]['count'])
            feature.SetField('avg_cc', float(mean_cc))
        mean_t_air = None
        if t_air_stats[feature_id]['count'] > 0:
            mean_t_air = (
                t_air_stats[feature_id]['sum'] /
                t_air_stats[feature_id]['count'])
            feature.SetField('avg_tmp_v', float(mean_t_air))

        if mean_t_air:
            feature.SetField(
                'avg_tmp_an', mean_t_air-t_ref_val)

        if wbgt_stats and wbgt_stats[feature_id]['count'] > 0:
            wbgt = (
                wbgt_stats[feature_id]['sum'] /
                wbgt_stats[feature_id]['count'])
            feature.SetField('avg_wbgt_v', float(wbgt))

        if light_loss_stats and light_loss_stats[feature_id]['count'] > 0:
            light_loss = (
                light_loss_stats[feature_id]['sum'] /
                light_loss_stats[feature_id]['count'])
            LOGGER.debug(f"Average light loss: {light_loss}")
            feature.SetField('avg_ltls_v', float(light_loss))

        if heavy_loss_stats and heavy_loss_stats[feature_id]['count'] > 0:
            heavy_loss = (
                heavy_loss_stats[feature_id]['sum'] /
                heavy_loss_stats[feature_id]['count'])
            LOGGER.debug(f"Average heavy loss: {heavy_loss}")
            feature.SetField('avg_hvls_v', float(heavy_loss))

        if energy_consumption_vector_path:
            aoi_geometry = feature.GetGeometryRef()
            aoi_shapely_geometry = shapely.wkb.loads(
                bytes(aoi_geometry.ExportToWkb()))
            aoi_shapely_geometry_prep = shapely.prepared.prep(
                aoi_shapely_geometry)
            avd_eng_cn = 0
            for building_fid in poly_rtree_index.intersection(
                    aoi_shapely_geometry.bounds):
                if aoi_shapely_geometry_prep.intersects(
                        building_shapely_polygon_lookup[building_fid]):
                    energy_consumption_value = (
                        energy_consumption_layer.GetFeature(
                            building_fid).GetField('energy_sav'))
                    if energy_consumption_value:
                        # this step lets us skip values that might be in
                        # nodata ranges that we can't help.
                        avd_eng_cn += float(energy_consumption_value)
            feature.SetField('avd_eng_cn', avd_eng_cn)

        target_uhi_layer.SetFeature(feature)
    target_uhi_layer.CommitTransaction()


def calculate_energy_savings(
        t_air_stats_pickle_path, t_ref_raw, uhi_max,
        energy_consumption_table_path, base_building_vector_path,
        target_building_vector_path):
    """Calculate energy savings.

    Energy savings is calculated from equations 8 or 9 in the User's Guide
    (depending on whether a cost has been provided in the energy consumption
    table).

    Args:
        t_air_stats_pickle_path (str): path to t_air zonal stats indexed by
            FID.
        t_ref_raw (float): single value for Tref.
        uhi_max (float): UHI max parameter from documentation.
        energy_consumption_table_path (str): path to energy consumption table
            that contains at least the columns 'type', and 'consumption'.  If
            the table also contains a 'cost' column, the output energy
            savings field will be multiplied by the floating-point cost
            provided in the 'cost' column.
        base_building_vector_path (str): path to existing vector to copy for
            the target vector that contains at least the field 'type'.
        target_building_vector_path (str): path to target vector that
            will contain the additional field 'energy_sav' calculated as
            ``consumption.increase(b) * ((T_(air,MAX)  - T_(air,i)))``.
            This vector must be in a linearly projected spatial reference
            system.

    Return:
        None.

    """
    LOGGER.info(f"Calculate energy savings for {target_building_vector_path}")
    LOGGER.info("Loading t_air_stats")
    with open(t_air_stats_pickle_path, 'rb') as t_air_stats_pickle_file:
        t_air_stats = pickle.load(t_air_stats_pickle_file)

    base_building_vector = gdal.OpenEx(
        base_building_vector_path, gdal.OF_VECTOR)
    shapefile_driver = gdal.GetDriverByName('ESRI Shapefile')
    LOGGER.info(f"Creating {os.path.basename(target_building_vector_path)}")
    try:
        # can't make a shapefile on top of an existing one
        os.remove(target_building_vector_path)
    except OSError:
        pass
    shapefile_driver.CreateCopy(
        target_building_vector_path, base_building_vector)
    base_building_vector = None
    target_building_vector = gdal.OpenEx(
        target_building_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_building_layer = target_building_vector.GetLayer()
    target_building_srs = target_building_layer.GetSpatialRef()
    target_building_square_units = target_building_srs.GetLinearUnits() ** 2
    target_building_layer.CreateField(
        ogr.FieldDefn('energy_sav', ogr.OFTReal))
    target_building_layer.CreateField(
        ogr.FieldDefn('mean_t_air', ogr.OFTReal))

    # Find the index of the 'type' column in a case-insensitive way.
    # We can assume that the field exists because we're checking for it in
    # validation as defined in MODEL_SPEC.
    fieldnames = [field.GetName().lower()
                  for field in target_building_layer.schema]
    type_field_index = fieldnames.index('type')

    energy_consumption_df = MODEL_SPEC.get_input(
        'energy_consumption_table_path').get_validated_dataframe(
        energy_consumption_table_path)

    target_building_layer.StartTransaction()
    last_time = time.time()
    for target_index, target_feature in enumerate(target_building_layer):
        last_time = _invoke_timed_callback(
            last_time, lambda: LOGGER.info(
                "energy savings approximately %.1f%% complete ",
                100 * float(target_index + 1) /
                target_building_layer.GetFeatureCount()),
            _LOGGING_PERIOD)

        feature_id = target_feature.GetFID()
        t_air_mean = None
        if feature_id in t_air_stats:
            pixel_count = float(t_air_stats[feature_id]['count'])
            if pixel_count > 0:
                t_air_mean = float(
                    t_air_stats[feature_id]['sum'] / pixel_count)
                target_feature.SetField('mean_t_air', t_air_mean)

        # Building type should be an integer and has to match the building
        # types in the energy consumption table.
        target_type = target_feature.GetField(int(type_field_index))
        if target_type not in energy_consumption_df.index:
            target_building_layer.CommitTransaction()
            target_building_layer = None
            target_building_vector = None
            raise ValueError(
                f"Encountered a building 'type' of: '{target_type}' in "
                f"FID: {target_feature.GetFID()} in the building vector layer "
                "that has no corresponding entry in the energy consumption "
                f"table at {energy_consumption_table_path}")

        consumption_increase = energy_consumption_df['consumption'][target_type]

        # Load building cost if we can, but don't adjust the value if the cost
        # column is not there.
        # NOTE: if the user has an empty column value but the 'cost' column
        # exists, this will raise an error.
        try:
            building_cost = energy_consumption_df['cost'][target_type]
        except KeyError:
            # KeyError when cost column not present.
            building_cost = 1

        # Calculate Equations 8, 9: Energy Savings.
        # We'll only calculate energy savings if there were polygons with valid
        # stats that could be aggregated from t_air_mean.
        if t_air_mean:
            building_area = target_feature.GetGeometryRef().Area()
            building_area_m2 = building_area * target_building_square_units
            savings = (
                consumption_increase * building_area_m2 * (
                    t_ref_raw - t_air_mean + uhi_max) * building_cost)
            target_feature.SetField('energy_sav', savings)

        target_building_layer.SetFeature(target_feature)
    target_building_layer.CommitTransaction()
    target_building_layer.SyncToDisk()


def pickle_zonal_stats(
        base_vector_path, base_raster_path, target_pickle_path):
    """Calculate Zonal Stats for a vector/raster pair and pickle result.

    Args:
        base_vector_path (str): path to vector file
        base_raster_path (str): path to raster file to aggregate over.
        target_pickle_path (str): path to desired target pickle file that will
            be a pickle of the pygeoprocessing.zonal_stats function.

    Returns:
        None.

    """
    LOGGER.info(f'Taking zonal statistics of {base_vector_path} '
                f'over {base_raster_path}')
    zonal_stats = pygeoprocessing.zonal_statistics(
        (base_raster_path, 1), base_vector_path,
        polygons_might_overlap=True)
    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(zonal_stats, pickle_file)


def calc_t_air_nomix_op(t_ref_val, hm_array, uhi_max):
    """Calculate air temperature T_(air,i)=T_ref+(1-HM_i)*UHI_max.

    Args:
        t_ref_val (float): The user-defined reference air temperature in
            degrees Celsius.
        hm_array (numpy.ndarray): The calculated Heat Mitigation index from
            equation 5 in the User's Guide.
        uhi_max (float): The user-defined maximum UHI magnitude.

    Returns:
        A numpy array with the same dimensions as ``hm_array`` with the
        calculated T_air_nomix values.

    """
    result = numpy.empty(hm_array.shape, dtype=numpy.float32)
    result[:] = TARGET_NODATA
    # TARGET_NODATA should never be None
    valid_mask = ~pygeoprocessing.array_equals_nodata(hm_array, TARGET_NODATA)
    result[valid_mask] = t_ref_val + (1-hm_array[valid_mask]) * uhi_max
    return result


def calc_cc_op_factors(
        shade_array, albedo_array, eti_array, cc_weight_shade,
        cc_weight_albedo, cc_weight_eti):
    """Calculate the cooling capacity index using weighted factors.

    Args:
        shade_array (numpy.ndarray): array of shade index values 0..1
        albedo_array (numpy.ndarray): array of albedo index values 0..1
        eti_array (numpy.ndarray): array of evapotransipration index values
            0..1
        cc_weight_shade (float): 0..1 weight to apply to shade
        cc_weight_albedo (float): 0..1 weight to apply to albedo
        cc_weight_eti (float): 0..1 weight to apply to eti

    Returns:
         CC_i = ((cc_weight_shade * shade) +
                 (cc_weight_albedo * albedo) +
                 (cc_weight_eti * ETI))

    """
    result = numpy.empty(shade_array.shape, dtype=numpy.float32)
    result[:] = TARGET_NODATA
    valid_mask = ~(
        pygeoprocessing.array_equals_nodata(shade_array, TARGET_NODATA) |
        pygeoprocessing.array_equals_nodata(albedo_array, TARGET_NODATA) |
        pygeoprocessing.array_equals_nodata(eti_array, TARGET_NODATA))
    result[valid_mask] = (
        cc_weight_shade*shade_array[valid_mask] +
        cc_weight_albedo*albedo_array[valid_mask] +
        cc_weight_eti*eti_array[valid_mask])
    return result


def calc_cc_op_intensity(intensity_array):
    """Calculate the cooling capacity index using building intensity.

    Args:
        intensity_array (numpy.ndarray): array of intensity values.

    Returns:
        A numpy array of ``1 - intensity_array``.

    """
    return 1 - intensity_array


def calc_eti_op(
        kc_array, kc_nodata, et0_array, et0_nodata, et_max, target_nodata):
    """Calculate ETI = (K_c * ET_0) / ET_max."""
    result = numpy.empty(kc_array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    # kc intermediate output should always have a nodata value defined
    valid_mask = ~pygeoprocessing.array_equals_nodata(kc_array, kc_nodata)
    if et0_nodata is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(et0_array, et0_nodata)
    result[valid_mask] = (
        kc_array[valid_mask] * et0_array[valid_mask] / et_max)
    return result


def calculate_wbgt(
        avg_rel_humidity, t_air_raster_path, target_vapor_pressure_path):
    """Raster calculator op to calculate wet bulb globe temperature.

    Args:
        avg_rel_humidity (float): number between 0-100.
        t_air_raster_path (string): path to T air raster.
        target_vapor_pressure_path (string): path to target vapor pressure
            raster.

    Returns:
        WBGT_i  = 0.567 * T_(air,i)  + 0.393 * e_i  + 3.94

        where e_i:
            e_i  = RH/100*6.105*exp(17.27*T_air/(237.7+T_air))

    """
    LOGGER.info('Calculating WBGT')
    pygeoprocessing.raster_map(
        op=lambda t_air: 0.567 * t_air + 0.393 * (
            (avg_rel_humidity / 100) * 6.105 * numpy.exp(
                17.27 * (t_air / (237.7 + t_air)))) + 3.94,
        rasters=[t_air_raster_path],
        target_path=target_vapor_pressure_path)


def hm_op(cc_array, green_area_sum, cc_park_array, green_area_threshold):
    """Calculate HM.

        cc_array (numpy.ndarray): this is the raw cooling index mapped from
            landcover values.
        green_area_sum (numpy.ndarray): this is the sum of green space pixels
            pixels within the user defined area for green space.
        cc_park_array (numpy.ndarray): this is the exponentially decayed
            cooling index due to proximity of green space.
        green_area_threshold (float): a value used to determine how much
            area is required to trigger a green area overwrite.

    Returns:
        cc_array if green area < green_area_threshold or cc_park < cc array,
        otherwise cc_park array is returned.

    """
    result = numpy.empty(cc_array.shape, dtype=numpy.float32)
    result[:] = TARGET_NODATA
    valid_mask = ~(pygeoprocessing.array_equals_nodata(cc_array, TARGET_NODATA) &
                   pygeoprocessing.array_equals_nodata(cc_park_array, TARGET_NODATA))
    cc_mask = ((cc_array >= cc_park_array) |
               (green_area_sum < green_area_threshold))
    result[cc_mask & valid_mask] = cc_array[cc_mask & valid_mask]
    result[~cc_mask & valid_mask] = cc_park_array[~cc_mask & valid_mask]
    return result


def mask_cc_green_areas_op(green_area_array, cc_array):
    """
    Mask out non-green areas from the cooling capacity (CC) raster.

    Both `green_area_array` and `cc_array` originate from the LULC raster:
        - `green_area_array` is a reclassification of LULC.
        - `cc_array` is derived from raster calculations using reclassified
           LULC arrays and ETI.
    This function sets `cc_array` to:
        - Its original cc value in green areas.
        - 0 in non-green areas.
        - Nodata if either `cc_array` or `green_area_array` are nodata

    Args:
        cc_array (numpy.ndarray): this is the raw cooling index mapped from
            landcover values.
        green_area_array (numpy.ndarray): this is the boolean array of green
            areas where 1 corresponds with lulc classes that are green areas
            and 0 represent areas that are not green.

    Returns:
        A modified `cc_array` where only green areas retain their original values, 
        non-green areas are set to 0, and nodata values are preserved.

    """
    result = numpy.empty(cc_array.shape, dtype=numpy.float32)
    result[:] = TARGET_NODATA
    valid_mask = ~(pygeoprocessing.array_equals_nodata(cc_array, TARGET_NODATA) |
                   pygeoprocessing.array_equals_nodata(green_area_array, TARGET_NODATA))
    green_area_mask = green_area_array.astype(bool)
    result[green_area_mask & valid_mask] = cc_array[green_area_mask & valid_mask]
    result[~green_area_mask & valid_mask] = 0

    return result


def map_work_loss(
        work_temp_threshold_array, temperature_raster_path,
        work_loss_raster_path):
    """Map work loss due to temperature.

    Args:
        work_temp_threshold_array (list): list of 3 sorted floats indicating
            the thresholds for 25, 50, and 75% work loss.
        temperature_raster_path (string): path to temperature raster in the
            same units as `work_temp_threshold_array`.
        work_loss_raster_path (string): path to target raster that maps per
            pixel work loss percent.

    Returns:
        None.

    """
    LOGGER.info(
        f'Calculating work loss using thresholds: {work_temp_threshold_array}')

    def classify_to_percent_op(temperature_array):
        result = numpy.empty(temperature_array.shape)
        result[temperature_array < work_temp_threshold_array[0]] = 0
        result[
            (temperature_array >= work_temp_threshold_array[0]) &
            (temperature_array < work_temp_threshold_array[1])] = 25
        result[
            (temperature_array >= work_temp_threshold_array[1]) &
            (temperature_array < work_temp_threshold_array[2])] = 50
        result[temperature_array >= work_temp_threshold_array[2]] = 75
        return result

    pygeoprocessing.raster_map(
        op=classify_to_percent_op,
        rasters=[temperature_raster_path],
        target_path=work_loss_raster_path,
        target_dtype=numpy.uint8)


def _invoke_timed_callback(
        reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Args:
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


def convolve_2d_by_exponential(
        decay_kernel_distance, signal_raster_path,
        target_convolve_raster_path):
    """Convolve signal by an exponential decay of a given radius.

    Args:
        decay_kernel_distance (float): radius of 1/e cutoff of decay kernel
            raster in pixels.
        signal_raster_path (str): path to single band signal raster.
        target_convolve_raster_path (str): path to convolved raster.

    Returns:
        None.

    """
    LOGGER.info(f"Starting a convolution over {signal_raster_path} with a "
                f"decay distance of {decay_kernel_distance}")
    temporary_working_dir = tempfile.mkdtemp(
        dir=os.path.dirname(target_convolve_raster_path))
    exponential_kernel_path = os.path.join(
        temporary_working_dir, 'exponential_decay_kernel.tif')
    pygeoprocessing.kernels.exponential_decay_kernel(
        target_kernel_path=exponential_kernel_path,
        max_distance=decay_kernel_distance * 5,
        expected_distance=decay_kernel_distance)
    pygeoprocessing.convolve_2d(
        (signal_raster_path, 1), (exponential_kernel_path, 1),
        target_convolve_raster_path, working_dir=temporary_working_dir,
        ignore_nodata_and_edges=True)
    shutil.rmtree(temporary_working_dir)


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
    validation_warnings = validation.validate(args, MODEL_SPEC)

    invalid_keys = validation.get_invalid_keys(validation_warnings)
    if ('biophysical_table_path' not in invalid_keys and
            'cc_method' not in invalid_keys):
        spec = copy.deepcopy(MODEL_SPEC.get_input('biophysical_table_path'))
        if args['cc_method'] == 'factors':
            spec.get_column('shade').required = True
            spec.get_column('albedo').required = True
        else:
            # args['cc_method'] must be 'intensity'.
            # If args['cc_method'] isn't one of these two allowed values
            # ('intensity' or 'factors'), it'll be caught by
            # validation.validate due to the allowed values stated in
            # MODEL_SPEC.
            spec.get_column('building_intensity').required = True

        error_msg = spec.validate(args['biophysical_table_path'])
        if error_msg:
            validation_warnings.append((['biophysical_table_path'], error_msg))

    return validation_warnings
