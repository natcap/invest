# coding=UTF-8
"""InVEST Habitat Quality model."""
import collections
import csv
import logging
import os

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

MISSING_SENSITIVITY_TABLE_THREATS_MSG = gettext(
    'Threats {threats} does not match any column in the sensitivity table. '
    'Sensitivity columns: {column_names}')  # (set of missing threats, set of found columns)
MISSING_THREAT_RASTER_MSG = gettext(
    "A threat raster for threats: {threat_list} was not found or it "
    "could not be opened by GDAL.")
DUPLICATE_PATHS_MSG = gettext("Threat paths must be unique. Duplicates: ")
INVALID_MAX_DIST_MSG = gettext(
    "The maximum distance value for threats: {threat_list} is less than "
    "or equal to 0. MAX_DIST must be a positive value.")
MISSING_MAX_DIST_MSG = gettext(
    "Maximum distance value is missing for threats: {threat_list}.")
MISSING_WEIGHT_MSG = gettext("Weight value is missing for threats: {threat_list}.")

MODEL_SPEC = spec.ModelSpec(
    model_id="habitat_quality",
    model_title=gettext("Habitat Quality"),
    userguide="habitat_quality.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("hq",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_cur_path", "lulc_fut_path", "lulc_bas_path"],
        ["threats_table_path", "access_vector_path",
         "sensitivity_table_path", "half_saturation_constant"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="lulc_cur_path",
            name=gettext("current land cover"),
            about=gettext(
                "Map of LULC at present. All values in this raster must have"
                " corresponding entries in the Sensitivity table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="lulc_fut_path",
            name=gettext("future land cover"),
            about=gettext(
                "Map of LULC in a future scenario. All values in this raster must have"
                " corresponding entries in the Sensitivity Table. Must use the same"
                " classification scheme and codes as in the Current LULC map."
            ),
            required=False,
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SingleBandRasterInput(
            id="lulc_bas_path",
            name=gettext("baseline land cover"),
            about=gettext(
                "Map of LULC in a baseline scenario, when intensive landscape management"
                " was relatively rare. All values in this raster must have corresponding"
                " entries in the Sensitivity table. Must use the same classification"
                " scheme and codes as in the Current LULC map."
            ),
            required=False,
            data_type=int,
            units=None,
            projected=True
        ),
        spec.CSVInput(
            id="threats_table_path",
            name=gettext("threats table"),
            about=gettext(
                "Table mapping each threat of interest to its properties and distribution"
                " maps. Paths are relative to the threats table path."
            ),
            columns=[
                spec.StringInput(
                    id="threat",
                    about=gettext(
                        "Name of the threat. Each threat name must have a corresponding"
                        " column in the Sensitivity table."
                    ),
                    regexp=None
                ),
                spec.NumberInput(
                    id="max_dist",
                    about=gettext(
                        "The maximum distance over which each threat affects habitat"
                        " quality. The impact of each degradation source will decline to"
                        " zero at this maximum distance. This value must be greater than"
                        " or equal to the pixel size of your LULC raster(s)."
                    ),
                    units=u.meter
                ),
                spec.RatioInput(
                    id="weight",
                    about=gettext(
                        "The impact of each threat on habitat quality, relative to other"
                        " threats."
                    ),
                    units=None
                ),
                spec.OptionStringInput(
                    id="decay",
                    about=gettext("The type of decay over space for each threat."),
                    options=[
                        spec.Option(
                            key="linear",
                            about=(
                                "Effects of the threat decay linearly with distance from"
                                " the threat.")),
                        spec.Option(
                            key="exponential",
                            about=(
                                "Effects of the threat decay exponentially with distance"
                                " from the threat."))
                    ]
                ),
                spec.SingleBandRasterInput(
                    id="cur_path",
                    about=(
                        "Path to a raster of the threat's distribution in the current"
                        " scenario. Each pixel value in this raster is the relative"
                        " intensity of the threat at that location, with values between 0"
                        " and 1."
                    ),
                    data_type=float,
                    units=None,
                    projected=None
                ),
                spec.SingleBandRasterInput(
                    id="fut_path",
                    about=(
                        "Path to a raster of the threat's distribution in a future"
                        " scenario. Each pixel value in this raster is the relative"
                        " intensity of the threat at that location, with values between 0"
                        " and 1."
                    ),
                    required="lulc_fut_path",
                    data_type=float,
                    units=None,
                    projected=None
                ),
                spec.SingleBandRasterInput(
                    id="base_path",
                    about=(
                        "Path to a raster of the threat's distribution in the baseline"
                        " scenario. Each pixel value in this raster is the relative"
                        " intensity of the threat at that location, with values between 0"
                        " and 1. Required if Baseline LULC is provided."
                    ),
                    required="lulc_bas_path",
                    data_type=float,
                    units=None,
                    projected=None
                )
            ],
            index_col="threat"
        ),
        spec.VectorInput(
            id="access_vector_path",
            name=gettext("accessibility to threats"),
            about=gettext(
                "Map of the relative protection that legal, institutional, social, and"
                " physical barriers provide against threats. Any cells not covered by a"
                " polygon will be set to 1."
            ),
            required=False,
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.RatioInput(
                    id="access",
                    about=(
                        "The region's relative accessibility to threats, where 0"
                        " represents completely inaccessible and 1 represents completely"
                        " accessible."
                    ),
                    units=None
                )
            ],
            projected=False
        ),
        spec.CSVInput(
            id="sensitivity_table_path",
            name=gettext("sensitivity table"),
            about=(
                "Table mapping each LULC class to data about the species' habitat"
                " preference and threat sensitivity in areas with that LULC."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.StringInput(id="name", about=None, required=False, regexp=None),
                spec.RatioInput(
                    id="habitat",
                    about=gettext(
                        "Suitability of this LULC class as habitat, where 0 is not"
                        " suitable and 1 is completely suitable."
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="[THREAT]",
                    about=(
                        "The relative sensitivity of each LULC class to each type of"
                        " threat, where 1 represents high sensitivity and 0 represents"
                        " that it is unaffected. There must be one threat column for each"
                        " threat name in the 'threats' column of the Threats Table."
                    ),
                    units=None
                )
            ],
            index_col="lucode"
        ),
        spec.NumberInput(
            id="half_saturation_constant",
            name=gettext("half-saturation constant"),
            about=gettext("Half-saturation constant used in the degradation equation."),
            units=u.none,
            expression="value > 0"
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="deg_sum_c",
            path="deg_sum_c.tif",
            about=gettext(
                "Relative level of habitat degradation on the current landscape."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="deg_sum_f",
            path="deg_sum_f.tif",
            about=gettext(
                "Relative level of habitat degradation on the future landscape."
            ),
            created_if="lulc_fut_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="quality_c",
            path="quality_c.tif",
            about=gettext(
                "Relative level of habitat quality on the current landscape."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="quality_f",
            path="quality_f.tif",
            about=gettext(
                "Relative level of habitat quality on the future landscape."
            ),
            created_if="lulc_fut_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="new_cover_c",
            path="new_cover_c.tif",
            about=gettext("Masked current land cover"),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="new_cover_f",
            path="new_cover_f.tif",
            about=gettext("Masked future land cover"),
            created_if="lulc_fut_path",
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="rarity_c",
            path="rarity_c.tif",
            about=(
                "Relative habitat rarity on the current landscape vis-a-vis the"
                " baseline map. The grid cell's values are defined between a"
                " range of 0 and 1 where 0.5 indicates no abundance change"
                " between the baseline and current or projected map. Values"
                " between 0 and 0.5 indicate a habitat is more abundant and the"
                " closer the value is to 0 the lesser the likelihood that the"
                " preservation of that habitat type on the current or future"
                " landscape is important to biodiversity conservation. Values"
                " between 0.5 and 1 indicate a habitat is less abundant and the"
                " closer the value is to 1 the greater the likelihood that the"
                " preservation of that habitat type on the current or future"
                " landscape is important to biodiversity conservation."
            ),
            created_if="lulc_bas_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="rarity_f",
            path="rarity_f.tif",
            about=(
                "Relative habitat rarity on the future landscape vis-a-vis the"
                " baseline map. The grid cell's values are defined between a"
                " range of 0 and 1 where 0.5 indicates no abundance change"
                " between the baseline and current or projected map. Values"
                " between 0 and 0.5 indicate a habitat is more abundant and the"
                " closer the value is to 0 the lesser the likelihood that the"
                " preservation of that habitat type on the current or future"
                " landscape is important to biodiversity conservation. Values"
                " between 0.5 and 1 indicate a habitat is less abundant and the"
                " closer the value is to 1 the greater the likelihood that the"
                " preservation of that habitat type on the current or future"
                " landscape is important to biodiversity conservation."
            ),
            created_if="lulc_bas_path and lulc_fut_path",
            data_type=float,
            units=None
        ),
        spec.CSVOutput(
            id="rarity_c_csv",
            path="rarity_c.csv",
            about=gettext(
                "Table of rarity values by LULC code for the current landscape."
            ),
            created_if="lulc_bas_path",
            columns=[
                spec.NumberOutput(
                    id="lucode", about=gettext("LULC class"), units=u.none
                ),
                spec.NumberOutput(
                    id="rarity_value",
                    about=gettext(
                        "Relative habitat rarity on the current landscape"
                        " vis-a-vis the baseline map. The rarity values are"
                        " defined between a range of 0 and 1 where 0.5 indicates"
                        " no abundance change between the baseline and current or"
                        " projected map. Values between 0 and 0.5 indicate a"
                        " habitat is more abundant and the closer the value is to"
                        " 0 the lesser the likelihood that the preservation of"
                        " that habitat type on the current or future landscape is"
                        " important to biodiversity conservation. Values between"
                        " 0.5 and 1 indicate a habitat is less abundant and the"
                        " closer the value is to 1 the greater the likelihood"
                        " that the preservation of that habitat type on the"
                        " current or future landscape is important to"
                        " biodiversity conservation."
                    ),
                    units=u.none
                )
            ],
            index_col="lucode"
        ),
        spec.CSVOutput(
            id="rarity_f_csv",
            path="rarity_f.csv",
            about=gettext(
                "Table of rarity values by LULC code for the future landscape."
            ),
            created_if="lulc_bas_path and lulc_fut_path",
            columns=[
                spec.NumberOutput(
                    id="lucode", about=gettext("LULC class"), units=u.none
                ),
                spec.NumberOutput(
                    id="rarity_value",
                    about=gettext(
                        "Relative habitat rarity on the future landscape"
                        " vis-a-vis the baseline map. The rarity values are"
                        " defined between a range of 0 and 1 where 0.5 indicates"
                        " no abundance change between the baseline and current or"
                        " projected map. Values between 0 and 0.5 indicate a"
                        " habitat is more abundant and the closer the value is to"
                        " 0 the lesser the likelihood that the preservation of"
                        " that habitat type on the current or future landscape is"
                        " important to biodiversity conservation. Values between"
                        " 0.5 and 1 indicate a habitat is less abundant and the"
                        " closer the value is to 1 the greater the likelihood"
                        " that the preservation of that habitat type on the"
                        " current or future landscape is important to"
                        " biodiversity conservation."
                    ),
                    units=u.none
                )
            ],
            index_col="lucode"
        ),
        spec.SingleBandRasterOutput(
            id="access_layer",
            path="intermediate/access_layer.tif",
            about=gettext("Rasterized access map"),
            data_type=float,
            units=None
        ),
        spec.VectorOutput(
            id="reprojected_access_vector",
            path="intermediate/access_projected_to_lulc_cur.gpkg",
            about=gettext("Access vector reprojected to LULC projection"),
            created_if="access_vector_path",
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[],
            projected=False
        ),
        spec.SingleBandRasterOutput(
            id="lulc_cur_aligned",
            path="intermediate/lulc_cur_aligned.tif",
            about=gettext("Aligned copy of the current LULC raster"),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="lulc_fut_aligned",
            path="intermediate/lulc_fut_aligned.tif",
            about=gettext("Aligned copy of the future LULC raster"),
            created_if="lulc_fut_path",
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="lulc_bas_aligned",
            path="intermediate/lulc_bas_aligned.tif",
            about=gettext("Aligned copy of the baseline LULC raster"),
            created_if="lulc_bas_path",
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="[THREAT]_aligned",
            path="intermediate/[THREAT]_aligned.tif",
            about=gettext("Aligned copy of each threat raster"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="filtered_[THREAT]_aligned",
            path="intermediate/filtered_[THREAT]_aligned.tif",
            about=gettext("Filtered threat raster"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="degradation_[THREAT]",
            path="intermediate/degradation_[THREAT].tif",
            about=gettext("Degradation raster for each threat"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="habitat_cur",
            path="intermediate/habitat_c.tif",
            about=gettext("Current habitat raster"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="habitat_fut",
            path="intermediate/habitat_f.tif",
            about=gettext("Future habitat raster"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="habitat_bas",
            path="intermediate/habitat_b.tif",
            about=gettext("Baseline habitat raster"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="[THREAT]_distance_transform_cur",
            path="intermediate/[THREAT]_distance_transform_c.tif",
            about=gettext("Map of distance to each threat in the current scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="[THREAT]_distance_transform_fut",
            path="intermediate/[THREAT]_distance_transform_f.tif",
            about=gettext("Map of distance to each threat in the future scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="[THREAT]_distance_transform_bas",
            path="intermediate/[THREAT]_distance_transform_b.tif",
            about=gettext("Map of distance to each threat in the baseline scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="filtered_[DECAY]_[THREAT]_cur",
            path="intermediate/filtered_[DECAY]_[THREAT]_c.tif",
            about=gettext("Decayed distance to each threat in the current scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="filtered_[DECAY]_[THREAT]_fut",
            path="intermediate/filtered_[DECAY]_[THREAT]_f.tif",
            about=gettext("Decayed distance to each threat in the future scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="filtered_[DECAY]_[THREAT]_bas",
            path="intermediate/filtered_[DECAY]_[THREAT]_b.tif",
            about=gettext("Decayed distance to each threat in the baseline scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="sens_[THREAT]_cur",
            path="intermediate/sens_[THREAT]_c.tif",
            about=gettext("Threat sensitivity in the current scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="sens_[THREAT]_fut",
            path="intermediate/sens_[THREAT]_f.tif",
            about=gettext("Threat sensitivity in the future scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="sens_[THREAT]_bas",
            path="intermediate/sens_[THREAT]_b.tif",
            about=gettext("Threat sensitivity in the baseline scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="degradation_[THREAT]_cur",
            path="intermediate/degradation_[THREAT]_c.tif",
            about=gettext("Degradation for each threat in the current scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="degradation_[THREAT]_fut",
            path="intermediate/degradation_[THREAT]_f.tif",
            about=gettext("Degradation for each threat in the future scenario"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="degradation_[THREAT]_bas",
            path="intermediate/degradation_[THREAT]_b.tif",
            about=gettext("Degradation for each threat in the baseline scenario"),
            data_type=float,
            units=None
        ),
        spec.TASKGRAPH_CACHE
    ]
)

# All out rasters besides rarity should be gte to 0. Set nodata accordingly.
_OUT_NODATA = float(numpy.finfo(numpy.float32).min)
# Scaling parameter from User's Guide eq. 4 for quality of habitat
_SCALING_PARAM = 2.5
# To help track and name threat rasters from paths in threat table columns
_THREAT_SCENARIO_MAP = {'cur': 'cur_path', 'fut': 'fut_path', 'bas': 'base_path'}


def execute(args):
    """Habitat Quality.

    This model calculates habitat degradation and quality for the current LULC
    as described in the InVEST user's guide. Optionally ``execute`` calculates
    habitat degradation and quality for a future LULC and habitat rarity for
    current and future LULC.

    Args:
        args (dict): a key, value mapping for the habitat quality inputs.
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files (required)
        args['lulc_cur_path'] (string): a path to an input land use/land
            cover raster (required)
        args['lulc_fut_path'] (string): a path to an input land use/land
            cover raster (optional)
        args['lulc_bas_path'] (string): a path to an input land use/land
            cover raster (optional, but required for rarity calculations)
        args['threats_table_path'] (string): a path to an input CSV
            containing data of all the considered threats. Each row is a
            degradation source and each column a different attribute of the
            source with the following names (case-insensitive):
            'THREAT','MAX_DIST','WEIGHT', 'DECAY', 'CUR_PATH' (required)
            'BASE_PATH', 'FUT_PATH' (optional).
        args['access_vector_path'] (string): a path to an input polygon
            shapefile containing data on the relative protection against
            threats (optional)
        args['sensitivity_table_path'] (string): a path to an input CSV file
            of LULC types, whether they are considered habitat, and their
            sensitivity to each threat (required)
        args['half_saturation_constant'] (float): a python float that
            determines the spread and central tendency of habitat quality
            scores (required)
        args['results_suffix'] (string): a python string that will be inserted
            into all raster path paths just before the file extension.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    LOGGER.info("Starting execute of Habitat Quality model.")
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    LOGGER.info("Checking Threat and Sensitivity tables for compliance")
    # Get CSVs as dictionaries and ensure the key is a string for threats.
    threat_df = MODEL_SPEC.get_input(
        'threats_table_path').get_validated_dataframe(
        args['threats_table_path']).fillna('')
    sensitivity_df = MODEL_SPEC.get_input(
        'sensitivity_table_path').get_validated_dataframe(
        args['sensitivity_table_path'])

    # Dictionary for reclassing habitat values
    sensitivity_reclassify_habitat_dict = sensitivity_df['habitat'].to_dict()

    # declare dictionaries to store the land cover and the threat rasters
    # pertaining to the different threats
    lulc_path_dict = {}
    threat_path_dict = collections.defaultdict(dict)
    # store land cover and threat rasters in a list for convenient access
    lulc_raster_list = []
    threat_raster_list = []
    aligned_lulc_raster_list = []
    aligned_threat_raster_list = []
    # list for checking threat values tasks
    threat_values_task_lookup = {}
    LOGGER.info("Validate threat rasters and collect unique LULC codes")
    # compile all the threat rasters associated with the land cover
    for scenario in ['cur', 'fut', 'bas']:
        lulc_arg = f'lulc_{scenario}_path'
        if args[lulc_arg]:
            # save land cover paths in a list for alignment and resize
            lulc_raster_list.append(args[lulc_arg])
            aligned_lulc_raster_list.append(file_registry[f'lulc_{scenario}_aligned'])
            lulc_path_dict[scenario] = file_registry[f'lulc_{scenario}_aligned']

            # for each threat given in the CSV file try opening the associated
            # raster which should be found relative to the Threat CSV
            for threat, row in threat_df.iterrows():
                LOGGER.debug(f"Validating path for threat: {threat}")

                threat_path = row[_THREAT_SCENARIO_MAP[scenario]]
                if not threat_path:
                    if scenario == 'bas':
                        threat_path = None
                    else:
                        raise ValueError(
                            'There was an Error locating a threat raster from '
                            'the path in CSV for column: '
                            f'{_THREAT_SCENARIO_MAP[scenario]} and threat: '
                            f'{threat}.')

                # save threat paths in a list for alignment and resize
                if threat_path:
                    # check for duplicate absolute threat path names that
                    # cause errors when trying to write aligned versions
                    if (threat_path not in threat_raster_list):
                        threat_raster_list.append(threat_path)
                    else:
                        raise ValueError(
                            DUPLICATE_PATHS_MSG + os.path.basename(threat_path)
                        )
                    # Check threat raster values are 0 <= x <= 1
                    threat_values_task = task_graph.add_task(
                        func=_raster_values_in_bounds,
                        args=((threat_path, 1), 0.0, 1.0),
                        store_result=True,
                        task_name=f'check_threat_values_{scenario[0]}_{threat}')
                    threat_values_task_lookup[threat_values_task.task_name] = {
                        'task': threat_values_task,
                        'path': threat_path,
                        'table_col': _THREAT_SCENARIO_MAP[scenario]}

                    # create paths for aligned rasters checking for the case the raster path
                    # is a folder
                    if os.path.isdir(threat_path):
                        threat_name = os.path.basename(os.path.dirname(threat_path))
                    else:
                        threat_name = os.path.splitext(os.path.basename(threat_path))[0]
                    aligned_threat_raster_list.append(file_registry['[THREAT]_aligned', threat_name])
                    threat_path_dict[scenario][threat] = file_registry['[THREAT]_aligned', threat_name]

    LOGGER.info("Checking threat raster values are valid ( 0 <= x <= 1 ).")
    # Assert that threat rasters have valid values.
    for values in threat_values_task_lookup.values():
        # get returned boolean to see if values were valid
        valid_threat_values = values['task'].get()
        if not valid_threat_values:
            raise ValueError(
                "Threat rasters should have values between 0 and 1, however,"
                f"Threat: {values['path']} for column: {values['table_col']}",
                " had values outside of this range.")

    LOGGER.info(
        'Aligning, resizing, and reprojecting raster inputs to that of the'
        ' current land cover.')
    lulc_raster_info = pygeoprocessing.get_raster_info(args['lulc_cur_path'])
    # ensure that the pixel size used is square
    lulc_pixel_size = lulc_raster_info['pixel_size']
    min_pixel_size = min([abs(x) for x in lulc_pixel_size])
    pixel_size = (min_pixel_size, -min_pixel_size)
    lulc_bbox = lulc_raster_info['bounding_box']
    lulc_wkt = lulc_raster_info['projection_wkt']

    LOGGER.debug(f"Raster paths for aligning: {aligned_lulc_raster_list + aligned_threat_raster_list}")
    # Align and resize all the land cover and threat rasters,
    # and store them in the intermediate folder
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        kwargs={
            'base_raster_path_list': lulc_raster_list + threat_raster_list,
            'target_raster_path_list': aligned_lulc_raster_list + aligned_threat_raster_list,
            'resample_method_list': ['near']*len(lulc_raster_list + threat_raster_list),
            'target_pixel_size': pixel_size,
            'bounding_box_mode': lulc_bbox,
            'target_projection_wkt': lulc_wkt},
        target_path_list=aligned_lulc_raster_list + aligned_threat_raster_list,
        task_name='align_input_rasters')

    LOGGER.info('Starting habitat_quality biophysical calculations')
    # Rasterize access vector, if value is null set to 1 (fully accessible),
    # else set to the value according to the ACCESS attribute
    # create a new raster based on the raster info of current land cover.
    # fill with 1.0 for case where no access shapefile provided,
    # which indicates we don't want to mask anything out later
    create_access_raster_task = task_graph.add_task(
        func=pygeoprocessing.new_raster_from_base,
        args=(lulc_path_dict['cur'], file_registry['access_layer'], gdal.GDT_Float32,
              [_OUT_NODATA]),
        kwargs={
            'fill_value_list': [1.0]
        },
        target_path_list=[file_registry['access_layer']],
        dependent_task_list=[align_task],
        task_name='access_raster')
    access_task_list = [create_access_raster_task]

    if args['access_vector_path']:
        LOGGER.debug("Reproject and rasterize Access vector")
        reproject_access_task = task_graph.add_task(
            func=pygeoprocessing.reproject_vector,
            kwargs={
                'base_vector_path': args['access_vector_path'],
                'target_projection_wkt': lulc_wkt,
                'target_path': file_registry['reprojected_access_vector'],
                'driver_name': 'GPKG'
            },
            target_path_list=[file_registry['reprojected_access_vector']],
            task_name='reproject_access_vector')

        rasterize_access_task = task_graph.add_task(
            func=pygeoprocessing.rasterize,
            args=(file_registry['reprojected_access_vector'], file_registry['access_layer']),
            kwargs={
                'option_list': ['ATTRIBUTE=ACCESS'],
                'burn_values': None
            },
            target_path_list=[file_registry['access_layer']],
            dependent_task_list=[
                create_access_raster_task, reproject_access_task],
            task_name='rasterize_access')
        access_task_list.append(rasterize_access_task)

    # calculate the weight sum which is the sum of all the threats' weights
    weight_sum = threat_df['weight'].sum()

    # for each land cover raster provided compute habitat quality
    for scenario, lulc_path in lulc_path_dict.items():
        LOGGER.info(f'Calculating habitat quality for landuse: {lulc_path}')

        threat_decay_task_list = []
        sensitivity_task_list = []
        individual_degradation_task_list = []

        # Create raster of habitat based on habitat field
        reclass_error_details = {
            'raster_name': f'LULC_{scenario[0]}', 'column_name': 'lucode',
            'table_name': 'Sensitivity'}
        habitat_raster_task = task_graph.add_task(
            func=utils.reclassify_raster,
            args=((lulc_path, 1), sensitivity_reclassify_habitat_dict,
                  file_registry[f'habitat_{scenario}'], gdal.GDT_Float32, _OUT_NODATA,
                  reclass_error_details),
            dependent_task_list=[align_task],
            task_name=f'habitat_raster_{scenario[0]}')

        # initialize a list that will store all the degradation rasters for each threat
        # after they have been adjusted for distance, weight, and access
        indiv_deg_raster_list = []

        # variable to indicate whether we should break out of calculations
        # for a land cover because a threat raster was not found
        exit_landcover = False

        # adjust each threat/threat raster for distance, weight, and access
        for threat, row in threat_df.iterrows():
            LOGGER.debug(
                f'Calculating threat: {threat}.\nThreat data: {row}')

            # get the threat raster for the specific threat
            threat_raster_path = threat_path_dict[scenario].get(threat, None)
            # if threat path is None then must be in Base scenario where
            # threats are not required.
            if threat_raster_path is None:
                LOGGER.warning(
                    f'The threat raster for {threat} could not be found for'
                    f' the land cover {scenario}. Skipping Habitat Quality'
                    ' calculation for this land cover.')
                exit_landcover = True
                break
            # Check to make sure max_dist is greater than 0
            if row['max_dist'] <= 0:
                raise ValueError(
                    f"The max distance for threat: '{threat}' is less than"
                    " or equal to 0. MAX_DIST should be a positive value.")

            dist_edt_task = task_graph.add_task(
                func=pygeoprocessing.distance_transform_edt,
                args=(
                    (threat_raster_path, 1),
                    file_registry[f'[THREAT]_distance_transform_{scenario}', threat]),
                target_path_list=[file_registry[f'[THREAT]_distance_transform_{scenario}', threat]],
                dependent_task_list=[align_task],
                task_name=f'distance edt {scenario} {threat}')

            dist_decay_task = task_graph.add_task(
                func=_decay_distance,
                args=(
                    file_registry[f'[THREAT]_distance_transform_{scenario}', threat],
                    row['max_dist'], row['decay'], file_registry[
                        f'filtered_[DECAY]_[THREAT]_{scenario}', row["decay"], threat]),
                target_path_list=[file_registry[f'filtered_[DECAY]_[THREAT]_{scenario}',
                    row["decay"], threat]],
                dependent_task_list=[dist_edt_task],
                task_name=f'distance decay {scenario} {threat}')
            threat_decay_task_list.append(dist_decay_task)

            # create sensitivity raster based on threat
            # Dictionary for reclassing threat sensitivity values
            sensitivity_reclassify_threat_dict = sensitivity_df[threat].to_dict()

            reclass_error_details = {
                'raster_name': 'LULC', 'column_name': 'lucode',
                'table_name': 'Sensitivity'}
            sens_threat_task = task_graph.add_task(
                func=utils.reclassify_raster,
                args=((lulc_path, 1), sensitivity_reclassify_threat_dict,
                      file_registry[f'sens_[THREAT]_{scenario}', threat],
                      gdal.GDT_Float32, _OUT_NODATA,
                      reclass_error_details),
                target_path_list=[file_registry[f'sens_[THREAT]_{scenario}', threat]],
                dependent_task_list=[align_task],
                task_name=f'sens_raster_{row["decay"]}_{scenario[0]}_{threat}')
            sensitivity_task_list.append(sens_threat_task)

            # get the normalized weight for the threat
            # which is used to calculate threat degradation
            weight_avg = row['weight'] / weight_sum

            # Calculate degradation for each threat
            individual_threat_task = task_graph.add_task(
                func=_calculate_individual_degradation,
                args=(
                    file_registry[f'filtered_[DECAY]_[THREAT]_{scenario}',
                        row["decay"], threat],
                    file_registry[f'sens_[THREAT]_{scenario}', threat],
                    weight_avg,
                    file_registry['access_layer'],
                    file_registry[f'degradation_[THREAT]_{scenario}', threat]),
                target_path_list=[file_registry[f'degradation_[THREAT]_{scenario}', threat]],
                dependent_task_list=[
                    sens_threat_task, dist_decay_task,
                    *access_task_list],
                task_name=f'deg raster {scenario} {threat}')
            individual_degradation_task_list.append(individual_threat_task)

            indiv_deg_raster_list.append(file_registry[
                f'degradation_[THREAT]_{scenario}', threat])

        # check to see if we got here because a threat raster was missing
        # for baseline lulc, if so then we want to skip to the next landcover
        if exit_landcover:
            continue

        LOGGER.info('Starting raster calculation on total degradation')

        total_degradation_task = task_graph.add_task(
            func=_calculate_total_degradation,
            args=(indiv_deg_raster_list, file_registry[f'deg_sum_{scenario[0]}']),
            target_path_list=[file_registry[f'deg_sum_{scenario[0]}']],
            dependent_task_list=individual_degradation_task_list,
            task_name=f'tot_degradation_{row["decay"]}{scenario}_{threat}')

        # Compute habitat quality
        # ksq: a term used below to compute habitat quality
        ksq = args['half_saturation_constant']**_SCALING_PARAM

        LOGGER.info('Starting raster calculation on quality')

        deg_hab_raster_list = [
            file_registry[f'deg_sum_{scenario[0]}'],
            file_registry[f'habitat_{scenario}']]

        _ = task_graph.add_task(
            func=_calculate_habitat_quality,
            args=(deg_hab_raster_list, file_registry[f'quality_{scenario[0]}'], ksq),
            target_path_list=[file_registry[f'quality_{scenario[0]}']],
            dependent_task_list=[habitat_raster_task, total_degradation_task],
            task_name='habitat_quality')

    # Compute Rarity if user supplied baseline raster
    if 'bas' not in lulc_path_dict:
        LOGGER.info('Baseline not provided to compute Rarity')
    else:
        # compute rarity for current landscape and future (if provided)
        for scenario in ['cur', 'fut']:
            if scenario not in lulc_path_dict:
                continue
            _ = task_graph.add_task(
                func=_compute_rarity_operation,
                args=((lulc_path_dict['bas'], 1), (lulc_path_dict[scenario], 1),
                      (file_registry[f'new_cover_{scenario[0]}'], 1),
                      file_registry[f'rarity_{scenario[0]}'],
                      file_registry[f'rarity_{scenario[0]}_csv']),
                dependent_task_list=[align_task],
                task_name=f'rarity_{scenario}')

    task_graph.close()
    task_graph.join()
    LOGGER.info("Habitat Quality Model complete.")
    return file_registry.registry


def _calculate_habitat_quality(deg_hab_raster_list, quality_out_path, ksq):
    """Calculate habitat quality from degradation inputs.

    Args:
        deg_hab_raster_list (list): list of string paths for the degraded
            habitat rasters.
        quality_out_path (string): path to output the habitat quality raster.
        ksq (float): a number representing half-saturation**_SCALING_PARAM

    Returns:
        None
    """
    pygeoprocessing.raster_map(
        op=lambda degradation, habitat: (
            habitat * (1 - (degradation**_SCALING_PARAM) /
                       (degradation**_SCALING_PARAM + ksq))),
        rasters=deg_hab_raster_list,
        target_path=quality_out_path)


def _calculate_individual_degradation(
        filtered_threat_raster_path, sens_raster_path, weight_avg, access_raster_path,
        indiv_threat_raster_path):
    """Calculate habitat degradation for a given threat.
    Args:
        filtered_threat_raster_path (string): path for the filtered threat raster
        sens_raster_path (string): path for the sensitivity raster
        weight_avg (float): normalized weight for the threat
        access_raster_path (string): path for the access raster
        indiv_threat_raster_path (string): path to output the habitat quality
            degradation raster for the individual threat.

    Returns:
        None
    """
    def degradation(filtered_threat_array, sens_array, access_array):
        """Computes the degradation value for a given threat.

        Args:
            filtered_threat_array (numpy.ndarray): filtered threat raster values
            sens_array (numpy.ndarray): sensitivity raster values
            access_array (numpy.ndarray): access raster based on values from the
                shapefile if provided or a raster filled with all 1s if not

        Returns:
            The degradation score for the pixel for the individual threat.
        """
        return filtered_threat_array * sens_array * weight_avg * access_array

    pygeoprocessing.raster_map(
        op=degradation,
        rasters=[filtered_threat_raster_path, sens_raster_path, access_raster_path],
        target_path=indiv_threat_raster_path)


def _calculate_total_degradation(indiv_deg_raster_list, deg_sum_raster_path):
    """Calculate total habitat degradation.

    Args:
        indiv_deg_raster_list (list): list of string paths for the degraded
            threat rasters for each threat.
        deg_sum_raster_path (string): path to output the habitat quality
            degradation raster.

    Returns:
        None
    """
    pygeoprocessing.raster_map(
        op=lambda *indiv_arrays: numpy.sum(indiv_arrays, axis=0),
        rasters=indiv_deg_raster_list,
        target_path=deg_sum_raster_path)


def _compute_rarity_operation(
        base_lulc_path_band, lulc_path_band, new_cover_path,
        rarity_raster_path, rarity_csv_path):
    """Calculate habitat rarity and generate raster and CSV output.

    Output rarity values will be an index from 0 - 1 where:
       pixel > 0.5 - more rare
       pixel < 0.5 - less rare
       pixel = 0.5 - no rarity change
       pixel = 0.0 - LULC not found in the baseline for comparison

    Args:
        base_lulc_path_band (tuple): a 2 tuple for the path to input base
            LULC raster of the form (path, band index).
        lulc_path_band (tuple):  a 2 tuple for the path to LULC for current
            or future scenario of the form (path, band index).
        new_cover_path (tuple): a 2 tuple for the path to intermediate
            raster file for trimming ``lulc_path_band`` to
            ``base_lulc_path_band`` of the form (path, band index).
        rarity_raster_path (string): path to output rarity raster.
        rarity_csv_path (string): path to output rarity CSV.

    Returns:
        None
    """
    # get the area of a base pixel to use for computing rarity where the
    # pixel sizes are different between base and cur/fut rasters
    base_raster_info = pygeoprocessing.get_raster_info(
        base_lulc_path_band[0])
    base_pixel_size = base_raster_info['pixel_size']
    base_area = float(abs(base_pixel_size[0]) * abs(base_pixel_size[1]))

    lulc_code_count_b = _raster_pixel_count(base_lulc_path_band)

    # get the area of a cur/fut pixel
    lulc_raster_info = pygeoprocessing.get_raster_info(lulc_path_band[0])
    lulc_pixel_size = lulc_raster_info['pixel_size']
    lulc_area = float(abs(lulc_pixel_size[0]) * abs(lulc_pixel_size[1]))

    # Trim cover_x to the mask of base.
    pygeoprocessing.raster_map(
        op=lambda base, cover_x: cover_x,
        rasters=[base_lulc_path_band[0], lulc_path_band[0]],
        target_path=new_cover_path[0])

    LOGGER.info('Starting rarity computation on'
                f' {os.path.basename(lulc_path_band[0])} land cover.')

    lulc_code_count_x = _raster_pixel_count(new_cover_path)

    # a dictionary to map LULC types to a number that depicts how
    # rare they are considered
    code_index = {}

    # compute rarity index for each lulc code
    # define 0.0 if an lulc code is found in the cur/fut landcover
    # but not the baseline
    for code in lulc_code_count_x:
        if code in lulc_code_count_b:
            numerator = lulc_code_count_x[code] * lulc_area
            denominator = lulc_code_count_b[code] * base_area
            ratio = 1.0 - (numerator / (denominator + numerator))
            code_index[code] = ratio
        else:
            code_index[code] = 0.0

    pygeoprocessing.reclassify_raster(
        new_cover_path, code_index, rarity_raster_path, gdal.GDT_Float32,
        _OUT_NODATA)

    _generate_rarity_csv(code_index, rarity_csv_path)

    LOGGER.info('Finished rarity computation on'
                f' {os.path.basename(lulc_path_band[0])} land cover.')


def _generate_rarity_csv(rarity_dict, target_csv_path):
    """Generate CSV containing rarity values by LULC code.

    Args:
        rarity_dict (dict): dictionary containing LULC codes (as keys)
            and their associated rarity values (as values).
        target_csv_path (string): path to output CSV.

    Returns:
        None
    """
    lulc_codes = sorted(rarity_dict)
    with open(target_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['lucode', 'rarity_value'])
        for lulc_code in lulc_codes:
            writer.writerow([lulc_code, rarity_dict[lulc_code]])


def _raster_pixel_count(raster_path_band):
    """Count unique pixel values in single band raster.

    Args:
        raster_path_band (tuple): a 2 tuple of the form
            (filepath to raster, band index) where the raster has a single
            band.

    Returns:
        dict of pixel values to frequency.
    """
    nodata = pygeoprocessing.get_raster_info(
        raster_path_band[0])['nodata'][raster_path_band[1]-1]

    counts = collections.defaultdict(int)
    for _, raster_block in pygeoprocessing.iterblocks(raster_path_band):
        for value, count in zip(
                *numpy.unique(raster_block, return_counts=True)):
            if value == nodata:
                continue
            counts[value] += count
    return counts


def _raster_values_in_bounds(raster_path_band, lower_bound, upper_bound):
    """Check raster values are between ``lower_bound`` and ``upper_bound``.

    Check that the raster has values ``lower_bound`` <= x <= ``upper_bound``.
    Nodata values are ignored.

    Args:
        raster_path_band (tuple): a 2 tuple for a GDAL raster path with
            the form (filepath, band index) to the raster on disk.
        lower_bound (int): integer for the lower bound of raster values,
            inclusive.
        upper_bound (int): integer for the upper bound of raster values,
            inclusive.

    Returns:
        True if values are within range and False otherwise.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path_band[0])
    raster_nodata = raster_info['nodata'][0]
    values_valid = True

    for _, raster_block in pygeoprocessing.iterblocks(raster_path_band):
        nodata_mask = ~pygeoprocessing.array_equals_nodata(raster_block, raster_nodata)
        if ((raster_block[nodata_mask] < lower_bound) |
                (raster_block[nodata_mask] > upper_bound)).any():
            values_valid = False
            break

    return values_valid


def _decay_distance(dist_raster_path, max_dist, decay_type, target_path):
    """Apply an exponential or linear decay to a distance transform raster.

    The function will set pixels greater than ``max_dist`` to 0.

    Args:
        dist_raster_path (string): a filepath for the raster to decay.
            The raster is expected to be a euclidean distance transform with
            values measuring distance in pixels.
        max_dist (float): max distance of threat in meters.
        decay_type (string): a string defining which decay method to use.
            Options include: 'linear' | 'exponential'.
        target_path (string): a filepath for a float output raster.

    Returns:
        None
    """
    # get raster pixel size to determine how many pixels max_dist covers
    threat_pixel_size = pygeoprocessing.get_raster_info(
        dist_raster_path)['pixel_size']

    # convert max distance from meters to the number of pixels that
    # represents on the raster
    max_dist_pixel = max_dist / abs(threat_pixel_size[0])
    LOGGER.debug(f'Max distance in pixels: {max_dist_pixel}')

    def linear_op(dist):
        """Linear decay operation."""
        return numpy.where(
            dist > max_dist_pixel, 0,
            (max_dist_pixel - dist) / max_dist_pixel)

    def exp_op(dist):
        """Exponential decay operation."""
        # Some background on where the 2.99 constant comes from:
        # With the constant of 2.99, the impact of the threat is reduced by
        # 95% (to 5%) at the specified max threat distance. So I suspect it's
        # based on the traditional 95% cutoff that is used in statistics. We
        # could tweak this cutoff (e.g., 99% decay at max distance), if we
        # wanted. - Lisa Mandle
        return numpy.where(
            dist > max_dist_pixel, 0,
            numpy.exp((-dist * 2.99) / max_dist_pixel))

    if decay_type == 'linear':
        decay_op = linear_op
    elif decay_type == 'exponential':
        decay_op = exp_op
    else:
        raise ValueError(
            "Unknown type of decay in threat table, should be"
            f" either 'linear' or 'exponential'. Input was '{decay_type}' for"
            f" output raster path : '{target_path}'")

    pygeoprocessing.raster_map(
        op=decay_op,
        rasters=[dist_raster_path],
        target_path=target_path)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire ``args`` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """
    validation_warnings = validation.validate(args, MODEL_SPEC)

    invalid_keys = validation.get_invalid_keys(validation_warnings)

    if ("threats_table_path" not in invalid_keys and
            "sensitivity_table_path" not in invalid_keys and
            "threat_raster_folder" not in invalid_keys):
        # Get CSVs as dictionaries and ensure the key is a string for threats.
        threat_df = MODEL_SPEC.get_input(
            'threats_table_path').get_validated_dataframe(
            args['threats_table_path']).fillna('')
        sensitivity_df = MODEL_SPEC.get_input(
            'sensitivity_table_path').get_validated_dataframe(
            args['sensitivity_table_path'])

        # check that the threat names in the threats table match with the
        # threats columns in the sensitivity table.
        sens_header_set = set(sensitivity_df.columns)
        threat_set = set(threat_df.index.values)
        missing_sens_header_set = threat_set.difference(sens_header_set)

        if missing_sens_header_set:
            validation_warnings.append(
                (['sensitivity_table_path'],
                 MISSING_SENSITIVITY_TABLE_THREATS_MSG.format(
                    threats=missing_sens_header_set,
                    column_names=sens_header_set)))

            invalid_keys.add('sensitivity_table_path')

        # check that max_dist and weight values are included in the
        # threats table and that max_dist >= 0
        invalid_max_dist = []
        missing_max_dist = []
        missing_weight = []
        for threat, row in threat_df.iterrows():
            if row['max_dist'] == '':
                missing_max_dist.append(threat)
            elif row['max_dist'] <= 0:
                invalid_max_dist.append(threat)

            if row['weight'] == '':
                missing_weight.append(threat)

        if invalid_max_dist:
            validation_warnings.append((
                ['threats_table_path'],
                INVALID_MAX_DIST_MSG.format(threat_list=invalid_max_dist)
            ))

        if missing_max_dist:
            validation_warnings.append((
                ['threats_table_path'],
                MISSING_MAX_DIST_MSG.format(threat_list=missing_max_dist)
            ))

        if missing_weight:
            validation_warnings.append((
                ['threats_table_path'],
                MISSING_WEIGHT_MSG.format(threat_list=missing_weight)
            ))

        if invalid_max_dist or missing_max_dist or missing_weight:
            invalid_keys.add('threats_table_path')

        # Validate threat raster paths and their nodata values
        bad_threat_paths = []
        duplicate_paths = []
        threat_path_list = []
        for scenario in ['cur', 'bas', 'fut']:
            lulc_arg = f'lulc_{scenario}_path'
            if lulc_arg in args and args[lulc_arg] != '':
                # for each threat given in the CSV file try opening the
                # associated raster which should be found in
                # threat_raster_folder
                for threat, row in threat_df.iterrows():
                    # Threat path from threat CSV is relative to CSV
                    threat_path = row[_THREAT_SCENARIO_MAP[scenario]]

                    if not threat_path:
                        if scenario != 'bas':
                            bad_threat_paths.append(
                                (threat, _THREAT_SCENARIO_MAP[scenario]))
                        continue

                    # check for duplicate absolute threat path names that
                    # cause errors when trying to write aligned versions
                    if threat_path not in threat_path_list:
                        threat_path_list.append(threat_path)
                    else:
                        duplicate_paths.append(
                            os.path.basename(threat_path))

        if bad_threat_paths:
            validation_warnings.append((
                ['threats_table_path'],
                MISSING_THREAT_RASTER_MSG.format(threat_list=bad_threat_paths)
            ))
            invalid_keys.add('threats_table_path')

        if duplicate_paths:
            validation_warnings.append((
                ['threats_table_path'],
                DUPLICATE_PATHS_MSG + str(duplicate_paths)))
            invalid_keys.add('threats_table_path')

    return validation_warnings
