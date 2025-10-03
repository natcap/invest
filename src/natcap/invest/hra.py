"""Habitat risk assessment (HRA) model for InVEST."""
# -*- coding: UTF-8 -*-
import collections
import itertools
import logging
import math
import os
import re
import shutil

import numpy
import pandas
import pygeoprocessing
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

# RESILIENCE stressor shorthand to use when parsing tables
_RESILIENCE_STRESSOR = 'resilience'

# Target cell type or values for raster files.
_TARGET_GDAL_TYPE_FLOAT32 = gdal.GDT_Float32
_TARGET_GDAL_TYPE_BYTE = gdal.GDT_Byte
_TARGET_NODATA_FLOAT32 = float(numpy.finfo(numpy.float32).min)
_TARGET_NODATA_BYTE = 255  # for unsigned 8-bit int

# ESPG code for warping rasters to WGS84 coordinate system.
_WGS84_SRS = osr.SpatialReference()
_WGS84_SRS.ImportFromEPSG(4326)
_WGS84_WKT = _WGS84_SRS.ExportToWkt()

# Resampling method for rasters.
_RESAMPLE_METHOD = 'near'

# An argument list that will be passed to the GTiff driver. Useful for
# blocksizes, compression, and more.
_DEFAULT_GTIFF_CREATION_OPTIONS = (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256')

MODEL_SPEC = spec.ModelSpec(
    model_id="habitat_risk_assessment",
    model_title=gettext("Habitat Risk Assessment"),
    userguide="habitat_risk_assessment.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=("hra",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["info_table_path", "criteria_table_path"],
        ["resolution", "max_rating"],
        ["risk_eq", "decay_eq"],
        ["aoi_vector_path"],
        ["n_overlapping_stressors"],
        ["visualize_outputs"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.CSVInput(
            id="info_table_path",
            name=gettext("habitat stressor table"),
            about=gettext("A table describing each habitat and stressor."),
            columns=[
                spec.StringInput(
                    id="name",
                    about=gettext(
                        "A unique name for each habitat or stressor. These names must"
                        " match the habitat and stressor names in the Criteria Scores"
                        " Table."
                    ),
                    regexp=None
                ),
                spec.RasterOrVectorInput(
                    id="path",
                    about=gettext(
                        "Map of where the habitat or stressor exists. For rasters, a"
                        " pixel value of 1 indicates presence of the habitat or stressor."
                        " 0 (or any other value) indicates absence of the habitat or"
                        " stressor. For vectors, a geometry indicates an area where the"
                        " habitat or stressor is present."
                    ),
                    data_type=float,
                    units=u.none,
                    geometry_types={
                        "LINESTRING",
                        "POLYGON",
                        "MULTIPOLYGON",
                        "MULTIPOINT",
                        "POINT",
                        "MULTILINESTRING",
                    },
                    fields=[],
                    projected=None
                ),
                spec.OptionStringInput(
                    id="type",
                    about=gettext("Whether this row is for a habitat or a stressor."),
                    options=[
                        spec.Option(key="habitat"),
                        spec.Option(key="stressor")
                    ]
                ),
                spec.NumberInput(
                    id="stressor buffer (meters)",
                    about=gettext(
                        "The desired buffer distance used to expand a given stressor’s"
                        " influence or footprint. This should be left blank for habitats,"
                        " but must be filled in for stressors. Enter 0 if no buffering is"
                        " desired for a given stressor. The model will round down this"
                        " buffer distance to the nearest cell unit. e.g., a buffer"
                        " distance of 600m will buffer a stressor’s footprint by two grid"
                        " cells if the resolution of analysis is 250m."
                    ),
                    units=u.meter
                )
            ],
            index_col="name"
        ),
        spec.CSVInput(
            id="criteria_table_path",
            name=gettext("criteria scores table"),
            about=gettext("A table of criteria scores for all habitats and stressors."),
            columns=None,
            index_col=None
        ),
        spec.NumberInput(
            id="resolution",
            name=gettext("resolution of analysis"),
            about=gettext(
                "The resolution at which to run the analysis. The model outputs will have"
                " this resolution."
            ),
            units=u.meter,
            expression="value > 0"
        ),
        spec.NumberInput(
            id="max_rating",
            name=gettext("maximum criteria score"),
            about=gettext("The highest possible criteria score in the scoring system."),
            units=u.none,
            expression="value > 0"
        ),
        spec.OptionStringInput(
            id="risk_eq",
            name=gettext("risk equation"),
            about=gettext(
                "The equation to use to calculate risk from exposure and consequence."
            ),
            options=[
                spec.Option(key="multiplicative", display_name="Multiplicative"),
                spec.Option(key="euclidean", display_name="Euclidean")
            ]
        ),
        spec.OptionStringInput(
            id="decay_eq",
            name=gettext("decay equation"),
            about=gettext("The equation to model effects of stressors in buffer areas."),
            options=[
                spec.Option(
                    key="none",
                    about="No decay. Stressor has full effect in the buffer area."),
                spec.Option(
                    key="linear",
                    about=(
                        "Stressor effects in the buffer area decay linearly with distance"
                        " from the stressor.")),
                spec.Option(
                    key="exponential",
                    about=(
                        "Stressor effects in the buffer area decay exponentially with"
                        " distance from the stressor."))
            ]
        ),
        spec.AOI.model_copy(update=dict(
            id="aoi_vector_path",
            about=gettext(
                "A GDAL-supported vector file containing features representing one or"
                " more planning regions or subregions."
            ),
            fields=[
                spec.StringInput(
                    id="name",
                    about=gettext(
                        "Uniquely identifies each feature. Required if the vector"
                        " contains more than one feature."
                    ),
                    required=False,
                    regexp=None
                )
            ],
            projected=True,
            projection_units=u.meter
        )),
        spec.NumberInput(
            id="n_overlapping_stressors",
            name=gettext("Number of Overlapping Stressors"),
            about=(
                "The number of overlapping stressors to consider as 'maximum' when"
                " reclassifying risk scores into high/medium/low.  Affects the breaks"
                " between risk classifications."
            ),
            units=u.none,
            expression="value > 0"
        ),
        spec.BooleanInput(
            id="visualize_outputs",
            name=gettext("Generate GeoJSONs"),
            about=gettext("Generate GeoJSON outputs for web visualization."),
            required=False
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="total_risk_[HABITAT]",
            path="outputs/TOTAL_RISK_[HABITAT].tif",
            about=gettext(
                "Habitat-specific cumulative risk from all the stressors"
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="total_risk_ecosystem",
            path="outputs/TOTAL_RISK_Ecosystem.tif",
            about=gettext(
                "Sum of habitat cumulative risk scores divided by the number of"
                " habitats occurring in each cell."
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="reclass_risk_[HABITAT]",
            path="outputs/RECLASS_RISK_[HABITAT].tif",
            about=gettext(
                "Reclassified habitat-specific risk from all the stressors in a"
                " grid cell into four categories, where 0 = No Risk, 1 = Low"
                " Risk, 2 = Medium Risk, and 3 = High Risk."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="reclass_risk_ecosystem",
            path="outputs/RECLASS_RISK_Ecosystem.tif",
            about=gettext("Reclassified ecosystem risk in each cell."),
            data_type=int,
            units=None
        ),
        spec.CSVOutput(
            id="summary_statistics",
            path="outputs/SUMMARY_STATISTICS.csv",
            about=gettext(
                "Table of summary statistics for each combination of habitat,"
                " stressor, and subregion"
            ),
            columns=[
                spec.StringOutput(id="HABITAT", about=gettext("Habitat name")),
                spec.StringOutput(id="STRESSOR", about=gettext("Stressor name")),
                spec.StringOutput(
                    id="SUBREGION", about=gettext("Subregion name")
                ),
                spec.NumberOutput(
                    id="E_MEAN",
                    about=gettext("Mean exposure score"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="E_MIN",
                    about=gettext("Minimum exposure score"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="E_MAX",
                    about=gettext("Maximum exposure score"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="C_MEAN",
                    about=gettext("Mean consequence score"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="C_MIN",
                    about=gettext("Minimum consequence score"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="C_MAX",
                    about=gettext("Maximum consequence score"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="R_MEAN", about=gettext("Mean risk score"), units=u.none
                ),
                spec.NumberOutput(
                    id="R_MIN", about=gettext("Minimum risk score"), units=u.none
                ),
                spec.NumberOutput(
                    id="R_MAX", about=gettext("Maximum risk score"), units=u.none
                ),
                spec.PercentOutput(
                    id="R_%HIGH",
                    about=gettext("the percentage of high risk areas.")
                ),
                spec.PercentOutput(
                    id="R_%MEDIUM",
                    about=gettext("the percentage of medium risk areas.")
                ),
                spec.PercentOutput(
                    id="R_%LOW",
                    about=gettext("the percentage of low risk areas.")
                )
            ],
            index_col=None
        ),
        spec.VectorOutput(
            id="reclass_risk_[HABITAT]_viz",
            path="visualization_outputs/RECLASS_RISK_[HABITAT].geojson",
            about=gettext(
                "Map of habitat-specific risk visualized in gradient color from"
                " white to red on a map."
            ),
            created_if="visualize_outputs",
            geometry_types={"POLYGON"},
            fields=[
                spec.IntegerOutput(
                    id="Risk Score",
                    about=gettext(
                        "Habitat risk from all stressors where 0 = No Risk, 1 ="
                        " Low Risk, 2 = Medium Risk, and 3 = High Risk."
                    )
                )
            ]
        ),
        spec.VectorOutput(
            id="reclass_risk_ecosystem_viz",
            path="visualization_outputs/RECLASS_RISK_Ecosystem.geojson",
            about=gettext(
                "Map of ecosystem risk visualized in gradient color from white to"
                " red on a map."
            ),
            created_if="visualize_outputs",
            geometry_types={"POLYGON"},
            fields=[
                spec.IntegerOutput(
                    id="Risk Score",
                    about=gettext(
                        "Ecosystem risk from all stressors where 0 = No Risk, 1 ="
                        " Low Risk, 2 = Medium Risk, and 3 = High Risk."
                    )
                )
            ]
        ),
        spec.VectorOutput(
            id="stressor_[STRESSOR]_viz",
            path="visualization_outputs/STRESSOR_[STRESSOR].geojson",
            about=gettext("Map of stressor extent visualized in orange color."),
            created_if="visualize_outputs",
            geometry_types={"POLYGON"},
            fields=[]
        ),
        spec.CSVOutput(
            id="summary_statistics_viz",
            path="visualization_outputs/SUMMARY_STATISTICS.csv",
            about=gettext(
                "This is the same file from one in the Output Folder. It is"
                " copied here so users can just upload the visualization outputs"
                " folder to the HRA web application, with all the files in one"
                " place."
            ),
            created_if="visualize_outputs",
            columns=[],
            index_col=None
        ),
        spec.SingleBandRasterOutput(
            id="aligned_[KEY]",
            path="intermediate_outputs/aligned_[KEY].tif",
            about=gettext(
                "A copy of each habitat, stressor, and criteria input, aligned "
                "to the same projection and extent"),
            data_type=float,
            units=u.none
        ),
        spec.VectorOutput(
            id="simplified_aoi",
            path="intermediate_outputs/simplified_aoi.gpkg",
            about=gettext("Simplified version of the input AOI vector"),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="c_[HABITAT]_[STRESSOR]",
            path="intermediate_outputs/C_[HABITAT]_[STRESSOR].tif",
            about=gettext(
                "Consequence score for a particular habitat/stressor combination."
            ),
            data_type=float,
            units=u.none
        ),
        spec.CSVOutput(
            id="composite_criteria",
            path="intermediate_outputs/composite_criteria.csv",
            about=gettext(
                "Table tracking each combination of habitat, stressor, criterion,"
                " rating, data quality, weight and whether the score applies to"
                " exposure or consequence."
            ),
            columns=[
                spec.StringOutput(id="habitat", about=gettext("Habitat name")),
                spec.StringOutput(id="stressor", about=gettext("Stressor name")),
                spec.StringOutput(
                    id="criterion", about=gettext("Criterion name")
                ),
                spec.StringOutput(
                    id="rating", about=gettext("Rating value or path to raster")
                ),
                spec.IntegerOutput(id="dq", about=gettext("Data quality")),
                spec.IntegerOutput(id="weight", about=gettext("Weight")),
                spec.StringOutput(
                    id="e/c", about=gettext("Exposure (E) or consequence (C)")
                )
            ],
            index_col=None
        ),
        spec.SingleBandRasterOutput(
            id="decayed_edt_[STRESSOR]",
            path="intermediate_outputs/decayed_edt_[STRESSOR].tif",
            about=gettext("Distance-weighted influence of the given stressor."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="e_[HABITAT]_[STRESSOR]",
            path="intermediate_outputs/E_[HABITAT]_[STRESSOR].tif",
            about=gettext(
                "Exposure score for a particular habitat/stressor combination."
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="habitat_mask",
            path="intermediate_outputs/habitat_mask.tif",
            about=gettext("Presence of one or more habitats."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="polygonize_mask_[KEY]",
            path="intermediate_outputs/polygonize_mask_[KEY].tif",
            about=gettext("Map of which pixels to polygonize for each habitat or stressor."),
            data_type=int,
            units=None,
            created_if="visualize_outputs"
        ),
        spec.VectorOutput(
            id="polygonized_[KEY]",
            path="intermediate_outputs/polygonized_[KEY].gpkg",
            about=gettext("Polygonized habitat or stressor map"),
            geometry_types={"POLYGON"},
            fields=[],
            created_if="visualize_outputs"
        ),
        spec.SingleBandRasterOutput(
            id="reclass_[HABITAT]_[STRESSOR]",
            path="intermediate_outputs/reclass_[HABITAT]_[STRESSOR].tif",
            about=gettext(
                "The reclassified (high/medium/low) risk of the given stressor to"
                " the given habitat."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="reclass_total_risk_[HABITAT]",
            path="intermediate_outputs/reclass_total_risk_[HABITAT].tif",
            about=gettext("The reclassified (high/medium/low) total risk"),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="recovery_[HABITAT]",
            path="intermediate_outputs/RECOVERY_[HABITAT].tif",
            about=gettext(
                "The resilience or recovery potential for the given habitat"
            ),
            data_type=float,
            units=u.none
        ),
        spec.VectorOutput(
            id="reprojected_[KEY]",
            path="intermediate_outputs/reprojected_[KEY].shp",
            about=gettext(
                "If any habitat, stressor or spatial criteria layers were"
                " provided in a spatial vector format, it will be reprojected to"
                " the AOI projection."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="rewritten_[KEY]",
            path="intermediate_outputs/rewritten_[KEY].tif",
            about=gettext(
                "If any habitat, stressor or spatial criteria layers were"
                " provided in a spatial raster format, it will be reprojected to"
                " the projection of the user’s Area of Interest and written as"
                " GeoTiff at this filepath."
            ),
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="risk_[HABITAT]_[STRESSOR]",
            path="intermediate_outputs/RISK_[HABITAT]_[STRESSOR].tif",
            about=gettext("Risk score for the given habitat-stressor pair."),
            data_type=float,
            units=u.none
        ),
        spec.VectorOutput(
            id="simplified_[KEY]",
            path="intermediate_outputs/simplified_[KEY].gpkg",
            about=gettext(
                "Any habitat, stressor or spatial criteria layers provided are"
                " simplified to 1/2 the user-defined raster resolution in order"
                " to speed up rasterization."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Habitat Risk Assessment.

    Args:
        args['workspace_dir'] (str): a path to the output workspace folder.
            It will overwrite any files that exist if the path already exists.
        args['results_suffix'] (str): a string appended to each output file
            path. (optional)
        args['info_table_path'] (str): a path to the CSV or Excel file that
            contains the name of the habitat (H) or stressor (s) on the
            ``NAME`` column that matches the names in criteria_table_path.
            Each H/S has its corresponding vector or raster path on the
            ``PATH`` column. The ``STRESSOR BUFFER (meters)`` column should
            have a buffer value if the ``TYPE`` column is a stressor.
        args['criteria_table_path'] (str): a path to the CSV or Excel file that
            contains the set of criteria ranking of each stressor on each
            habitat.
        args['resolution'] (int): a number representing the desired pixel
            dimensions of output rasters in meters.
        args['max_rating'] (str, int or float): a number representing the
            highest potential value that should be represented in rating in the
            criteria scores table.
        args['risk_eq'] (str): a string identifying the equation that should be
            used in calculating risk scores for each H-S overlap cell. This
            will be either 'Euclidean' or 'Multiplicative'.
        args['decay_eq'] (str): a string identifying the equation that should
            be used in calculating the decay of stressor buffer influence. This
            can be 'None', 'Linear', or 'Exponential'.
        args['aoi_vector_path'] (str): a path to the shapefile containing one
            or more planning regions used to get the average risk value for
            each habitat-stressor combination over each area. Optionally, if
            each of the shapefile features contain a 'name' field, it will
            be used as a way of identifying each individual shape.
        args['n_overlapping_stressors'] (number): This number will be used
            in risk reclassification instead of the calculated maximum
            number of stressor layers that overlap.
        args['n_workers'] (int): the number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process. (optional)
        args['visualize_outputs'] (bool): if True, create output GeoJSONs and
            save them in a visualization_outputs folder, so users can visualize
            results on the web app. Default to True if not specified.
            (optional)

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, graph = MODEL_SPEC.setup(args)

    target_srs_wkt = pygeoprocessing.get_vector_info(
        args['aoi_vector_path'])['projection_wkt']

    if args['risk_eq'] == 'multiplicative':
        max_pairwise_risk = args['max_rating'] * args['max_rating']
    elif args['risk_eq'] == 'euclidean':
        max_pairwise_risk = math.sqrt(
            ((args['max_rating'] - 1) ** 2) +
            ((args['max_rating'] - 1) ** 2))
    else:
        raise ValueError(
            "args['risk_eq'] must be either 'Multiplicative' or 'Euclidean' "
            f"not {args['risk_eq']}")
    LOGGER.info(
        f"The maximum pairwise risk score for {args['risk_eq']} "
        f"risk is {max_pairwise_risk}")

    # parse the info table and get info dicts for habitats, stressors.
    habitats_info, stressors_info = _parse_info_table(
        args['info_table_path'])

    # parse the criteria table to get the composite table
    criteria_habitats, criteria_stressors = _parse_criteria_table(
        args['criteria_table_path'], file_registry['composite_criteria'])

    # Validate that habitats and stressors match precisely.
    for label, info_table_set, criteria_table_set in [
            ('habitats', set(habitats_info.keys()), criteria_habitats),
            ('stressors', set(stressors_info.keys()), criteria_stressors)]:
        if info_table_set != criteria_table_set:
            missing_from_info_table = ", ".join(
                sorted(criteria_table_set - info_table_set))
            missing_from_criteria_table = ", ".join(
                sorted(info_table_set - criteria_table_set))
            raise ValueError(
                f"The {label} in the info and criteria tables do not match:\n"
                f"  Missing from info table: {missing_from_info_table}\n"
                f"  Missing from criteria table: {missing_from_criteria_table}"
            )

    criteria_df = utils.read_csv_to_dataframe(file_registry['composite_criteria'])
    # Because criteria may be spatial, we need to prepare those spatial inputs
    # as well.
    spatial_criteria_attrs = {}
    for (habitat, stressor, criterion, rating) in criteria_df[
            ['habitat', 'stressor', 'criterion',
             'rating']].itertuples(index=False):
        try:
            float(rating)  # numeric rating
            continue
        except ValueError:
            # If we can't cast it to a float, assume it's a string and
            # therefore spatial.
            pass

        # If the rating is non-numeric, it should be a spatial criterion.
        # this dict matches the structure of the outputs for habitat/stressor
        # dicts, from _parse_info_table
        name = f'{habitat}-{stressor}-{criterion}'
        spatial_criteria_attrs[name] = {
            'name': name,
            'path': rating,  # verified gdal file in _parse_criteria_table
        }

    # Preprocess habitat, stressor spatial criteria datasets.
    # All of these are spatial in nature but might be rasters or vectors.
    user_files_to_aligned_raster_paths = {}
    alignment_source_raster_paths = {}
    alignment_source_vector_paths = {}
    alignment_dependent_tasks = []
    for name, attributes in itertools.chain(habitats_info.items(),
                                            stressors_info.items(),
                                            spatial_criteria_attrs.items()):
        source_filepath = attributes['path']
        gis_type = pygeoprocessing.get_gis_type(source_filepath)
        user_files_to_aligned_raster_paths[
            source_filepath] = file_registry['aligned_[KEY]', name]

        # If the input is already a raster, run it through raster_calculator to
        # ensure we know the nodata value and pixel values.
        if gis_type == pygeoprocessing.RASTER_TYPE:
            alignment_source_raster_paths[
                file_registry['rewritten_[KEY]', name]] = (
                    file_registry['aligned_[KEY]', name])
            # Habitats/stressors must have pixel values of 0 or 1.
            # Criteria may be between [0, max_criteria_score]
            if name in spatial_criteria_attrs:
                # Spatial criteria rasters can represent any positive real
                # values, though they're supposed to be in the range [0, max
                # criteria score], inclusive.
                prep_raster_task = graph.add_task(
                    func=_prep_input_criterion_raster,
                    kwargs={
                        'source_raster_path': source_filepath,
                        'target_filepath': file_registry['rewritten_[KEY]', name],
                    },
                    task_name=(
                        f'Rewrite {name} criteria raster for consistency'),
                    target_path_list=[file_registry['rewritten_[KEY]', name]],
                    dependent_task_list=[]
                )
            else:
                # habitat/stressor rasters represent presence/absence in the
                # form of 1 or 0 pixel values.
                prep_raster_task = graph.add_task(
                    func=_mask_binary_presence_absence_rasters,
                    kwargs={
                        'source_raster_paths': [source_filepath],
                        'target_mask_path': file_registry['rewritten_[KEY]', name],
                    },
                    task_name=(
                        f'Rewrite {name} habitat/stressor raster for '
                        'consistency'),
                    target_path_list=[file_registry['rewritten_[KEY]', name]],
                    dependent_task_list=[]
                )
            alignment_dependent_tasks.append(prep_raster_task)

        # If the input is a vector, reproject to the AOI SRS and simplify.
        # Rasterization happens in the alignment step.
        elif gis_type == pygeoprocessing.VECTOR_TYPE:
            # Using Shapefile here because its driver appears to not raise a
            # warning if a MultiPolygon geometry is inserted into a Polygon
            # layer, which was happening on a real-world sample dataset while
            # in development.
            reprojected_vector_task = graph.add_task(
                pygeoprocessing.reproject_vector,
                kwargs={
                    'base_vector_path': source_filepath,
                    'target_projection_wkt': target_srs_wkt,
                    'target_path': file_registry['reprojected_[KEY]', name],
                },
                task_name=f'Reproject {name} to AOI',
                target_path_list=[file_registry['reprojected_[KEY]', name]],
                dependent_task_list=[]
            )

            # Spatial habitats/stressors are rasterized as presence/absence, so
            # we don't need to preserve any columns.
            fields_to_preserve = None
            if name in spatial_criteria_attrs:
                # In spatial criteria vectors, the 'rating' field contains the
                # numeric rating that needs to be rasterized.
                fields_to_preserve = ['rating']

            alignment_source_vector_paths[
                file_registry['simplified_[KEY]', name]] = (
                    file_registry['aligned_[KEY]', name])
            alignment_dependent_tasks.append(graph.add_task(
                func=_simplify,
                kwargs={
                    'source_vector_path': source_filepath,
                    'tolerance': args['resolution'] / 2,  # by the nyquist theorem.
                    'target_vector_path': file_registry['simplified_[KEY]', name],
                    'preserve_columns': fields_to_preserve,
                },
                task_name=f'Simplify {name}',
                target_path_list=[file_registry['simplified_[KEY]', name]],
                dependent_task_list=[reprojected_vector_task]
            ))

    alignment_task = graph.add_task(
        func=_align,
        kwargs={
            'raster_path_map': alignment_source_raster_paths,
            'vector_path_map': alignment_source_vector_paths,
            'target_pixel_size': (
                args['resolution'], -args['resolution']),
            'target_srs_wkt': target_srs_wkt,
            'all_touched_vectors': [
                file_registry['simplified_[KEY]', key] for key in
                habitats_info | stressors_info],
        },
        task_name='Align raster stack',
        target_path_list=list(itertools.chain(
            alignment_source_raster_paths.values(),
            alignment_source_vector_paths.values())),
        dependent_task_list=alignment_dependent_tasks
    )

    # --> Create a binary mask of habitat pixels.
    all_habitats_mask_task = graph.add_task(
        _mask_binary_presence_absence_rasters,
        kwargs={
            'source_raster_paths':[
                file_registry['aligned_[KEY]', habitat] for habitat in habitats_info],
            'target_mask_path': file_registry['habitat_mask'],
        },
        task_name='Create mask of all habitats',
        target_path_list=[file_registry['habitat_mask']],
        dependent_task_list=[alignment_task]
    )

    # --> for stressor in stressors, do a decayed EDT.
    decayed_edt_tasks = {}  # {stressor name: decayed EDT task}
    for stressor in stressors_info:
        decayed_edt_tasks[stressor] = graph.add_task(
            _calculate_decayed_distance,
            kwargs={
                'stressor_raster_path': file_registry['aligned_[KEY]', stressor],
                'decay_type': args['decay_eq'],
                'buffer_distance': stressors_info[stressor]['buffer'],
                'target_edt_path': file_registry['decayed_edt_[STRESSOR]', stressor],
            },
            task_name=f'Make decayed EDT for {stressor}',
            target_path_list=[file_registry['decayed_edt_[STRESSOR]', stressor]],
            dependent_task_list=[alignment_task]
        )

    # Save this dataframe to make indexing in this loop a little cheaper
    # Resilience/recovery calculations are only done for Consequence criteria.
    reclassed_habitat_risk_tasks = []
    cumulative_risk_to_habitat_paths = []
    cumulative_risk_to_habitat_tasks = []
    habitat_stressor_pairs = []
    all_pairwise_risk_tasks = []
    for habitat in habitats_info:
        pairwise_risk_tasks = []
        pairwise_risk_paths = []
        reclassified_pairwise_risk_paths = []
        reclassified_pairwise_risk_tasks = []

        for stressor in stressors_info:
            criteria_tasks = []
            habitat_stressor_pairs.append((habitat, stressor))

            for criteria_type in ['e', 'c']:
                # This rather complicated filter just grabs the rows matching
                # this habitat, stressor and criteria type.  It's the pandas
                # equivalent of SELECT * FROM criteria_df WHERE the habitat,
                # stressor and criteria type match the current habitat,
                # stressor and criteria type.
                local_criteria_df = criteria_df[
                    (criteria_df['habitat'] == habitat) &
                    (criteria_df['stressor'] == stressor) &
                    (criteria_df['e/c'] == criteria_type.upper())]

                # If we are doing consequence calculations, add in the
                # resilience/recovery parameters for this habitat as additional
                # criteria.
                # Note that if a user provides an E-type RESILIENCE criterion,
                # it will be ignored in all criteria calculations.
                if criteria_type == 'c':
                    local_resilience_df = criteria_df[
                        (criteria_df['habitat'] == habitat) &
                        (criteria_df['stressor'] == _RESILIENCE_STRESSOR) &
                        (criteria_df['e/c'] == 'C')]
                    local_criteria_df = pandas.concat(
                        [local_criteria_df, local_resilience_df])

                # This produces a list of dicts in the form:
                # [{'rating': (score), 'weight': (score), 'dq': (score)}],
                # which is what _calc_criteria() expects.
                attributes_list = []
                for attrs in local_criteria_df[
                        ['rating', 'weight', 'dq']].to_dict(orient='records'):
                    try:
                        float(attrs['rating'])
                    except ValueError:
                        # When attrs['rating'] is not a number, we should
                        # assume it's a spatial file.
                        attrs['rating'] = user_files_to_aligned_raster_paths[
                            attrs['rating']]
                    attributes_list.append(attrs)

                criteria_tasks.append(graph.add_task(
                    _calc_criteria,
                    kwargs={
                        'attributes_list': attributes_list,
                        'habitat_mask_raster_path':
                            file_registry['aligned_[KEY]', habitat],
                        'target_criterion_path':
                            file_registry[f'{criteria_type}_[HABITAT]_[STRESSOR]', habitat, stressor],
                        'decayed_edt_raster_path':
                            file_registry['decayed_edt_[STRESSOR]', stressor],
                    },
                    task_name=(
                        f'Calculate {criteria_type} score for '
                        f'{habitat} / {stressor}'),
                    target_path_list=[file_registry[
                        f'{criteria_type}_[HABITAT]_[STRESSOR]', habitat, stressor]],
                    dependent_task_list=[
                        decayed_edt_tasks[stressor],
                        all_habitats_mask_task
                    ]))

            pairwise_risk_paths.append(file_registry['risk_[HABITAT]_[STRESSOR]', habitat, stressor])

            pairwise_risk_task = graph.add_task(
                _calculate_pairwise_risk,
                kwargs={
                    'habitat_mask_raster_path':
                        file_registry['aligned_[KEY]', habitat],
                    'exposure_raster_path': file_registry['e_[HABITAT]_[STRESSOR]', habitat, stressor],
                    'consequence_raster_path': file_registry['c_[HABITAT]_[STRESSOR]', habitat, stressor],
                    'risk_equation': args['risk_eq'],
                    'target_risk_raster_path': file_registry[
                        'risk_[HABITAT]_[STRESSOR]', habitat, stressor],
                },
                task_name=f'Calculate pairwise risk for {habitat}/{stressor}',
                target_path_list=[file_registry['risk_[HABITAT]_[STRESSOR]', habitat, stressor]],
                dependent_task_list=criteria_tasks
            )
            pairwise_risk_tasks.append(pairwise_risk_task)

            reclassified_pairwise_risk_paths.append(file_registry[
                'reclass_[HABITAT]_[STRESSOR]', habitat, stressor])
            reclassified_pairwise_risk_tasks.append(graph.add_task(
                pygeoprocessing.raster_calculator,
                kwargs={
                    'base_raster_path_band_const_list': [
                        (file_registry['aligned_[KEY]', habitat], 1),
                        (max_pairwise_risk, 'raw'),
                        (file_registry['risk_[HABITAT]_[STRESSOR]', habitat, stressor], 1)],
                    'local_op': _reclassify_score,
                    'target_raster_path': file_registry[
                        'reclass_[HABITAT]_[STRESSOR]', habitat, stressor],
                    'datatype_target': _TARGET_GDAL_TYPE_BYTE,
                    'nodata_target': _TARGET_NODATA_BYTE
                },
                task_name=f'Reclassify risk for {habitat}/{stressor}',
                target_path_list=[file_registry[
                    'reclass_[HABITAT]_[STRESSOR]', habitat, stressor]],
                dependent_task_list=[pairwise_risk_task]
            ))

        # Sum the pairwise risk scores to get cumulative risk to the habitat.
        cumulative_risk_to_habitat_paths.append(file_registry['total_risk_[HABITAT]', habitat])
        cumulative_risk_task = graph.add_task(
            _sum_rasters,
            kwargs={
                'raster_path_list': pairwise_risk_paths,
                'target_nodata': _TARGET_NODATA_FLOAT32,
                'target_datatype': _TARGET_GDAL_TYPE_FLOAT32,
                'target_result_path': file_registry['total_risk_[HABITAT]', habitat],
                'normalize': False,
            },
            task_name=f'Cumulative risk to {habitat}',
            target_path_list=[file_registry['total_risk_[HABITAT]', habitat]],
            dependent_task_list=pairwise_risk_tasks
        )
        cumulative_risk_to_habitat_tasks.append(cumulative_risk_task)

        reclassified_cumulative_risk_task = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (file_registry['aligned_[KEY]', habitat], 1),
                    (max_pairwise_risk * args['n_overlapping_stressors'], 'raw'),
                    (file_registry['total_risk_[HABITAT]', habitat], 1)],
                'local_op': _reclassify_score,
                'target_raster_path': file_registry['reclass_total_risk_[HABITAT]', habitat],
                'datatype_target': _TARGET_GDAL_TYPE_BYTE,
                'nodata_target': _TARGET_NODATA_BYTE,
            },
            task_name=f'Reclassify risk for {habitat}/{stressor}',
            target_path_list=[file_registry['reclass_total_risk_[HABITAT]', habitat]],
            dependent_task_list=[cumulative_risk_task]
        )
        reclassed_habitat_risk_tasks.append(reclassified_cumulative_risk_task)

        max_risk_classification_path = file_registry['reclass_risk_[HABITAT]', habitat]
        _ = graph.add_task(
            pygeoprocessing.raster_calculator,
            kwargs={
                'base_raster_path_band_const_list': [
                    (file_registry['aligned_[KEY]', habitat], 1),
                    (file_registry['reclass_total_risk_[HABITAT]', habitat], 1)
                ] + [(path, 1) for path in reclassified_pairwise_risk_paths],
                'local_op': _maximum_reclassified_score,
                'target_raster_path': max_risk_classification_path,
                'datatype_target': _TARGET_GDAL_TYPE_BYTE,
                'nodata_target': _TARGET_NODATA_BYTE,
            },
            task_name=f'Maximum reclassification for {habitat}',
            target_path_list=[max_risk_classification_path],
            dependent_task_list=[
                reclassified_cumulative_risk_task,
                *reclassified_pairwise_risk_tasks
            ]
        )
        all_pairwise_risk_tasks.extend(pairwise_risk_tasks)
        all_pairwise_risk_tasks.extend(reclassified_pairwise_risk_tasks)

    # total risk to the ecosystem is the sum of all cumulative risk rasters
    # across all habitats.
    # InVEST 3.3.3 has this as the cumulative risk (a straight sum).
    # InVEST 3.10.2 has this as the mean risk per habitat.
    # This is currently implemented as mean risk per habitat.
    ecosystem_risk_path = file_registry['total_risk_ecosystem']
    ecosystem_risk_task = graph.add_task(
        _sum_rasters,
        kwargs={
            'raster_path_list': cumulative_risk_to_habitat_paths,
            'target_nodata': _TARGET_NODATA_FLOAT32,
            'target_datatype': _TARGET_GDAL_TYPE_FLOAT32,
            'target_result_path': ecosystem_risk_path,
            'normalize': True,
        },
        task_name='Cumulative risk to ecosystem.',
        target_path_list=[ecosystem_risk_path],
        dependent_task_list=[
            all_habitats_mask_task,
            *cumulative_risk_to_habitat_tasks
        ]
    )

    # This represents the risk across all stressors.
    # I'm guessing about the risk break to use here, but since the
    # `ecosystem_risk_path` here is the sum across habitats, it makes sense to
    # use max_pairwise_risk * n_habitats.
    _ = graph.add_task(
        pygeoprocessing.raster_calculator,
        kwargs={
            'base_raster_path_band_const_list': [
                (file_registry['habitat_mask'], 1),
                (max_pairwise_risk * len(habitats_info), 'raw'),
                (ecosystem_risk_path, 1)],
            'local_op': _reclassify_score,
            'target_raster_path': file_registry['reclass_risk_ecosystem'],
            'datatype_target': _TARGET_GDAL_TYPE_BYTE,
            'nodata_target': _TARGET_NODATA_BYTE,
        },
        task_name='Reclassify risk to the Ecosystem',
        target_path_list=[file_registry['reclass_total_risk_[HABITAT]', habitat]],
        dependent_task_list=[all_habitats_mask_task, ecosystem_risk_task]
    )

    # Recovery attributes are calculated with the same numerical method as
    # other criteria, but are unweighted by distance to a stressor.
    for habitat in habitats_info:
        resilience_criteria_df = criteria_df[
            (criteria_df['habitat'] == habitat) &
            (criteria_df['stressor'] == _RESILIENCE_STRESSOR)]
        criteria_attributes_list = []
        for attrs in resilience_criteria_df[
                ['rating', 'weight', 'dq']].to_dict(orient='records'):
            try:
                float(attrs['rating'])
            except ValueError:
                # When attrs['rating'] is not a number, we should assume it's a
                # spatial file.
                attrs['rating'] = user_files_to_aligned_raster_paths[
                    attrs['rating']]
            criteria_attributes_list.append(attrs)

        _ = graph.add_task(
            _calc_criteria,
            kwargs={
                'attributes_list': criteria_attributes_list,
                'habitat_mask_raster_path':
                    file_registry['aligned_[KEY]', habitat],
                'target_criterion_path': file_registry['recovery_[HABITAT]', habitat],
                'decayed_edt_raster_path': None,  # not a stressor so no EDT
            },
            task_name=f'Calculate recovery score for {habitat}',
            target_path_list=[file_registry['recovery_[HABITAT]', habitat]],
            dependent_task_list=[all_habitats_mask_task]
        )

    aoi_simplify_task = graph.add_task(
        func=_simplify,
        kwargs={
            'source_vector_path': args['aoi_vector_path'],
            'tolerance': args['resolution'] / 2,  # by the nyquist theorem
            'target_vector_path': file_registry['simplified_aoi'],
            'preserve_columns': ['name'],
        },
        task_name='Simplify AOI',
        target_path_list=[file_registry['simplified_aoi']],
        dependent_task_list=[]
    )

    summary_csv_path = file_registry['summary_statistics']
    _ = graph.add_task(
        func=_create_summary_statistics_file,
        kwargs={
            'subregions_vector_path': file_registry['simplified_aoi'],
            'habitats': list(habitats_info.keys()),
            'stressors': list(stressors_info.keys()),
            'file_registry': file_registry,
            'target_summary_csv_path': summary_csv_path,
        },
        task_name='Create summary statistics table',
        target_path_list=[summary_csv_path],
        dependent_task_list=[
            aoi_simplify_task,
            *all_pairwise_risk_tasks,
            *reclassed_habitat_risk_tasks
        ]
    )

    graph.join()
    if not args['visualize_outputs']:
        LOGGER.info('HRA complete!')
        graph.close()
        return file_registry.registry

    # Although the generation of visualization outputs could have more precise
    # task dependencies, the effort involved in tracking them precisely doesn't
    # feel worth it when the visualization steps are such a standalone
    # component and it isn't clear if we'll end up keeping this in HRA or
    # refactoring it out (should we end up doing more such visualizations).
    LOGGER.info('Generating visualization outputs')
    shutil.copy(  # copy in the summary table.
        summary_csv_path,
        file_registry['summary_statistics_viz'])

    for key, type in ([(habitat, 'habitat') for habitat in habitats_info] +
                      [('ecosystem', 'habitat')] +
                      [(stressor, 'stressor') for stressor in stressors_info]):
        if type == 'habitat':
            if key == 'ecosystem':
                source_raster_path = file_registry['reclass_risk_ecosystem']
                target_geojson_path = file_registry[f'reclass_risk_ecosystem_viz']
            else:
                source_raster_path = file_registry['reclass_risk_[HABITAT]', key]
                target_geojson_path = file_registry[f'reclass_risk_[HABITAT]_viz', key]
            field_name = 'Risk Score'
            layer_name = f'RECLASS_RISK_{key}'
        else:  # type == 'stressor'
            source_raster_path = file_registry['aligned_[KEY]', key]
            target_geojson_path = file_registry[f'stressor_[STRESSOR]_viz', key]
            field_name = 'Stressor'
            layer_name = f'STRESSOR_{key}'

        rewrite_for_polygonize_task = graph.add_task(
            func=_create_mask_for_polygonization,
            kwargs={
                'source_raster_path': source_raster_path,
                'target_raster_path': file_registry['polygonize_mask_[KEY]', key],
            },
            task_name=f'Rewrite {key} for polygonization',
            target_path_list=[file_registry['polygonize_mask_[KEY]', key]],
            dependent_task_list=[]
        )

        polygonize_task = graph.add_task(
            func=_polygonize,
            kwargs={
                'source_raster_path': source_raster_path,
                'mask_raster_path': file_registry['polygonize_mask_[KEY]', key],
                'target_polygonized_vector': file_registry['polygonized_[KEY]', key],
                'field_name': field_name,
                'layer_name': layer_name
            },
            task_name=f'Polygonizing {key}',
            target_path_list=[file_registry['polygonized_[KEY]', key]],
            dependent_task_list=[rewrite_for_polygonize_task]
        )

        _ = graph.add_task(
            pygeoprocessing.reproject_vector,
            kwargs={
                'base_vector_path': file_registry['polygonized_[KEY]', key],
                'target_projection_wkt': _WGS84_WKT,
                'target_path': target_geojson_path,
                'driver_name': 'GeoJSON',
            },
            task_name=f'Reproject {key} to AOI',
            target_path_list=[target_geojson_path],
            dependent_task_list=[polygonize_task]
        )

    LOGGER.info(
        'HRA model completed. Please visit http://marineapps.'
        'naturalcapitalproject.org/ to visualize your outputs.')

    graph.close()
    graph.join()
    LOGGER.info('HRA complete!')
    return file_registry.registry


def _create_mask_for_polygonization(source_raster_path, target_raster_path):
    """Create a mask of non-nodata pixels.

    This mask raster is intended to be used as a mask input for GDAL's
    polygonization function.

    Args:
        source_raster_path (string): The source raster from which the mask
            raster will be created. This raster is assumed to be an integer
            raster.
        target_raster_path (string): The path to where the target raster should
            be written.

    Returns:
        ``None``
    """
    nodata = pygeoprocessing.get_raster_info(source_raster_path)['nodata'][0]

    def _rewrite(raster_values):
        """Convert any non-nodata values to 1, all other values to 0.

        Args:
            raster_values (numpy.array): Integer pixel values from the source
                raster.

        Returns:
            out_array (numpy.array): An unsigned byte mask with pixel values of
            0 (on nodata pixels) or 1 (on non-nodata pixels).
        """
        return (raster_values != nodata).astype(numpy.uint8)

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _rewrite, target_raster_path,
        gdal.GDT_Byte, 0)


def _polygonize(source_raster_path, mask_raster_path,
                target_polygonized_vector, field_name, layer_name):
    """Polygonize a raster.

    Args:
        source_raster_path (string): The source raster to polygonize.  This
            raster must be an integer raster.
        mask_raster_path (string): The path to a mask raster, where pixel
            values are 1 where polygons should be created, 0 otherwise.
        target_polygonized_vector (string): The path to a vector into which the
            new polygons will be inserted.  A new GeoPackage will be created at
            this location.
        field_name (string): The name of a field into which the polygonized
            region's numerical value should be recorded.  A new field with this
            name will be created in ``target_polygonized_vector``, and with an
            Integer field type.
        layer_name (string): The layer name to use in the target vector.

    Returns:
        ``None``
    """
    LOGGER.info(f'Polygonizing {source_raster_path} --> '
                f'{target_polygonized_vector} using mask {mask_raster_path}')
    raster = gdal.OpenEx(source_raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)

    mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())

    driver = gdal.GetDriverByName('GPKG')
    vector = driver.Create(
        target_polygonized_vector, 0, 0, 0, gdal.GDT_Unknown)
    layer = vector.CreateLayer(layer_name, raster_srs, ogr.wkbPolygon)

    # Create an integer field that contains values from the raster
    field_defn = ogr.FieldDefn(str(field_name), ogr.OFTInteger)
    field_defn.SetWidth(3)
    field_defn.SetPrecision(0)
    layer.CreateField(field_defn)

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())

    layer.StartTransaction()
    # 0 represents field index 0, into which pixel values will be written.
    gdal.Polygonize(band, mask_band, layer, 0)
    layer.CommitTransaction()

    layer = None
    vector = None
    driver = None
    band = None
    raster = None
    mask_band = None
    mask_raster = None


def _create_summary_statistics_file(
        subregions_vector_path,
        habitats,
        stressors,
        file_registry,
        target_summary_csv_path):
    """Create summary statistics table.

    This table tracks for each habitat, stressor and subregion the min, max and
    mean Exposure, Criteria and Risk score.  Additionally, the percentage of
    risk scores in each classification category (HIGH, MEDIUM, LOW, NONE) are
    also recorded.  If subregion names are not provided by user input, a single
    subregion is assumed, called "Total Region".

    If multiple subregions have the same name, they are treated as a single
    subregion.

    Args:
        subregions_vector_path (string): The path to a vector of subregions.
            If this vector has a ``"NAME"`` column (case-insensitive), it will
            be used to uniquely identify the subregion.
        habitats (list): list of habitats
        stressors (list): list of stressors
        file_registry (FileRegistry): used to look up raster paths to summarize
        target_summary_csv_path (string): The path to where the target CSV
            should be written on disk.

    Returns:
        ``None``
    """
    subregions_vector = gdal.OpenEx(subregions_vector_path)
    subregions_layer = subregions_vector.GetLayer()
    name_field = None
    for field_defn in subregions_layer.schema:
        source_fieldname = field_defn.GetName()
        if source_fieldname.lower() == 'name':
            name_field = source_fieldname
            break
    subregion_fid_to_name = {}
    for feature in subregions_layer:
        if name_field is None:
            subregion_name = 'Total Region'
        else:
            subregion_name = feature.GetField(name_field)
        subregion_fid_to_name[feature.GetFID()] = subregion_name
    subregions_layer = None
    subregions_vector = None

    records = []
    for habitat, stressor in itertools.product(sorted(habitats),
                                               sorted(stressors)):
        e_raster = file_registry[f'e_[HABITAT]_[STRESSOR]', habitat, stressor]
        c_raster = file_registry[f'c_[HABITAT]_[STRESSOR]', habitat, stressor]
        r_raster = file_registry[f'risk_[HABITAT]_[STRESSOR]', habitat, stressor]
        classes_raster = file_registry[f'reclass_[HABITAT]_[STRESSOR]', habitat, stressor]

        subregion_stats_by_name = collections.defaultdict(
            lambda: {
                **{f"{prefix}_MIN": float('inf') for prefix in "ECR"},
                **{f"{prefix}_MAX": 0 for prefix in "ECR"},
                **{f"{prefix}_SUM": 0 for prefix in "ECR"},
                **{f"{prefix}_COUNT": 0 for prefix in "ECR"},
                **{f"R_N_{tally}": 0 for tally in (
                    "PIXELS", "NONE", "LOW", "MEDIUM", "HIGH")}
            })

        for prefix, raster_path in (
                ('E', e_raster),
                ('C', c_raster),
                ('R', r_raster)):
            raster_stats = pygeoprocessing.zonal_statistics(
                (raster_path, 1), subregions_vector_path)

            for feature_id, stats_under_feature in raster_stats.items():
                feature_name = subregion_fid_to_name[feature_id]
                subregion_stats = subregion_stats_by_name[feature_name]

                for opname, reduce_func in (
                        ('MIN', min), ('MAX', max), ('SUM', sum),
                        ('COUNT', sum)):
                    fieldname = f'{prefix}_{opname}'
                    try:
                        subregion_stats[fieldname] = reduce_func([
                            subregion_stats[fieldname],
                            stats_under_feature[opname.lower()]])
                    except TypeError:
                        # TypeError when stats_under_feature[op] is None,
                        # which happens when the polygon is entirely over
                        # nodata
                        pass
                subregion_stats_by_name[feature_name] = subregion_stats

        raster_stats = pygeoprocessing.zonal_statistics(
            (classes_raster, 1), subregions_vector_path,
            include_value_counts=True)
        for feature_id, stats_under_feature in raster_stats.items():
            counts = collections.defaultdict(int)
            counts.update(stats_under_feature['value_counts'])
            feature_name = subregion_fid_to_name[feature_id]
            subregion_stats = subregion_stats_by_name[feature_name]
            for classified_value, field in (
                    (0, 'NONE'), (1, 'LOW'), (2, 'MEDIUM'), (3, 'HIGH')):
                subregion_stats[f'R_N_{field}'] += counts[
                    classified_value]
                subregion_stats['R_N_PIXELS'] += counts[classified_value]
            subregion_stats_by_name[feature_name] = subregion_stats

        for subregion_name, subregion_stats in (
                subregion_stats_by_name.items()):
            record = {
                'HABITAT': habitat,
                'STRESSOR': stressor,
                'SUBREGION': subregion_name,
            }
            for prefix in ('E', 'C', 'R'):
                # Copying over SUM and COUNT for per-habitat summary
                # statistics later.
                for op in ('MIN', 'MAX', 'SUM', 'COUNT'):
                    key = f'{prefix}_{op}'
                    record[key] = subregion_stats[key]
                try:
                    record[f'{prefix}_MEAN'] = (
                        subregion_stats[f'{prefix}_SUM'] /
                        subregion_stats[f'{prefix}_COUNT'])
                except ZeroDivisionError:
                    record[f'{prefix}_MEAN'] = 0

            n_pixels = subregion_stats['R_N_PIXELS']
            for classification in ('NONE', 'LOW', 'MEDIUM', 'HIGH'):
                percent_classified = 0
                if n_pixels > 0:  # avoid a division by 0
                    percent_classified = (
                        subregion_stats[f'R_N_{classification}'] /
                        n_pixels) * 100
                record[f'R_%{classification}'] = percent_classified
            records.append(record)

    pairwise_df = pandas.DataFrame.from_records(records)

    all_stressors_id = '(FROM ALL STRESSORS)'
    for habitat in habitats:
        stats = pygeoprocessing.zonal_statistics(
            (file_registry['reclass_total_risk_[HABITAT]', habitat], 1),
            subregions_vector_path, include_value_counts=True)
        subregion_stats_by_name = collections.defaultdict(
            lambda: {0: 0, 1: 0, 2: 0, 3: 0})
        for feature_id, stats_under_feature in stats.items():
            feature_name = subregion_fid_to_name[feature_id]
            subregion_stats = subregion_stats_by_name[feature_name]
            for key, count in stats_under_feature['value_counts'].items():
                subregion_stats_by_name[feature_name][key] += count

        for subregion_name, class_counts in subregion_stats_by_name.items():
            n_pixels = max(sum(class_counts.values()), 1)  # avoid div-by-0
            record = {
                'HABITAT': habitat,
                'STRESSOR': all_stressors_id,
                'SUBREGION': subregion_name,
                'R_%NONE': (class_counts[0] / n_pixels) * 100,
                'R_%LOW': (class_counts[1] / n_pixels) * 100,
                'R_%MEDIUM': (class_counts[2] / n_pixels) * 100,
                'R_%HIGH': (class_counts[3] / n_pixels) * 100,
            }
            matching_df = pairwise_df[
                (pairwise_df['HABITAT'] == habitat) &
                (pairwise_df['SUBREGION'] == subregion_name)]
            for prefix in ('E', 'C', 'R'):
                record[f'{prefix}_MIN'] = matching_df[
                    f'{prefix}_MIN'].min()
                record[f'{prefix}_MAX'] = matching_df[
                    f'{prefix}_MAX'].max()

                # Handle division-by-zero in mean calculation
                count_sum = matching_df[f'{prefix}_COUNT'].sum()
                mean = 0
                if count_sum > 0:
                    mean = matching_df[f'{prefix}_SUM'].sum() / count_sum
                record[f'{prefix}_MEAN'] = mean

            records.append(record)

    out_dataframe = pandas.DataFrame.from_records(
        records, columns=[
            'HABITAT', 'STRESSOR', 'SUBREGION',
            'E_MIN', 'E_MAX', 'E_MEAN',
            'C_MIN', 'C_MAX', 'C_MEAN',
            'R_MIN', 'R_MAX', 'R_MEAN',
            'R_%HIGH', 'R_%MEDIUM', 'R_%LOW', 'R_%NONE'
        ])
    out_dataframe.sort_values(['HABITAT', 'STRESSOR', 'SUBREGION'],
                              inplace=True)
    out_dataframe.to_csv(target_summary_csv_path, index=False)


def _align(raster_path_map, vector_path_map, target_pixel_size,
           target_srs_wkt, all_touched_vectors=None):
    """Align a stack of rasters and/or vectors.

    In HRA, habitats and stressors (and optionally criteria) may be defined as
    a stack of rasters and/or vectors.  This function enables this stack to be
    precisely aligned and rasterized, while minimizing sampling error in
    rasterization.

    Rasters passed in to this function will be aligned and resampled using a
    datatype-appropriate interpolation function: nearest-neighbor for integer
    rasters, bilinear for floating-point.

    Vectors passed in to this function will be rasterized onto new rasters that
    align with the rest of the stack.

    All aligned rasters and rasterized vectors will have a bounding box that
    matches the union of the bounding boxes of all spatial inputs to this
    function, and with the target pixel size and SRS.

    To be safe, this function assumes that the source SRS of any dataset might
    be different from the target SRS, and so all source bounding boxes are
    transformed into the target SRS before the target bounding box is computed.

    Args:
        raster_path_map (dict): A dict mapping source raster paths to aligned
            raster paths.  This dict may be empty.
        vector_path_map (dict): A dict mapping source vector paths to aligned
            raster paths.  This dict may be empty.  These source vectors must
            already be in the target projection's SRS.  If a 'rating'
            (case-insensitive) column is present, then those values will be
            rasterized.  Otherwise, 1 will be rasterized.
        target_pixel_size (tuple): The pixel size of the target rasters, in
            the form (x, y), expressed in meters.
        target_srs_wkt (string): The target SRS of the aligned rasters.
        all_touched_vectors=None (set): A set of vector paths found in
            ``vector_path_map`` that should be rasterized with
            ``ALL_TOUCHED=TRUE``.

    Returns:
        ``None``
    """
    # Step 1: Create a bounding box of the union of all input spatial rasters
    # and vectors.
    bounding_box_list = []
    source_raster_paths = []
    aligned_raster_paths = []
    resample_method_list = []
    for source_raster_path, aligned_raster_path in raster_path_map.items():
        source_raster_paths.append(source_raster_path)
        aligned_raster_paths.append(aligned_raster_path)
        raster_info = pygeoprocessing.get_raster_info(source_raster_path)

        # Integer (discrete) rasters should be nearest-neighbor, continuous
        # rasters should be interpolated with bilinear.
        if numpy.issubdtype(raster_info['numpy_type'], numpy.integer):
            resample_method_list.append('near')
        else:
            resample_method_list.append('bilinear')

        bounding_box_list.append(pygeoprocessing.transform_bounding_box(
            raster_info['bounding_box'], raster_info['projection_wkt'],
            target_srs_wkt))

    for source_vector_path in vector_path_map.keys():
        vector_info = pygeoprocessing.get_vector_info(source_vector_path)
        bounding_box_list.append(pygeoprocessing.transform_bounding_box(
            vector_info['bounding_box'], vector_info['projection_wkt'],
            target_srs_wkt))

    # Bounding box is in the order [minx, miny, maxx, maxy]
    target_bounding_box = pygeoprocessing.merge_bounding_box_list(
        bounding_box_list, 'union')

    # Step 2: Align raster inputs.
    # If any rasters were provided, they will be aligned to the bounding box
    # we determined, interpolating and warping as needed.
    if raster_path_map:
        LOGGER.info(f'Aligning {len(raster_path_map)} rasters')
        pygeoprocessing.align_and_resize_raster_stack(
            base_raster_path_list=source_raster_paths,
            target_raster_path_list=aligned_raster_paths,
            resample_method_list=resample_method_list,
            target_pixel_size=target_pixel_size,
            bounding_box_mode=target_bounding_box,
            target_projection_wkt=target_srs_wkt,
            raster_align_index=None,  # Calculate precise intersection.
        )

    # Step 3: Rasterize vectors onto aligned rasters.
    # If any vectors were provided, they will be rasterized onto new rasters
    # that align with the bounding box we determined earlier.
    # This approach yields more precise rasters than resampling an
    # already-rasterized vector through align_and_resize_raster_stack.
    if all_touched_vectors is None:
        all_touched_vectors = set([])
    if vector_path_map:
        LOGGER.info(f'Aligning {len(vector_path_map)} vectors')
        for source_vector_path, target_raster_path in vector_path_map.items():
            # if there's a 'rating' column, then we're rasterizing a vector
            # attribute that represents a positive floating-point value.
            vector = gdal.OpenEx(source_vector_path, gdal.OF_VECTOR)
            layer = vector.GetLayer()
            raster_type = _TARGET_GDAL_TYPE_BYTE
            nodata_value = _TARGET_NODATA_BYTE
            burn_values = [1]
            rasterize_option_list = []

            # Only rasterize with ALL_TOUCHED=TRUE if we know it should be
            # rasterized as such (habitats/stressors)
            if source_vector_path in all_touched_vectors:
                rasterize_option_list.append('ALL_TOUCHED=TRUE')

            for field in layer.schema:
                fieldname = field.GetName()
                if fieldname.lower() == 'rating':
                    rasterize_option_list.append(
                        f'ATTRIBUTE={fieldname}')
                    raster_type = _TARGET_GDAL_TYPE_FLOAT32
                    nodata_value = _TARGET_NODATA_FLOAT32
                    burn_values = None
                    break

            layer = None
            vector = None

            pygeoprocessing.create_raster_from_bounding_box(
                target_raster_path=target_raster_path,
                target_bounding_box=target_bounding_box,
                target_pixel_size=target_pixel_size,
                target_pixel_type=raster_type,
                target_srs_wkt=target_srs_wkt,
                target_nodata=nodata_value,
                fill_value=nodata_value
            )

            pygeoprocessing.rasterize(
                source_vector_path, target_raster_path,
                burn_values=burn_values, option_list=rasterize_option_list)


def _simplify(source_vector_path, tolerance, target_vector_path,
              preserve_columns=None):
    """Simplify a geometry to a given tolerance.

    This function uses the GEOS SimplifyPreserveTopology function under the
    hood.  For docs, see GEOSTopologyPreserveSimplify() at
    https://libgeos.org/doxygen/geos__c_8h.html

    Args:
        source_vector_path (string): The path to a source vector to simplify.
        tolerance (number): The numerical tolerance to simplify by.
        target_vector_path (string): Where the simplified geometry should be
            stored on disk.
        preserve_columns=None (iterable or None): If provided, this is an
            iterable of string column names (case-insensitive) that should be
            carried over from the source vector to the target vector.  If
            ``None``, no columns will be carried over.

    Returns:
        ``None``.
    """
    LOGGER.info(
        f'Simplifying vector with tolerance {tolerance}: {source_vector_path}')
    if preserve_columns is None:
        preserve_columns = []
    preserve_columns = set(name.lower() for name in preserve_columns)

    source_vector = gdal.OpenEx(source_vector_path)
    source_layer = source_vector.GetLayer()

    target_driver = gdal.GetDriverByName('GPKG')
    target_vector = target_driver.Create(
        target_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    target_layer_name = os.path.splitext(
        os.path.basename(target_vector_path))[0]

    # Use the same geometry type from the source layer. This may be wkbUnknown
    # if the layer contains multiple geometry types.
    target_layer = target_vector.CreateLayer(
        target_layer_name,
        srs=source_layer.GetSpatialRef(),
        geom_type=source_layer.GetGeomType())

    for field in source_layer.schema:
        if field.GetName().lower() in preserve_columns:
            new_definition = ogr.FieldDefn(field.GetName(), field.GetType())
            target_layer.CreateField(new_definition)

    target_layer_defn = target_layer.GetLayerDefn()
    target_layer.StartTransaction()

    for source_feature in source_layer:
        target_feature = ogr.Feature(target_layer_defn)
        source_geom = source_feature.GetGeometryRef()
        simplified_geom = source_geom.SimplifyPreserveTopology(tolerance)
        if simplified_geom is not None:
            target_geom = simplified_geom
        else:
            # If the simplification didn't work for whatever reason, fall back
            # to the original geometry.
            LOGGER.debug(
                f"Simplification of {os.path.basename(source_vector_path)} "
                f"feature FID:{source_feature.GetFID()} failed; falling back "
                "to original geometry")
            target_geom = source_geom

        for fieldname in [field.GetName() for field in target_layer.schema]:
            target_feature.SetField(
                fieldname, source_feature.GetField(fieldname))

        target_feature.SetGeometry(target_geom)
        target_layer.CreateFeature(target_feature)

    target_layer.CommitTransaction()
    target_layer = None
    target_vector = None


def _prep_input_criterion_raster(
        source_raster_path, target_filepath):
    """Prepare an input criterion raster for internal use.

    In order to make raster calculations more consistent within HRA, it's
    helpful to preprocess spatially-explicity criteria rasters for consistency.
    Rasters produced by this function will have:

        * A nodata value matching ``_TARGET_NODATA_FLOAT32``
        * Source values < 0 are converted to ``_TARGET_NODATA_FLOAT32``

    Args:
        source_raster_path (string): The path to a user-provided criterion
            raster.
        target_filepath (string): The path to where the translated raster
            should be written on disk.

    Returns:
        ``None``.
    """
    source_nodata = pygeoprocessing.get_raster_info(
        source_raster_path)['nodata'][0]

    def _translate_op(source_rating_array):
        """Translate rating pixel values.

            * Nodata values should have the new (internal) nodata.
            * Anything < 0 should become nodata.
            * Anything >= 0 is assumed to be valid and left as-is.

        Args:
            source_rating_array (numpy.array): An array of rating values.

        Returns:
            ``target_rating_array`` (numpy.array): An array with
                potentially-translated values.
        """
        target_rating_array = numpy.full(
            source_rating_array.shape, _TARGET_NODATA_FLOAT32,
            dtype=numpy.float32)
        valid_mask = (
             (~pygeoprocessing.array_equals_nodata(
                 source_rating_array, source_nodata)) &
             (source_rating_array >= 0.0))
        target_rating_array[valid_mask] = source_rating_array[valid_mask]
        return target_rating_array

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1)], _translate_op, target_filepath,
        _TARGET_GDAL_TYPE_FLOAT32, _TARGET_NODATA_FLOAT32)


def _mask_binary_presence_absence_rasters(
        source_raster_paths, target_mask_path):
    """Create a mask where any values in a raster stack are 1.

    Given a stack of aligned source rasters, if any pixel values in a pixel
    stack are exactly equal to 1, the output mask at that pixel will be 1.

    This can be applied to user-defined rasters or in creating a mask across a
    stack of presence/absence rasters.

    Args:
        source_raster_paths (list): A list of string paths to rasters.
        target_mask_path (string): The path to there the mask raster
            will be written.

    Returns:
        ``None``
    """
    def _translate_op(*input_arrays):
        """Translate the input array to nodata, except where values are 1."""
        presence = numpy.full(input_arrays[0].shape, _TARGET_NODATA_BYTE,
                              dtype=numpy.uint8)
        for input_array in input_arrays:
            presence[input_array == 1] = 1
        return presence

    pygeoprocessing.raster_calculator(
        [(source_path, 1) for source_path in source_raster_paths],
        _translate_op, target_mask_path,
        _TARGET_GDAL_TYPE_BYTE, _TARGET_NODATA_BYTE)


def _parse_info_table(info_table_path):
    """Parse the HRA habitat/stressor info table.

    Args:
        info_table_path (string): The path to the info table.  May be either
            CSV or Excel format.  The columns 'name', 'path', 'type' and
            'stressor buffer (meters)' are all required.  The stressor buffer
            only needs values for those rows representing stressors.

    Returns:
        (habitats, stressors) (tuple): a 2-tuple of dicts where the habitat or
            stressor name (respectively) maps to attributes about that habitat
            or stressor:

                * Habitats have a structure of
                  ``{habitat_name: {'path': path to spatial layer}}``
                * Stressors have a structure of
                  ``{stressor_name: {'path': path to spatial layer, 'buffer':
                      buffer distance}}``
    """
    try:
        table = MODEL_SPEC.get_input(
            'info_table_path').get_validated_dataframe(info_table_path)
    except ValueError as err:
        if 'Index has duplicate keys' in str(err):
            raise ValueError("Habitat and stressor names may not overlap.")
        else:
            raise err

    table = table.rename(columns={'stressor buffer (meters)': 'buffer'})

    # Drop the buffer column from the habitats list; we don't need it.
    habitats = table.loc[table['type'] == 'habitat'].drop(
        columns=['type', 'buffer']).to_dict(orient='index')

    # Keep the buffer column in the stressors dataframe.
    stressors = table.loc[table['type'] == 'stressor'].drop(
        columns=['type']).to_dict(orient='index')

    return (habitats, stressors)


def _parse_criteria_table(criteria_table_path, target_composite_csv_path):
    """Parse the criteria table.

    The criteria table is a single table that's really representing a
    multidimensional dataset, where each combination of habitat, stressor and
    criterion has a RATING, DQ (data quality) and WEIGHT.  Criteria are further
    divided into Exposure and Consequence categories.  Habitat Resilience
    consequence criteria are also defined in a similar structure and are also
    included in this table.

    Args:
        criteria_table_path (string): The path to a CSV file on disk.
        target_composite_csv_path (string): The path to where a new CSV should
            be written containing similar information but in a more easily
            parseable (for a program) format.

    Returns:
        criteria_habitats (set): A set of string names of habitats found in the
            criteria table, lowercased.
        criteria_stressors (set): A set of string names of stressors found in
            the criteria table, lowercased.
    """
    # This function requires that the table is read as a numpy array, so it's
    # easiest to read the table directly.
    table = utils.read_csv_to_dataframe(
        criteria_table_path, header=None).to_numpy()

    # clean up any leading or trailing whitespace.
    for row_num in range(table.shape[0]):
        for col_num in range(table.shape[1]):
            value = table[row_num][col_num]
            if isinstance(value, str):
                table[row_num][col_num] = value.strip()

    # Some fields are required and will lead to cryptic errors if not found or
    # slightly misspelled.
    habitat_name_header = 'HABITAT NAME'
    overlap_section_header = 'HABITAT STRESSOR OVERLAP PROPERTIES'
    habitat_resilience_header = 'HABITAT RESILIENCE ATTRIBUTES'
    required_section_headers = {
        habitat_name_header, overlap_section_header,
        habitat_resilience_header}
    missing_sections = required_section_headers - set(table[:, 0])
    if missing_sections:
        raise AssertionError(
            "The criteria table is missing these section headers: "
            f"{', '.join(missing_sections)}")

    # Habitats are loaded from the top row (table[0])
    known_habitats = set(table[0]).difference(
        {habitat_name_header, numpy.nan, 'CRITERIA TYPE'})

    # Stressors are loaded from the first column (table[:, 0])
    known_stressors = set()
    for row_index, value in enumerate(table[:, 0]):
        try:
            if value == overlap_section_header or numpy.isnan(value):
                known_stressors.add(table[row_index + 1, 0])
        except TypeError:
            # calling numpy.isnan on a string raises a TypeError.
            pass
    # The overlap section header is presented after an empty line, so it'll
    # normally show up in this set.  Remove it.
    # It's also all too easy to end up with multiple nan rows between sections,
    # so if a numpy.nan ends up in the set, remove it.
    for value in (overlap_section_header, numpy.nan):
        try:
            known_stressors.remove(value)
        except KeyError:
            pass

    # Fill in habitat names in the table's top row for easier reference.
    criteria_col = None
    habitats = set()
    for col_index, value in enumerate(table[0]):
        if value == habitat_name_header:
            continue
        if value == 'CRITERIA TYPE':
            criteria_col = col_index
            break  # We're done with habitats
        if not isinstance(value, str):
            value = table[0][col_index-1]  # Fill in from column to the left
            habitats.add(value)
            table[0][col_index] = value

    habitat_columns = collections.defaultdict(list)
    for col_index, col_value in enumerate(table[0]):
        if col_value in habitats:
            habitat_columns[col_value].append(col_index)

    records = []
    current_stressor = None
    for row_index, row in enumerate(table[1:], start=1):
        if row[0] == overlap_section_header:
            continue

        if (row[0] in known_stressors or
                row[0] == habitat_resilience_header):
            if row[0] == habitat_resilience_header:
                row[0] = _RESILIENCE_STRESSOR  # Shorten for convenience

            # Lowercase stressors to match expectations for how the info table
            # is being parsed.
            current_stressor = row[0].lower()
            current_stressor_header_row = row_index
            continue  # can skip this row

        try:
            if numpy.all(numpy.isnan(row.astype(numpy.float32))):
                continue
        except (TypeError, ValueError):
            # Either of these exceptions are thrown when there are string types
            # in the row
            pass

        # {habitat: {rating/dq/weight/ec dict}
        stressor_habitat_data = {
            'stressor': current_stressor,
            'criterion': row[0],
            'e/c': row[criteria_col],
        }
        for habitat, habitat_col_indices in habitat_columns.items():
            # Lowercase habitats to match expectations for how the info table
            # is being parsed.
            stressor_habitat_data['habitat'] = habitat.lower()
            for col_index in habitat_col_indices:
                # attribute is rating, dq or weight
                attribute_name = table[current_stressor_header_row][col_index]
                attribute_value = row[col_index]
                try:
                    # Just a test to see if this value is a number.
                    # It's OK if it stays an int.
                    float(attribute_value)
                except ValueError:
                    gdal_path = utils._GDALPath.from_uri(attribute_value)
                    # If we can't cast it to a float, assume it's a string path
                    # to a raster or vector. Don't expand remote paths.
                    if gdal_path.is_local:
                        attribute_value = utils.expand_path(
                            attribute_value, criteria_table_path)
                    else:
                        attribute_value = gdal_path.to_normalized_path()
                    try:
                        _ = pygeoprocessing.get_gis_type(attribute_value)
                    except ValueError:
                        # File is not a spatial file or file not found
                        raise ValueError(
                            "Criterion could not be opened as a spatial "
                            f"file {attribute_value}")

                stressor_habitat_data[
                    attribute_name.lower()] = attribute_value
            # Keep the copy() unless you want to all of the 'records' dicts to
            # have the same contents!  This is normal behavior given python's
            # memory model.
            records.append(stressor_habitat_data.copy())

    # the primary key of this table is (habitat, stressor, criterion)
    overlap_df = pandas.DataFrame.from_records(
        records, columns=['habitat', 'stressor', 'criterion', 'rating', 'dq',
                          'weight', 'e/c'])
    overlap_df.to_csv(target_composite_csv_path, index=False)

    # Lowercase habitats and stressors to match expectations for how the
    # info table is being parsed.
    known_habitats = {habitat.lower() for habitat in known_habitats}
    known_stressors = {stressor.lower() for stressor in known_stressors}

    return (known_habitats, known_stressors)


def _calculate_decayed_distance(stressor_raster_path, decay_type,
                                buffer_distance, target_edt_path):
    """Decay the influence of a stressor given decay type and buffer distance.

    Args:
        stressor_raster_path (string): The path to a stressor raster, where
            pixel values of 1 indicate presence of the stressor and 0 or nodata
            indicate absence of the stressor.
        decay_type (string): The type of decay.  Valid values are:

            * ``linear``: Pixel values decay linearly from the stressor pixels
              out to the buffer distance.
            * ``exponential``: Pixel values decay exponentially out to the
              buffer distance.
            * ``none``: Pixel values do not decay out to the buffer distance.
              Instead, all pixels within the buffer distance have the same
              influence as though it were overlapping the stressor itself.
        buffer_distance (number): The distance out to which the stressor has an
            influence.  This is in linearly projected units and should be in
            meters.
        target_edt_path (string): The path where the target EDT raster should
            be written.

    Returns:
        ``None``

    Raises:
        ``AssertionError``: When an invalid ``decay_type`` is provided.
    """
    decay_type = decay_type.lower()
    if decay_type not in MODEL_SPEC.get_input('decay_eq').list_options():
        raise AssertionError(f'Invalid decay type {decay_type} provided.')

    if buffer_distance == 0:
        LOGGER.info(
            f'Buffer distance for {target_edt_path} is 0, skipping distance '
            'transform')
        # The UG states that if the buffer distance is 0, then we don't do a
        # buffer or decay at all.  A raster_calculator call will be cheaper and
        # easier to compute.

        def _no_buffer(stressor_presence_array):
            """Translate a stressor presence array to match an EDT.

            Args:
                stressor_presence_array (numpy.array): A numpy byte array with
                    values of 1 indicating presence.  Absence is indicated by 0
                    or nodata (or any value other than 1).

            Returns:
                A numpy array with values of 1 (stressor presence) or 0
                (absence).  The array's type is float32.
            """
            result = numpy.full(stressor_presence_array.shape, 0,
                                numpy.float32)
            result[stressor_presence_array == 1] = 1
            return result

        pygeoprocessing.raster_calculator(
            [(stressor_raster_path, 1)], _no_buffer, target_edt_path,
            _TARGET_GDAL_TYPE_FLOAT32, _TARGET_NODATA_FLOAT32)
        return

    pygeoprocessing.distance_transform_edt((stressor_raster_path, 1),
                                           target_edt_path)
    # We're assuming that we're working with square pixels
    pixel_size = abs(pygeoprocessing.get_raster_info(
        stressor_raster_path)['pixel_size'][0])
    buffer_distance_in_pixels = buffer_distance / pixel_size

    target_edt_raster = gdal.OpenEx(target_edt_path, gdal.GA_Update)
    target_edt_band = target_edt_raster.GetRasterBand(1)
    edt_nodata = target_edt_band.GetNoDataValue()
    for block_info in pygeoprocessing.iterblocks((target_edt_path, 1),
                                                 offset_only=True):
        source_edt_block = target_edt_band.ReadAsArray(**block_info)

        # The pygeoprocessing target datatype for EDT is a float32
        decayed_edt = numpy.full(source_edt_block.shape, 0,
                                 dtype=numpy.float32)
        # The only valid pixels here are those that are within the buffer
        # distance and also valid in the source edt.
        pixels_within_buffer = (
            source_edt_block < buffer_distance_in_pixels)
        nodata_pixels = pygeoprocessing.array_equals_nodata(
            source_edt_block, edt_nodata)
        valid_pixels = ((~nodata_pixels) & pixels_within_buffer)

        if decay_type == 'linear':
            decayed_edt[valid_pixels] = (
                1 - (source_edt_block[valid_pixels] /
                     buffer_distance_in_pixels))
        elif decay_type == 'exponential':
            decayed_edt[valid_pixels] = numpy.exp(
                -source_edt_block[valid_pixels])
        elif decay_type == 'none':
            # everything within the buffer distance has a value of 1
            decayed_edt[valid_pixels] = 1

        # Any values less than 1e-6 are numerically noise and should be 0.
        # Mostly useful for exponential decay, but should also apply to
        # linear decay.
        numpy.where(decayed_edt[valid_pixels] < 1e-6,
                    0.0,
                    decayed_edt[valid_pixels])

        # Reset any nodata pixels that were in the original EDT block.
        decayed_edt[nodata_pixels] = edt_nodata

        target_edt_band.WriteArray(decayed_edt,
                                   xoff=block_info['xoff'],
                                   yoff=block_info['yoff'])

    target_edt_band = None
    target_edt_raster = None


def _calc_criteria(attributes_list, habitat_mask_raster_path,
                   target_criterion_path,
                   decayed_edt_raster_path=None):
    """Calculate Exposure or Consequence for a single habitat/stressor pair.

    Args:
        attributes_list (list): A list of dicts for all of the criteria for
            this criterion type (E or C).  Each dict must have the following
            keys:

                * ``rating``: A numeric criterion rating (if consistent across
                  the whole study area) or the path to a raster with
                  spatially-explicit ratings.  A rating raster must align with
                  ``habitat_mask_raster_path``.
                * ``dq``: The numeric data quality rating for this criterion.
                * ``weight``: The numeric weight for this criterion.

        habitat_mask_raster_path (string): The path to a raster with pixel
            values of 1 indicating presence of habitats, and 0 or nodata
            representing the absence of habitats.
        target_criterion_path (string): The path to where the calculated
            criterion layer should be written.
        decayed_edt_raster_path=None (string or None): If provided, this is the
            path to an raster of weights that should be applied to the
            numerator of the criterion calculation.

    Returns:
        ``None``
    """
    pygeoprocessing.new_raster_from_base(
        habitat_mask_raster_path, target_criterion_path,
        _TARGET_GDAL_TYPE_FLOAT32, [_TARGET_NODATA_FLOAT32])

    habitat_mask_raster = gdal.OpenEx(habitat_mask_raster_path)
    habitat_band = habitat_mask_raster.GetRasterBand(1)

    target_criterion_raster = gdal.OpenEx(target_criterion_path,
                                          gdal.GA_Update)
    target_criterion_band = target_criterion_raster.GetRasterBand(1)

    if decayed_edt_raster_path:
        decayed_edt_raster = gdal.OpenEx(
            decayed_edt_raster_path, gdal.GA_Update)
        decayed_edt_band = decayed_edt_raster.GetRasterBand(1)

    for block_info in pygeoprocessing.iterblocks((habitat_mask_raster_path, 1),
                                                 offset_only=True):
        habitat_mask = habitat_band.ReadAsArray(**block_info)
        valid_mask = (habitat_mask == 1)

        criterion_score = numpy.full(
            habitat_mask.shape, _TARGET_NODATA_FLOAT32, dtype=numpy.float32)
        numerator = numpy.zeros(habitat_mask.shape, dtype=numpy.float32)
        denominator = numpy.zeros(habitat_mask.shape, dtype=numpy.float32)
        for attribute_dict in attributes_list:
            # A rating of 0 means that the criterion should be ignored.
            # RATING may be either a number or a raster.
            try:
                rating = float(attribute_dict['rating'])
                if rating == 0:
                    continue
            except ValueError:
                # When rating is a string filepath, it represents a raster.
                try:
                    # Opening a raster is fairly inexpensive, so it should be
                    # fine to re-open the raster on each block iteration.
                    rating_raster = gdal.OpenEx(attribute_dict['rating'])
                    rating_band = rating_raster.GetRasterBand(1)
                    rating_nodata = rating_band.GetNoDataValue()
                    rating = rating_band.ReadAsArray(**block_info)[valid_mask]

                    # Any habitat pixels with a nodata rating (no rating
                    # specified by the user) should be
                    # interpreted as having a rating of 0.
                    rating[pygeoprocessing.array_equals_nodata(
                        rating, rating_nodata)] = 0
                finally:
                    rating_band = None
                    rating_raster = None
            data_quality = attribute_dict['dq']
            weight = attribute_dict['weight']

            # The (data_quality + weight) denominator running sum is
            # re-calculated for each block.  While this is inefficient,
            # ``dq`` and ``weight`` are always scalars and so the wasted CPU
            # time is pretty trivial, even on large habitat/stressor matrices.
            # Plus, it's way easier to read and more maintainable to just have
            # everything be recalculated.
            numerator[valid_mask] += (rating / (data_quality * weight))
            denominator[valid_mask] += (1 / (data_quality * weight))

        # This is not clearly documented in the UG, but in the source code of
        # previous (3.3.1, 3.10.2) versions of HRA, the numerator is multiplied
        # by the stressor's weighted distance raster.
        # This will give highest values to pixels that overlap stressors and
        # decaying values further away from the overlapping pixels.
        if decayed_edt_raster_path:
            numerator[valid_mask] *= decayed_edt_band.ReadAsArray(
                **block_info)[valid_mask]

        # It is possible for the user to skip all pairwise calculations by
        # setting the rating to 0 for every habitat/stressor combination.
        # Doing so will leave the denominator at 0, resulting in a numpy
        # divide-by-zero warning and +/- inf pixel values.
        # JD is making the call to instead write out a raster filled with 0
        # everywhere it is valid.
        if numpy.sum(denominator[valid_mask]) == 0:
            criterion_score[valid_mask] = 0
        else:
            # This is the normal calculation when the user has defined at least
            # one rating for a habitat/stressor pair.
            criterion_score[valid_mask] = (
                numerator[valid_mask] / denominator[valid_mask])

        target_criterion_band.WriteArray(criterion_score,
                                         xoff=block_info['xoff'],
                                         yoff=block_info['yoff'])
    decayed_edt_band = None
    decayed_edt_raster = None
    target_criterion_band = None
    target_criterion_raster = None


def _calculate_pairwise_risk(habitat_mask_raster_path, exposure_raster_path,
                             consequence_raster_path, risk_equation,
                             target_risk_raster_path):
    """Calculate risk for a single habitat/stressor pair.

    Args:
        habitat_mask_raster_path (string): The path to a habitat mask raster,
            where pixels with a value of 1 represent the presence of habitats,
            and values of 0 or nodata represent the absence of habitats.
        exposure_raster_path (string): The path to a raster of computed
            exposure scores.  Must align with ``habitat_mask_raster_path``.
        consequence_raster_path (string): The path to a raster of computed
            consequence scores.  Must align with ``habitat_mask_raster_path``.
        risk_equation (string): The risk equation to use.  Must be one of
            ``multiplicative`` or ``euclidean``.

    Returns:
        ``None``

    Raises:
        ``AssertionError`` when an invalid risk equation is provided.
    """
    risk_equation = risk_equation.lower()
    if risk_equation not in MODEL_SPEC.get_input('risk_eq').list_options():
        raise AssertionError(
            f'Invalid risk equation {risk_equation} provided')

    def _calculate_risk(habitat_mask, exposure, consequence):
        habitat_pixels = (habitat_mask == 1)
        risk_array = numpy.full(habitat_mask.shape, _TARGET_NODATA_FLOAT32,
                                dtype=numpy.float32)
        if risk_equation == 'multiplicative':
            risk_array[habitat_pixels] = (
                exposure[habitat_pixels] * consequence[habitat_pixels])
        else:  # risk_equation == 'euclidean':
            # The numpy.maximum guards against low E, C values ending up less
            # than 1 and, when squared, have positive, larger-than-reasonable
            # risk values.
            risk_array[habitat_pixels] = numpy.sqrt(
                numpy.maximum(0, (exposure[habitat_pixels] - 1)) ** 2 +
                numpy.maximum(0, (consequence[habitat_pixels] - 1)) ** 2)
        return risk_array

    pygeoprocessing.raster_calculator(
        [(habitat_mask_raster_path, 1),
         (exposure_raster_path, 1),
         (consequence_raster_path, 1)],
        _calculate_risk, target_risk_raster_path, _TARGET_GDAL_TYPE_FLOAT32,
        _TARGET_NODATA_FLOAT32)


def _reclassify_score(habitat_mask, max_pairwise_risk, scores):
    """Reclassify risk scores into high/medium/low.

    The output raster will break values into 3 buckets based on the
    ``max_pairwise_risk`` (shortened here to "MPR"):

        * Scores in the range [ 0, MPR*(1/3) ) have a value of 1
          indicating low risk.
        * Scores in the range [ MPR*(1/3), MPR*(2/3) ) have a value of 2
          indicating medium risk.
        * Scores in the range [ MPR*(2/3), infinity ) have a value of 3
          indicating high risk.

    Args:
        habitat_mask (numpy.array): A numpy array where 1 indicates presence of
            habitats and 0 or ``_TARGET_NODATA_BYTE`` indicate absence.
        max_pairwise_risk (float): The maximum likely pairwise risk value.
        scores (numpy.array): A numpy array of floating-point risk scores.

    Returns:
        ``reclassified`` (numpy.array): An unsigned byte numpy array of a
            shape/size matching ``habitat_mask`` and ``scores``.
    """
    habitat_pixels = (habitat_mask == 1)
    reclassified = numpy.full(habitat_mask.shape, _TARGET_NODATA_BYTE,
                              dtype=numpy.uint8)
    reclassified[habitat_pixels] = numpy.digitize(
        scores[habitat_pixels],
        [0, max_pairwise_risk*(1/3), max_pairwise_risk*(2/3)],
        right=True)  # bins[i-1] >= x > bins[i]
    return reclassified


def _maximum_reclassified_score(habitat_mask, *risk_classes):
    """Determine the maximum risk score in a stack of risk rasters.

    Args:
        habitat_mask (numpy.array): A numpy array where values of 1 indicate
            presence of habitats.  Values of 0 or ``_TARGET_NODATA_BYTE``
            indicate absence of habitats.
        *risk_classes (list of numpy.arrays): A variable number of unsigned
            integer reclassified risk arrays.

    Returns:
        A numpy.array with each pixel that has habitats on it has the highest
        risk score in the ``risk_classes`` stack of arrays.
    """
    target_matrix = numpy.zeros(habitat_mask.shape, dtype=numpy.uint8)
    valid_pixels = (habitat_mask == 1)

    for risk_class_matrix in risk_classes:
        target_matrix[valid_pixels] = numpy.maximum(
            target_matrix[valid_pixels], risk_class_matrix[valid_pixels])

    target_matrix[~valid_pixels] = _TARGET_NODATA_BYTE
    return target_matrix


def _sum_rasters(raster_path_list, target_nodata, target_datatype,
                 target_result_path, normalize=False):
    """Sum a stack of rasters.

    Where all rasters agree about nodata, the output raster will also be
    nodata.  Otherwise, pixel values will be the sum of the stack, where nodata
    values are converted to 0.

    Args:
        raster_path_list (list): list of raster paths to sum
        target_nodata (float): desired target nodata value
        target_datatype (int): The GDAL ``GDT_*`` type of the output raster
        target_result_path (string): path to write out the sum raster
        normalize=False (bool): whether to normalize each pixel value by the
            number of valid pixels in the stack.  Defaults to False.

    Returns:
        ``None``
    """
    nodata_list = [pygeoprocessing.get_raster_info(
        path)['nodata'][0] for path in raster_path_list]

    def _sum_op(*array_list):
        result = numpy.zeros(array_list[0].shape, dtype=numpy.float32)
        pixels_have_valid_values = numpy.zeros(result.shape, dtype=bool)
        valid_pixel_count = numpy.zeros(result.shape, dtype=numpy.uint16)
        for array, nodata in zip(array_list, nodata_list):
            non_nodata_pixels = ~pygeoprocessing.array_equals_nodata(array, nodata)
            pixels_have_valid_values |= non_nodata_pixels
            valid_pixel_count += non_nodata_pixels

            result[non_nodata_pixels] += array[non_nodata_pixels]

        if normalize:
            result[pixels_have_valid_values] /= valid_pixel_count[
                pixels_have_valid_values]
        result[~pixels_have_valid_values] = target_nodata
        return result

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list],
        _sum_op, target_result_path, target_datatype, target_nodata)


def _override_datastack_archive_criteria_table_path(
        criteria_table_path, data_dir, known_files):
    """Prepare the HRA criteria_table_path input for archiving in a datastack.

    This function rewrites a criteria table (which may contain spatial ratings
    layers), copying any spatial layers into the data directory provided, which
    will be included in the datastack archive.

    Args:
        criteria_table_path (string): The path to the criteria table provided
            by the user.
        data_dir (string): the path to where data files will be written.
        known_files (dict): a dict mapping original source files to their
            location within the ``data_dir``.

    Note:
        The ``known_files`` dict may be modified by this function.

    Returns:
        The path to where the rewritten criteria table path is located within
        the data dir.
    """
    args_key = 'criteria_table_path'
    criteria_table_array = utils.read_csv_to_dataframe(
        criteria_table_path, header=None).to_numpy()
    contained_data_dir = os.path.join(data_dir, f'{args_key}_data')

    known_rating_cols = set()
    for row in range(1, len(criteria_table_array)):  # skip named habitats
        # When we encounter an empty row, reset the known ratings columns in
        # case one of the sub-tables changes the order around.
        try:
            if numpy.all(numpy.isnan(
                    criteria_table_array[row].astype(numpy.float32))):
                known_rating_cols = set()
                continue
        except ValueError:
            # ValueError when there are any string values in the row
            pass

        if not known_rating_cols:
            for col in range(1, len(criteria_table_array[0])):
                if criteria_table_array[row, col] == 'RATING':
                    known_rating_cols.add(col)
            continue  # skip the RATING headers row.

        for col in known_rating_cols:
            value = criteria_table_array[row, col]
            try:
                float(value)
                continue
            except ValueError:
                # When value is obviously not a number.
                pass

            # Expand the path if it's not absolute
            value = utils.expand_path(value, criteria_table_path)
            if not os.path.exists(value):
                LOGGER.warning(f'File not found: {value}')
                continue
            if value in known_files:
                LOGGER.info(
                    f"File {value} already known, perhaps from another "
                    f"cell or table.  Reusing {known_files[value]}")
                criteria_table_array[row, col] = known_files[value]
            else:
                dir_for_this_spatial_data = os.path.join(
                    contained_data_dir,
                    os.path.splitext(os.path.basename(value))[0])
                LOGGER.info(f"Copying spatial file {value} --> "
                            f"{dir_for_this_spatial_data}")
                new_path = utils.copy_spatial_files(
                    value, dir_for_this_spatial_data)
                criteria_table_array[row, col] = new_path
                known_files[value] = new_path

    target_output_path = os.path.join(data_dir, f'{args_key}.csv')
    numpy.savetxt(target_output_path, criteria_table_array, delimiter=',',
                  fmt="%s", encoding="UTF-8")
    return target_output_path


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
    return validation.validate(args, MODEL_SPEC)
