"""Stormwater Retention."""
import logging
import math
import os

import numpy
import pygeoprocessing
import pygeoprocessing.kernels
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

# a constant nodata value to use for intermediates and outputs
FLOAT_NODATA = -1
UINT8_NODATA = 255
UINT16_NODATA = 65535
NONINTEGER_SOILS_RASTER_MESSAGE = 'Soil group raster data type must be integer'

MODEL_SPEC = spec.ModelSpec(
    model_id="stormwater",
    model_title=gettext("Urban Stormwater Retention"),
    userguide="stormwater.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_path", "soil_group_path", "precipitation_path", "biophysical_table"],
        ["adjust_retention_ratios", "retention_radius", "road_centerlines_path"],
        ["aggregate_areas_path", "replacement_cost"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="lulc_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover type must be"
                " assigned a unique integer code."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.SOIL_GROUP,
        spec.SingleBandRasterInput(
            id="precipitation_path",
            name=gettext("precipitation"),
            about=gettext("Map of average annual precipitation."),
            data_type=float,
            units=u.millimeter / u.year,
            projected=None
        ),
        spec.CSVInput(
            id="biophysical_table",
            name=gettext("Biophysical table"),
            about=gettext(
                "Table mapping each LULC code found in the LULC raster to biophysical"
                " data about that LULC class. If you provide the percolation coefficient"
                " column (PE_[X]) for any soil group, you must provide it for all four"
                " soil groups."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.NumberInput(
                    id="emc_[POLLUTANT]",
                    about=gettext(
                        "Event mean concentration of the pollutant in stormwater. You may"
                        " include any number of these columns for different pollutants,"
                        " or none at all."
                    ),
                    units=u.milligram / u.liter
                ),
                spec.RatioInput(
                    id="rc_a",
                    about=gettext("Stormwater runoff coefficient for soil group A"),
                    units=None
                ),
                spec.RatioInput(
                    id="rc_b",
                    about=gettext("Stormwater runoff coefficient for soil group B"),
                    units=None
                ),
                spec.RatioInput(
                    id="rc_c",
                    about=gettext("Stormwater runoff coefficient for soil group C"),
                    units=None
                ),
                spec.RatioInput(
                    id="rc_d",
                    about=gettext("Stormwater runoff coefficient for soil group D"),
                    units=None
                ),
                spec.RatioInput(
                    id="pe_a",
                    about=gettext("Stormwater percolation coefficient for soil group A"),
                    required=False,
                    units=None
                ),
                spec.RatioInput(
                    id="pe_b",
                    about=gettext("Stormwater percolation coefficient for soil group B"),
                    required=False,
                    units=None
                ),
                spec.RatioInput(
                    id="pe_c",
                    about=gettext("Stormwater percolation coefficient for soil group C"),
                    required=False,
                    units=None
                ),
                spec.RatioInput(
                    id="pe_d",
                    about=gettext("Stormwater percolation coefficient for soil group D"),
                    required=False,
                    units=None
                ),
                spec.BooleanInput(
                    id="is_connected",
                    about=(
                        "Enter 1 if the LULC class is a connected impervious surface, 0"
                        " if not. This column is only used if the 'adjust retention"
                        " ratios' option is selected. If 'adjust retention ratios' is"
                        " selected and this column exists, the adjustment algorithm takes"
                        " into account the LULC as well as road centerlines. If this"
                        " column does not exist, only the road centerlines are used."
                    ),
                    required=False
                )
            ],
            index_col="lucode"
        ),
        spec.BooleanInput(
            id="adjust_retention_ratios",
            name=gettext("Adjust retention ratios"),
            about=gettext(
                "If true, adjust retention ratios. The adjustment algorithm accounts for"
                " drainage effects of nearby impervious surfaces which are directly"
                " connected to artifical urban drainage channels (typically roads,"
                " parking lots, etc.) Connected impervious surfaces are indicated by the"
                " is_connected columnin the biophysical table and/or the road centerlines"
                " vector."
            )
        ),
        spec.NumberInput(
            id="retention_radius",
            name=gettext("Retention radius"),
            about=(
                "Radius around each pixel to adjust retention ratios. Measured in raster"
                " coordinate system units. For the adjustment algorithm, a pixel is"
                " 'near' a connected impervious surface if its centerpoint is within this"
                " radius of connected-impervious LULC and/or a road centerline."
            ),
            required="adjust_retention_ratios",
            allowed="adjust_retention_ratios",
            units=u.other
        ),
        spec.VectorInput(
            id="road_centerlines_path",
            name=gettext("Road centerlines"),
            about=gettext("Map of road centerlines"),
            required="adjust_retention_ratios",
            allowed="adjust_retention_ratios",
            geometry_types={"LINESTRING", "MULTILINESTRING"},
            fields=[],
            projected=None
        ),
        spec.AOI.model_copy(update=dict(
            id="aggregate_areas_path",
            about=gettext(
                "Areas over which to aggregate results (typically watersheds or"
                " sewersheds). The aggregated data are: average retention ratio and total"
                " retention volume; average percolation ratio and total percolation"
                " volume if percolation data was provided; total retention value if"
                " replacement cost was provided; and total avoided pollutant load for"
                " each pollutant provided."
            ),
            required=False
        )),
        spec.NumberInput(
            id="replacement_cost",
            name=gettext("Replacement cost"),
            about=gettext("Replacement cost of stormwater retention devices"),
            required=False,
            units=u.currency / u.meter**3
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="retention_ratio",
            path="retention_ratio.tif",
            about=gettext(
                "Map of the stormwater retention ratio, derived from the LULC raster and"
                " biophysical table RC_x columns."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="adjusted_retention_ratio",
            path="adjusted_retention_ratio.tif",
            about=gettext(
                "Map of the adjusted retention ratio, calculated according to equation"
                " (124) from the ‘retention_ratio, ratio_average, near_road’, and"
                " ‘near_impervious_lulc’ intermediate outputs."
            ),
            created_if="adjust_retention_ratios",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="retention_volume",
            path="retention_volume.tif",
            about=gettext("Map of retention volume."),
            data_type=float,
            units=u.meter**3 / u.year
        ),
        spec.SingleBandRasterOutput(
            id="percolation_ratio",
            path="percolation_ratio.tif",
            about=gettext(
                "Map of percolation ratio derived by cross-referencing the LULC and soil"
                " group rasters with the biophysical table."
            ),
            # created_if="percolation",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="percolation_volume",
            path="percolation_volume.tif",
            about=gettext("Map of percolation (potential aquifer recharge) volume."),
            # created_if="percolation",
            data_type=float,
            units=u.meter**3 / u.year
        ),
        spec.SingleBandRasterOutput(
            id="runoff_ratio",
            path="runoff_ratio.tif",
            about=gettext(
                "Map of the stormwater runoff ratio. This is the inverse of"
                " ‘retention_ratio.tif’"
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="runoff_volume",
            path="runoff_volume.tif",
            about=gettext("Map of runoff volume."),
            data_type=float,
            units=u.meter**3 / u.year
        ),
        spec.SingleBandRasterOutput(
            id="retention_value",
            path="retention_value.tif",
            about=gettext("Map of the value of water retained."),
            created_if="replacement_cost",
            data_type=float,
            units=u.currency / u.year
        ),
        spec.VectorOutput(
            id="aggregate",
            path="aggregate_data.gpkg",
            about=gettext(
                "Map of aggregate data. This is identical to the aggregate areas input"
                " vector, but each polygon is given additional fields with the aggregate"
                " data."
            ),
            created_if="aggregate_areas_path",
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.RatioOutput(
                    id="mean_retention_ratio",
                    about=gettext("Average retention ratio over this polygon")
                ),
                spec.NumberOutput(
                    id="total_retention_volume",
                    about=gettext("Total retention volume over this polygon"),
                    units=u.meter**3 / u.year
                ),
                spec.RatioOutput(
                    id="mean_runoff_ratio",
                    about=gettext("Average runoff coefficient over this polygon")
                ),
                spec.NumberOutput(
                    id="total_runoff_volume",
                    about=gettext("Total runoff volume over this polygon"),
                    units=u.meter**3 / u.year
                ),
                spec.RatioOutput(
                    id="mean_percolation_ratio",
                    about=gettext(
                        "Average percolation (recharge) ratio over this polygon"
                    ),
                    # created_if="percolation"
                ),
                spec.NumberOutput(
                    id="total_percolation_volume",
                    about=gettext(
                        "Total volume of potential aquifer recharge over this polygon"
                    ),
                    # created_if="percolation",
                    units=u.meter**3 / u.year
                ),
                spec.NumberOutput(
                    id="[POLLUTANT]_total_avoided_load",
                    about=gettext(
                        "Total avoided (retained) amount of pollutant over this polygon"
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="[POLLUTANT]_total_load",
                    about=gettext(
                        "Total amount of pollutant in runoff over this polygon"
                    ),
                    units=u.kilogram / u.year
                ),
                spec.NumberOutput(
                    id="total_retention_value",
                    about=gettext(
                        "Total value of the retained volume of water over this polygon"
                    ),
                    created_if="replacement_cost",
                    units=u.currency / u.year
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="avoided_pollutant_load_[POLLUTANT]",
            path="avoided_pollutant_load_[POLLUTANT].tif",
            about=gettext("Avoided load of each pollutant"),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="actual_pollutant_load_[POLLUTANT]",
            path="actual_pollutant_load_[POLLUTANT].tif",
            about=gettext("Actual load of each pollutant"),
            data_type=float,
            units=u.kilogram / u.year
        ),
        spec.SingleBandRasterOutput(
            id="lulc_aligned",
            path="intermediate/lulc_aligned.tif",
            about=gettext(
                "Copy of the soil group raster input, cropped to the intersection"
                " of the three raster inputs."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="soil_group_aligned",
            path="intermediate/soil_group_aligned.tif",
            about=gettext(
                "Copy of the soil group raster input, aligned to the LULC raster"
                " and cropped to the intersection of the three raster inputs."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="precipitation_aligned",
            path="intermediate/precipitation_aligned.tif",
            about=gettext(
                "Copy of the precipitation raster input, aligned to the LULC"
                " raster and cropped to the intersection of the three raster"
                " inputs."
            ),
            data_type=float,
            units=u.millimeter / u.year
        ),
        spec.VectorOutput(
            id="reprojected_centerlines",
            path="intermediate/reprojected_centerlines.gpkg",
            about=gettext(
                "Copy of the road centerlines vector input, reprojected to the"
                " LULC raster projection."
            ),
            geometry_types={"LINESTRING", "MULTILINESTRING"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="rasterized_centerlines",
            path="intermediate/rasterized_centerlines.tif",
            about=gettext(
                "A rasterized version of the reprojected centerlines vector,"
                " where 1 means the pixel is a road and 0 means it isn’t."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="is_connected_lulc",
            path="intermediate/is_connected_lulc.tif",
            about=gettext(
                "A binary raster derived from the LULC raster and biophysical"
                " table is_connected column, where 1 means the pixel has a"
                " directly-connected impervious LULC type, and 0 means it does"
                " not."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="road_distance",
            path="intermediate/road_distance.tif",
            about=gettext(
                "A raster derived from the rasterized centerlines map, where each"
                " pixel’s value is its minimum distance to a road pixel (measured"
                " centerpoint-to-centerpoint)."
            ),
            data_type=float,
            units=u.pixel
        ),
        spec.SingleBandRasterOutput(
            id="connected_lulc_distance",
            path="intermediate/connected_lulc_distance.tif",
            about=gettext(
                "A raster derived from the is_connected_lulc map, where each"
                " pixel’s value is its minimum distance to a connected impervious"
                " LULC pixel (measured centerpoint-to-centerpoint)."
            ),
            data_type=float,
            units=u.pixel
        ),
        spec.SingleBandRasterOutput(
            id="near_road",
            path="intermediate/near_road.tif",
            about=gettext(
                "A binary raster derived from the road_distance map, where 1"
                " means the pixel is within the retention radius of a road pixel,"
                " and 0 means it isn’t."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="near_connected_lulc",
            path="intermediate/near_connected_lulc.tif",
            about=gettext(
                "A binary raster derived from the connected_lulc_distance map,"
                " where 1 means the pixel is within the retention radius of a"
                " connected impervious LULC pixel, and 0 means it isn’t."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="search_kernel",
            path="intermediate/search_kernel.tif",
            about=gettext(
                "A binary raster representing the search kernel that is convolved"
                " with the retention_ratio raster to calculate the averaged"
                " retention ratio within the retention radius of each pixel."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="ratio_average",
            path="intermediate/ratio_average.tif",
            about=gettext(
                "A raster where each pixel’s value is the average of its"
                " neighborhood of pixels in the retention_ratio map, calculated"
                " by convolving the search kernel with the retention ratio"
                " raster."
            ),
            data_type=float,
            units=None
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Execute the urban stormwater retention model.

    Args:
        args['workspace_dir'] (str): path to a directory to write intermediate
            and final outputs. May already exist or not.
        args['results_suffix'] (str, optional): string to append to all output
            file names from this model run
        args['n_workers'] (int): if present, indicates how many worker
            processes should be used in parallel processing. -1 indicates
            single process mode, 0 is single process but non-blocking mode,
            and >= 1 is number of processes.
        args['lulc_path'] (str): path to LULC raster
        args['soil_group_path'] (str): path to soil group raster, where pixel
            values 1, 2, 3, 4 correspond to groups A, B, C, D
        args['precipitation_path'] (str): path to raster of total annual
            precipitation in millimeters
        args['biophysical_table'] (str): path to biophysical table with columns
            'lucode', 'EMC_x' (event mean concentration mg/L) for each
            pollutant x, 'RC_y' (retention coefficient) and 'PE_y'
            (percolation coefficient) for each soil group y, and
            'is_connected' if args['adjust_retention_ratios'] is True
        args['adjust_retention_ratios'] (bool): If True, apply retention ratio
            adjustment algorithm.
        args['retention_radius'] (float): If args['adjust_retention_ratios']
            is True, use this radius in the adjustment algorithm.
        args['road_centerliens_path'] (str): Path to linestring vector of road
            centerlines. Only used if args['adjust_retention_ratios'] is True.
        args['aggregate_areas_path'] (str): Optional path to polygon vector of
            areas to aggregate results over.
        args['replacement_cost'] (float): Cost to replace stormwater retention
            devices in units currency per cubic meter

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    # get the necessary base raster info
    source_lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_path'])
    pixel_size = source_lulc_raster_info['pixel_size']
    # in case the input raster doesn't have square pixels, take the average
    # all the rasters are warped to this square pixel size in the align task
    avg_pixel_size = (abs(pixel_size[0]) + abs(pixel_size[1])) / 2
    pixel_area = abs(pixel_size[0] * pixel_size[1])

    lulc_nodata = source_lulc_raster_info['nodata'][0]
    precipitation_nodata = pygeoprocessing.get_raster_info(
        args['precipitation_path'])['nodata'][0]
    soil_group_nodata = pygeoprocessing.get_raster_info(
        args['soil_group_path'])['nodata'][0]

    # Align all three input rasters to the same projection
    align_inputs = [args['lulc_path'],
                    args['soil_group_path'], args['precipitation_path']]
    align_outputs = [
        file_registry['lulc_aligned'],
        file_registry['soil_group_aligned'],
        file_registry['precipitation_aligned']]
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            align_inputs,
            align_outputs,
            ['near' for _ in align_inputs],
            (avg_pixel_size, -avg_pixel_size),
            'intersection'),
        kwargs={'raster_align_index': 0},
        target_path_list=align_outputs,
        task_name='align input rasters')

    # Build a lookup dictionary mapping each LULC code to its row
    # sort by the LULC codes upfront because we use the sorted list in multiple
    # places. it's more efficient to do this once.
    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table').get_validated_dataframe(
        args['biophysical_table']).sort_index()
    sorted_lucodes = biophysical_df.index.to_list()

    # convert the nested dictionary in to a 2D array where rows are LULC codes
    # in sorted order and columns correspond to soil groups in order
    # this facilitates efficiently looking up the ratio values with numpy
    #
    # Biophysical table has runoff coefficents so subtract
    # from 1 to get retention coefficient.
    # add a placeholder in column 0 so that the soil groups 1, 2, 3, 4 line
    # up with their indices in the array. this is more efficient than
    # decrementing the whole soil group array by 1.
    retention_ratio_array = numpy.array([
        1 - biophysical_df[f'rc_{soil_group}'].to_numpy()
        for soil_group in ['a', 'b', 'c', 'd']], dtype=numpy.float32).T

    # Calculate stormwater retention ratio and volume from
    # LULC, soil groups, biophysical data, and precipitation
    retention_ratio_task = task_graph.add_task(
        func=lookup_ratios,
        args=(
            file_registry['lulc_aligned'],
            file_registry['soil_group_aligned'],
            retention_ratio_array,
            sorted_lucodes,
            file_registry['retention_ratio']),
        target_path_list=[file_registry['retention_ratio']],
        dependent_task_list=[align_task],
        task_name='calculate stormwater retention ratio'
    )

    # (Optional) adjust stormwater retention ratio using roads
    if args['adjust_retention_ratios']:
        reproject_roads_task = task_graph.add_task(
            func=pygeoprocessing.reproject_vector,
            args=(
                args['road_centerlines_path'],
                source_lulc_raster_info['projection_wkt'],
                file_registry['reprojected_centerlines']),
            kwargs={'driver_name': 'GPKG'},
            target_path_list=[file_registry['reprojected_centerlines']],
            task_name='reproject road centerlines vector to match rasters',
            dependent_task_list=[])

        # for gdal.GDT_Byte, setting the datatype is not enough
        # must also set PIXELTYPE=DEFAULT to guarantee unsigned byte type
        # otherwise, `new_raster_from_base` may pass down the
        # PIXELTYPE=SIGNEDBYTE attribute from a signed base to the new raster
        creation_opts = pygeoprocessing.geoprocessing_core.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1]  # noqa
        unsigned_byte_creation_opts = creation_opts + ('PIXELTYPE=DEFAULT',)
        # pygeoprocessing.rasterize expects the target raster to already exist
        make_raster_task = task_graph.add_task(
            func=pygeoprocessing.new_raster_from_base,
            args=(
                file_registry['lulc_aligned'],
                file_registry['rasterized_centerlines'],
                gdal.GDT_Byte,
                [UINT8_NODATA]),
            kwargs={
                'raster_driver_creation_tuple': (
                    'GTIFF', unsigned_byte_creation_opts)
            },
            target_path_list=[file_registry['rasterized_centerlines']],
            task_name='create raster to pass to pygeoprocessing.rasterize',
            dependent_task_list=[align_task])

        rasterize_centerlines_task = task_graph.add_task(
            func=pygeoprocessing.rasterize,
            args=(
                file_registry['reprojected_centerlines'],
                file_registry['rasterized_centerlines'],
                [1]),
            target_path_list=[file_registry['rasterized_centerlines']],
            task_name='rasterize road centerlines vector',
            dependent_task_list=[make_raster_task, reproject_roads_task])

        # Make a boolean raster showing which pixels are within the given
        # radius of a road centerline
        near_road_task = task_graph.add_task(
            func=is_near,
            args=(
                file_registry['rasterized_centerlines'],
                args['retention_radius'] / avg_pixel_size,  # convert the radius to pixels
                file_registry['road_distance'],
                file_registry['near_road']),
            target_path_list=[
                file_registry['road_distance'],
                file_registry['near_road']],
            task_name='find pixels within radius of road centerlines',
            dependent_task_list=[rasterize_centerlines_task])

        # Make a boolean raster indicating which pixels are directly
        # connected impervious LULC type
        connected_lulc_task = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (file_registry['lulc_aligned'], 1),
                biophysical_df['is_connected'].astype(int).to_dict(),
                file_registry['is_connected_lulc'],
                gdal.GDT_Byte,
                UINT8_NODATA),
            target_path_list=[file_registry['is_connected_lulc']],
            task_name='calculate binary connected lulc raster',
            dependent_task_list=[align_task]
        )

        # Make a boolean raster showing which pixels are within the given
        # radius of connected land cover
        near_connected_lulc_task = task_graph.add_task(
            func=is_near,
            args=(
                file_registry['is_connected_lulc'],
                args['retention_radius'] / avg_pixel_size,  # convert the radius to pixels
                file_registry['connected_lulc_distance'],
                file_registry['near_connected_lulc']),
            target_path_list=[
                file_registry['connected_lulc_distance'],
                file_registry['near_connected_lulc']],
            task_name='find pixels within radius of connected lulc',
            dependent_task_list=[connected_lulc_task])

        average_ratios_task = task_graph.add_task(
            func=raster_average,
            args=(
                file_registry['retention_ratio'],
                args['retention_radius'],
                file_registry['search_kernel'],
                file_registry['ratio_average']),
            target_path_list=[file_registry['ratio_average']],
            task_name='average retention ratios within radius',
            dependent_task_list=[retention_ratio_task])

        # Using the averaged retention ratio raster and boolean
        # "within radius" rasters, adjust the retention ratios
        adjust_retention_ratio_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=adjust_op,
                rasters=[
                    file_registry['retention_ratio'],
                    file_registry['ratio_average'],
                    file_registry['near_connected_lulc'],
                    file_registry['near_road']],
                target_path=file_registry['adjusted_retention_ratio'],
                target_nodata=FLOAT_NODATA),
            target_path_list=[file_registry['adjusted_retention_ratio']],
            task_name='adjust stormwater retention ratio',
            dependent_task_list=[retention_ratio_task, average_ratios_task,
                                 near_connected_lulc_task, near_road_task])

        final_retention_ratio_path = file_registry['adjusted_retention_ratio']
        final_retention_ratio_task = adjust_retention_ratio_task
    else:
        final_retention_ratio_path = file_registry['retention_ratio']
        final_retention_ratio_task = retention_ratio_task

    # Calculate stormwater retention volume from ratios and precipitation
    retention_volume_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (final_retention_ratio_path, 1),
            (file_registry['precipitation_aligned'], 1),
            (precipitation_nodata, 'raw'),
            (pixel_area, 'raw')],
            volume_op,
            file_registry['retention_volume'],
            gdal.GDT_Float32,
            FLOAT_NODATA),
        target_path_list=[file_registry['retention_volume']],
        dependent_task_list=[align_task, final_retention_ratio_task],
        task_name='calculate stormwater retention volume'
    )

    # Calculate stormwater runoff ratios and volume
    runoff_ratio_task = task_graph.add_task(
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=retention_to_runoff_op,
            rasters=[final_retention_ratio_path],
            target_path=file_registry['runoff_ratio'],
            target_nodata=FLOAT_NODATA),
        target_path_list=[file_registry['runoff_ratio']],
        dependent_task_list=[final_retention_ratio_task],
        task_name='calculate stormwater runoff ratio'
    )

    runoff_volume_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=([
            (file_registry['runoff_ratio'], 1),
            (file_registry['precipitation_aligned'], 1),
            (precipitation_nodata, 'raw'),
            (pixel_area, 'raw')],
            volume_op,
            file_registry['runoff_volume'],
            gdal.GDT_Float32,
            FLOAT_NODATA),
        target_path_list=[file_registry['runoff_volume']],
        dependent_task_list=[align_task, runoff_ratio_task],
        task_name='calculate stormwater runoff volume'
    )
    aggregation_task_dependencies = [retention_volume_task]
    data_to_aggregate = [
        # tuple of (raster path, output field name, op) for aggregation
        (final_retention_ratio_path, 'mean_retention_ratio', 'mean'),
        (file_registry['retention_volume'], 'total_retention_volume', 'sum'),
        (file_registry['runoff_ratio'], 'mean_runoff_ratio', 'mean'),
        (file_registry['runoff_volume'], 'total_runoff_volume', 'sum')]

    # (Optional) Calculate stormwater percolation ratio and volume from
    # LULC, soil groups, biophysical table, and precipitation
    if 'pe_a' in biophysical_df.columns:
        LOGGER.info('percolation data detected in biophysical table. '
                    'Will calculate percolation ratio and volume rasters.')
        percolation_ratio_array = numpy.array([
            biophysical_df[f'pe_{soil_group}'].to_numpy()
            for soil_group in ['a', 'b', 'c', 'd']], dtype=numpy.float32).T
        percolation_ratio_task = task_graph.add_task(
            func=lookup_ratios,
            args=(
                file_registry['lulc_aligned'],
                file_registry['soil_group_aligned'],
                percolation_ratio_array,
                sorted_lucodes,
                file_registry['percolation_ratio']),
            target_path_list=[file_registry['percolation_ratio']],
            dependent_task_list=[align_task],
            task_name='calculate stormwater percolation ratio')

        percolation_volume_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (file_registry['percolation_ratio'], 1),
                (file_registry['precipitation_aligned'], 1),
                (precipitation_nodata, 'raw'),
                (pixel_area, 'raw')],
                volume_op,
                file_registry['percolation_volume'],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[file_registry['percolation_volume']],
            dependent_task_list=[align_task, percolation_ratio_task],
            task_name='calculate stormwater retention volume'
        )
        aggregation_task_dependencies.append(percolation_volume_task)
        data_to_aggregate.append((file_registry['percolation_ratio'],
                                 'mean_percolation_ratio', 'mean'))
        data_to_aggregate.append((file_registry['percolation_volume'],
                                 'total_percolation_volume', 'sum'))

    # get all EMC columns from an arbitrary row in the dictionary
    # strip the first four characters off 'EMC_pollutant' to get pollutant name
    pollutants = [
        col[4:] for col in biophysical_df.columns if col.startswith('emc_')]
    LOGGER.debug(f'Pollutants found in biophysical table: {pollutants}')

    # Calculate avoided pollutant load for each pollutant from retention volume
    # and biophysical table EMC value
    avoided_load_paths = []
    actual_load_paths = []
    for pollutant in pollutants:
        # two output rasters for each pollutant
        avoided_load_paths.append(file_registry['avoided_pollutant_load_[POLLUTANT]', pollutant])
        actual_load_paths.append(file_registry['actual_pollutant_load_[POLLUTANT]', pollutant])
        # make an array mapping each LULC code to the pollutant EMC value
        emc_array = biophysical_df[f'emc_{pollutant}'].to_numpy(dtype=numpy.float32)

        # calculate avoided load from retention volume
        avoided_load_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (file_registry['lulc_aligned'], 1),
                (lulc_nodata, 'raw'),
                (file_registry['retention_volume'], 1),
                (sorted_lucodes, 'raw'),
                (emc_array, 'raw')],
                pollutant_load_op,
                file_registry['avoided_pollutant_load_[POLLUTANT]', pollutant],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[file_registry['avoided_pollutant_load_[POLLUTANT]', pollutant]],
            dependent_task_list=[retention_volume_task],
            task_name=f'calculate avoided pollutant {pollutant} load'
        )
        # calculate actual load from runoff volume
        actual_load_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (file_registry['lulc_aligned'], 1),
                (lulc_nodata, 'raw'),
                (file_registry['runoff_volume'], 1),
                (sorted_lucodes, 'raw'),
                (emc_array, 'raw')],
                pollutant_load_op,
                file_registry['actual_pollutant_load_[POLLUTANT]', pollutant],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[file_registry['actual_pollutant_load_[POLLUTANT]', pollutant]],
            dependent_task_list=[runoff_volume_task],
            task_name=f'calculate actual pollutant {pollutant} load'
        )
        aggregation_task_dependencies += [avoided_load_task, actual_load_task]
        data_to_aggregate.append((file_registry['avoided_pollutant_load_[POLLUTANT]', pollutant],
                                 f'{pollutant}_total_avoided_load', 'sum'))
        data_to_aggregate.append(
            (file_registry['actual_pollutant_load_[POLLUTANT]', pollutant],
                f'{pollutant}_total_load', 'sum'))

    # (Optional) Do valuation if a replacement cost is defined
    # you could theoretically have a cost of 0 which should be allowed
    if args['replacement_cost'] is not None:
        valuation_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=([
                (file_registry['retention_volume'], 1),
                (args['replacement_cost'], 'raw')],
                retention_value_op,
                file_registry['retention_value'],
                gdal.GDT_Float32,
                FLOAT_NODATA),
            target_path_list=[file_registry['retention_value']],
            dependent_task_list=[retention_volume_task],
            task_name='calculate stormwater retention value'
        )
        aggregation_task_dependencies.append(valuation_task)
        data_to_aggregate.append(
            (file_registry['retention_value'], 'total_retention_value', 'sum'))

    # (Optional) Aggregate to watersheds if an aggregate vector is defined
    if args['aggregate_areas_path']:
        _ = task_graph.add_task(
            func=aggregate_results,
            args=(
                args['aggregate_areas_path'],
                file_registry['aggregate'],
                source_lulc_raster_info['projection_wkt'],
                data_to_aggregate),
            target_path_list=[file_registry['aggregate']],
            dependent_task_list=aggregation_task_dependencies,
            task_name='aggregate data over polygons'
        )

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def lookup_ratios(lulc_path, soil_group_path, ratio_lookup, sorted_lucodes,
                  output_path):
    """Look up retention/percolation ratios from LULC codes and soil groups.

    Args:
        lulc_array (numpy.ndarray): 2D array of LULC codes
        soil_group_array (numpy.ndarray): 2D array with the same shape as
            ``lulc_array``. Values in {1, 2, 3, 4} corresponding to soil
            groups A, B, C, and D.
        ratio_lookup (numpy.ndarray): 2D array where rows correspond to
            sorted LULC codes and the 4 columns correspond to soil groups
            A, B, C, D in order. Shape: (number of lulc codes, 4).
        sorted_lucodes (list[int]): List of LULC codes sorted from smallest
            to largest. These correspond to the rows of ``ratio_lookup``.
        output_path (str): path to a raster to write out the result. has the
            same shape as the lulc and soil group rasters. Each value is the
            corresponding ratio for that LULC code x soil group pair.

    Returns:
        None
    """
    # insert a column on the left side of the array so that the soil
    # group codes 1-4 line up with their indexes. this is faster than
    # decrementing every value in a large raster.
    ratio_lookup = numpy.insert(ratio_lookup, 0,
                                numpy.zeros(ratio_lookup.shape[0]), axis=1)
    pygeoprocessing.raster_map(
        op=lambda lulc_array, soil_group_array: ratio_lookup[
            # the index of each lucode in the sorted lucodes array
            numpy.digitize(lulc_array, sorted_lucodes, right=True),
            soil_group_array],
        rasters=[lulc_path, soil_group_path],
        target_path=output_path,
        target_nodata=FLOAT_NODATA,
        target_dtype=numpy.float32)


def volume_op(ratio_array, precip_array, precip_nodata, pixel_area):
    """Calculate stormwater volumes from precipitation and ratios.

    Args:
        ratio_array (numpy.ndarray): 2D array of stormwater ratios. Assuming
            that its nodata value is the global FLOAT_NODATA.
        precip_array (numpy.ndarray): 2D array of precipitation amounts
            in millimeters/year
        precip_nodata (float): nodata value for the precipitation array
        pixel_area (float): area of each pixel in m^2

    Returns:
        2D numpy.ndarray of precipitation volumes in m^3/year
    """
    volume_array = numpy.full(ratio_array.shape, FLOAT_NODATA,
                              dtype=numpy.float32)
    valid_mask = ~pygeoprocessing.array_equals_nodata(ratio_array, FLOAT_NODATA)
    if precip_nodata is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(precip_array, precip_nodata)

    # precipitation (mm/yr) * pixel area (m^2) *
    # 0.001 (m/mm) * ratio = volume (m^3/yr)
    volume_array[valid_mask] = (
        precip_array[valid_mask] *
        ratio_array[valid_mask] *
        pixel_area * 0.001)
    return volume_array


def retention_to_runoff_op(retention_array):
    """Calculate runoff ratios from retention ratios: runoff = 1 - retention.

    Args:
        retention_array (numpy.ndarray): array of stormwater retention ratios.
            It is assumed that the nodata value is the global FLOAT_NODATA.

    Returns:
        numpy.ndarray of stormwater runoff ratios
    """
    return 1 - retention_array


def pollutant_load_op(lulc_array, lulc_nodata, volume_array, sorted_lucodes,
                      emc_array):
    """Calculate pollutant loads from EMC and stormwater volumes.

    Used for both actual pollutant load (where `volume_array` is the runoff
    volume) and avoided pollutant load (where `volume_array` is the
    retention volume).

    Args:
        lulc_array (numpy.ndarray): 2D array of LULC codes
        lulc_nodata (int): nodata value for the LULC array
        volume_array (numpy.ndarray): 2D array of stormwater volumes, with the
            same shape as ``lulc_array``. It is assumed that the volume nodata
            value is the global FLOAT_NODATA.
        sorted_lucodes (numpy.ndarray): 1D array of the LULC codes in order
            from smallest to largest
        emc_array (numpy.ndarray): 1D array of pollutant EMC values for each
            lucode. ``emc_array[i]`` is the EMC for the LULC class at
            ``sorted_lucodes[i]``.

    Returns:
        2D numpy.ndarray with the same shape as ``lulc_array``.
        Each value is the avoided pollutant load on that pixel in kg/yr,
        or FLOAT_NODATA if any of the inputs have nodata on that pixel.
    """
    load_array = numpy.full(
        lulc_array.shape, FLOAT_NODATA, dtype=numpy.float32)
    valid_mask = ~pygeoprocessing.array_equals_nodata(volume_array, FLOAT_NODATA)
    if lulc_nodata is not None:
        valid_mask &= ~pygeoprocessing.array_equals_nodata(lulc_array, lulc_nodata)

    # bin each value in the LULC array such that
    # lulc_array[i,j] == sorted_lucodes[lulc_index[i,j]]. thus,
    # emc_array[lulc_index[i,j]] is the EMC for the lucode at lulc_array[i,j]
    lulc_index = numpy.digitize(lulc_array, sorted_lucodes, right=True)
    # need to mask out nodata pixels from lulc_index before indexing emc_array
    # otherwise we get an IndexError when the nodata value is > all LULC values
    valid_lulc_index = lulc_index[valid_mask]
    # EMC for pollutant (mg/L) * 1000 (L/m^3) * 0.000001 (kg/mg) *
    # retention (m^3/yr) = pollutant load (kg/yr)
    load_array[valid_mask] = (emc_array[valid_lulc_index] *
                              0.001 * volume_array[valid_mask])
    return load_array


def retention_value_op(retention_volume_array, replacement_cost):
    """Multiply retention volumes by the retention replacement cost.

    Args:
        retention_volume_array (numpy.ndarray): 2D array of retention volumes.
            Assumes that the retention volume nodata value is the global
            FLOAT_NODATA.
        replacement_cost (float): Replacement cost per cubic meter of water

    Returns:
        numpy.ndarray of retention values with the same dimensions as the input
    """
    value_array = numpy.full(retention_volume_array.shape, FLOAT_NODATA,
                             dtype=numpy.float32)
    valid_mask = ~pygeoprocessing.array_equals_nodata(
        retention_volume_array, FLOAT_NODATA)

    # retention (m^3/yr) * replacement cost ($/m^3) = retention value ($/yr)
    value_array[valid_mask] = (
        retention_volume_array[valid_mask] * replacement_cost)
    return value_array


def adjust_op(ratio_array, avg_ratio_array, near_connected_lulc_array,
              near_road_array):
    """Apply the retention ratio adjustment algorithm to an array of ratios.

    This is meant to be used with raster_map. Assumes that nodata is already
    filtered out.

    Args:
        ratio_array (numpy.ndarray): 2D array of stormwater retention ratios
        avg_ratio_array (numpy.ndarray): 2D array of averaged ratios
        near_connected_lulc_array (numpy.ndarray): 2D boolean array where 1
            means this pixel is near a directly-connected LULC area
        near_road_array (numpy.ndarray): 2D boolean array where 1
            means this pixel is near a road centerline

    Returns:
        2D numpy array of adjusted retention ratios. Has the same shape as
        ``retention_ratio_array``.
    """
    # adjustment factor:
    # - 0 if any of the nearby pixels are impervious/connected;
    # - average of nearby pixels, otherwise
    # equation 2-4: Radj_ij = R_ij + (1 - R_ij) * C_ij
    return ratio_array + (1 - ratio_array) * (
        avg_ratio_array * ~(
            near_connected_lulc_array | near_road_array
        ).astype(bool))


def aggregate_results(base_aggregate_areas_path, target_vector_path, srs_wkt,
                      aggregations):
    """Aggregate outputs into regions of interest.

    Args:
        base_aggregate_areas_path (str): path to vector of polygon(s) to
            aggregate over. This is the original input.
        target_vector_path (str): path to write out the results. This will be a
            copy of the base vector with added fields, reprojected to the
            target WKT and saved in geopackage format.
        srs_wkt (str): a Well-Known Text representation of the target spatial
            reference. The base vector is reprojected to this spatial reference
            before aggregating the rasters over it.
        aggregations (list[tuple(str,str,str)]): list of tuples describing the
            datasets to aggregate. Each tuple has 3 items. The first is the
            path to a raster to aggregate. The second is the field name for
            this aggregated data in the output vector. The third is either
            'mean' or 'sum' indicating the aggregation to perform.

    Returns:
        None
    """
    pygeoprocessing.reproject_vector(base_aggregate_areas_path, srs_wkt,
                                     target_vector_path, driver_name='GPKG')
    aggregate_vector = gdal.OpenEx(target_vector_path, gdal.GA_Update)
    aggregate_layer = aggregate_vector.GetLayer()

    for raster_path, field_id, aggregation_op in aggregations:
        # aggregate the raster by the vector region(s)
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), target_vector_path)

        # set up the field to hold the aggregate data
        aggregate_field = ogr.FieldDefn(field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)
        aggregate_layer.ResetReading()

        # save the aggregate data to the field for each feature
        for feature in aggregate_layer:
            feature_id = feature.GetFID()
            if aggregation_op == 'mean':
                pixel_count = aggregate_stats[feature_id]['count']
                try:
                    value = (aggregate_stats[feature_id]['sum'] / pixel_count)
                except ZeroDivisionError:
                    LOGGER.warning(
                        f'Polygon {feature_id} does not overlap {raster_path}')
                    value = 0.0
            elif aggregation_op == 'sum':
                value = aggregate_stats[feature_id]['sum']
            feature.SetField(field_id, float(value))
            aggregate_layer.SetFeature(feature)

    # save the aggregate vector layer and clean up references
    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


def is_near(input_path, radius, distance_path, out_path):
    """Make binary raster of which pixels are within a radius of a '1' pixel.

    Args:
        input_path (str): path to a binary raster where '1' pixels are what
            we're measuring distance to, in this case roads/connected areas
        radius (float): distance in pixels which is considered "near".
            pixels this distance or less from a '1' pixel are marked '1' in
            the output. Distances are measured centerpoint to centerpoint.
        distance_path (str): path to write out the raster of distances
        out_path (str): path to write out the final thresholded raster.
            Pixels marked '1' are near to a '1' pixel in the input, pixels
            marked '0' are not.

    Returns:
        None
    """
    # Calculate the distance from each pixel to the nearest '1' pixel
    pygeoprocessing.distance_transform_edt((input_path, 1), distance_path)

    # Threshold that to a binary array so '1' means it's within the radius
    pygeoprocessing.raster_map(
        op=lambda dist: dist <= radius,
        rasters=[distance_path],
        target_path=out_path,
        target_dtype=numpy.uint8,
        target_nodata=UINT8_NODATA)


def raster_average(raster_path, radius, kernel_path, out_path):
    """Average pixel values within a radius.

    Make a search kernel where a pixel has '1' if its centerpoint is within
    the radius of the center pixel's centerpoint.
    For each pixel in a raster, center the search kernel on top of it. Then
    its "neighborhood" includes all the pixels that are below a '1' in the
    search kernel. Add up the neighborhood pixel values and divide by how
    many there are.

    This accounts for edge pixels and nodata pixels. For instance, if the
    kernel covers a 3x3 pixel area centered on each pixel, most pixels will
    have 9 valid pixels in their neighborhood, most edge pixels will have 6,
    and most corner pixels will have 4. Edge and nodata pixels in the
    neighborhood don't count towards the total (denominator in the average).

    Args:
        raster_path (str): path to the raster file to average
        radius (float): distance to average around each pixel's centerpoint in
            raster coordinate system units
        kernel_path (str): path to write out the search kernel raster, an
            intermediate output required by pygeoprocessing.convolve_2d
        out_path (str): path to write out the averaged raster output

    Returns:
        None
    """
    pixel_radius = radius / abs(pygeoprocessing.get_raster_info(
        raster_path)['pixel_size'][0])
    pygeoprocessing.kernels.dichotomous_kernel(
        target_kernel_path=kernel_path,
        max_distance=pixel_radius,
        normalize=False)

    # convolve the signal (input raster) with the kernel and normalize
    # this is equivalent to taking an average of each pixel's neighborhood
    pygeoprocessing.convolve_2d(
        (raster_path, 1),
        (kernel_path, 1),
        out_path,
        # pixels with nodata or off the edge of the raster won't count towards
        # the sum or the number of values to normalize by
        ignore_nodata_and_edges=True,
        # divide by number of valid pixels in the kernel (averaging)
        normalize_kernel=True,
        # output will have nodata where ratio_path has nodata
        mask_nodata=True,
        target_datatype=gdal.GDT_Float32,
        target_nodata=FLOAT_NODATA)


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
    if 'soil_group_path' not in invalid_keys:
        # check that soil group raster has integer type
        soil_group_dtype = pygeoprocessing.get_raster_info(
            args['soil_group_path'])['numpy_type']
        if not numpy.issubdtype(soil_group_dtype, numpy.integer):
            validation_warnings.append(
                (['soil_group_path'], NONINTEGER_SOILS_RASTER_MESSAGE))
    return validation_warnings
