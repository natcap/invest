"""InVEST Carbon Edge Effect Model.

An implementation of the model described in 'Degradation in carbon stocks
near tropical forest edges', by Chaplin-Kramer et. al (2015).
"""
import copy
import logging
import os
import pickle
import time

import numpy
import pandas
import pygeoprocessing
import scipy.spatial
import shapely.errors
import shapely.geometry
import shapely.prepared
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

# grid cells are 100km. Becky says 500km is a good upper bound to search
DISTANCE_UPPER_BOUND = 500e3

# helpful to have a global nodata defined for the whole model
NODATA_VALUE = -1

MODEL_SPEC = spec.ModelSpec(
    model_id="forest_carbon_edge_effect",
    model_title=gettext("Forest Carbon Edge Effect"),
    userguide="carbon_edge.html",
    validate_spatial_overlap=["aoi_vector_path", "lulc_raster_path"],
    different_projections_ok=False,
    aliases=("fc",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_raster_path", "biophysical_table_path", "pools_to_calculate"],
        [
            "compute_forest_edge_effects",
            "tropical_forest_edge_carbon_model_vector_path",
            "n_nearest_model_points",
            "biomass_to_carbon_conversion_factor"
        ],
        ["aoi_vector_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.IntegerInput(
            id="n_nearest_model_points",
            name=gettext("number of points to average"),
            about=(
                "Number of closest regression models that are used when calculating the"
                " total biomass. Each local model is linearly weighted by distance such"
                " that the pixel's biomass is a function of each of these points with the"
                " closest point having the largest effect. Must be greater than 0."
                " Required if Compute Forest Edge Effects is selected."
            ),
            required="compute_forest_edge_effects",
            allowed="compute_forest_edge_effects",
            units=u.none,
            expression="value > 0"
        ),
        spec.AOI.model_copy(update=dict(
            id="aoi_vector_path",
            required=False,
            projected=True
        )),
        spec.CSVInput(
            id="biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "A table mapping each LULC code from the LULC map to biophysical data for"
                " that LULC class."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.BooleanInput(
                    id="is_tropical_forest",
                    about=gettext(
                        "Enter 1 if the LULC class is tropical forest, 0 if it is not"
                        " tropical forest."
                    )
                ),
                spec.NumberInput(
                    id="c_above",
                    about=gettext(
                        "Carbon density value for the aboveground carbon pool."
                    ),
                    units=u.metric_ton / u.hectare
                ),
                spec.NumberInput(
                    id="c_below",
                    about=gettext(
                        "Carbon density value for the belowground carbon pool. Required"
                        " if calculating all pools."
                    ),
                    required="pools_to_calculate == 'all'",
                    units=u.metric_ton / u.hectare
                ),
                spec.NumberInput(
                    id="c_soil",
                    about=gettext(
                        "Carbon density value for the soil carbon pool. Required if"
                        " calculating all pools."
                    ),
                    required="pools_to_calculate == 'all'",
                    units=u.metric_ton / u.hectare
                ),
                spec.NumberInput(
                    id="c_dead",
                    about=gettext(
                        "Carbon density value for the dead matter carbon pool. Required"
                        " if calculating all pools."
                    ),
                    required="pools_to_calculate == 'all'",
                    units=u.metric_ton / u.hectare
                )
            ],
            index_col="lucode"
        ),
        spec.SingleBandRasterInput(
            id="lulc_raster_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes. Each land use/land cover type must be"
                " assigned a unique integer code. All values in this raster must have"
                " corresponding entries in the Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.OptionStringInput(
            id="pools_to_calculate",
            name=gettext("carbon pools to calculate"),
            about=gettext("Which carbon pools to consider."),
            options=[
                spec.Option(
                    key="all",
                    about=(
                        "Use all pools (aboveground, belowground, soil, and dead matter)"
                        " in the carbon pool calculation.")),
                spec.Option(
                    key="above_ground",
                    display_name="aboveground only",
                    about=(
                        "Only use the aboveground pool in the carbon pool calculation."
                    )
                )
            ]
        ),
        spec.BooleanInput(
            id="compute_forest_edge_effects",
            name=gettext("compute forest edge effects"),
            about=gettext("Account for forest edge effects on aboveground carbon.")
        ),
        spec.VectorInput(
            id="tropical_forest_edge_carbon_model_vector_path",
            name=gettext("global regression models"),
            about=gettext(
                "Map storing the optimal regression model for each tropical subregion and"
                " the corresponding theta parameters for that regression equation."
                " Default data is provided. Required if Compute Forest Edge Effects is"
                " selected."
            ),
            required="compute_forest_edge_effects",
            allowed="compute_forest_edge_effects",
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.OptionStringInput(
                    id="method",
                    about=gettext("Optimal regression model for the area."),
                    options=[
                        spec.Option(key="1", about="asymptotic"),
                        spec.Option(key="2", about="logarithmic"),
                        spec.Option(key="3", about="linear"),
                    ]
                ),
                spec.NumberInput(
                    id="theta1",
                    about=gettext("θ₁ parameter for the regression equation."),
                    units=u.none
                ),
                spec.NumberInput(
                    id="theta2",
                    about=gettext("θ₂ parameter for the regression equation."),
                    units=u.none
                ),
                spec.NumberInput(
                    id="theta3",
                    about=gettext(
                        "θ₃ parameter for the regression equation. Used only for the"
                        " asymptotic model."
                    ),
                    units=u.none
                )
            ],
            projected=True,
            projection_units=u.meter
        ),
        spec.RatioInput(
            id="biomass_to_carbon_conversion_factor",
            name=gettext("forest edge biomass to carbon conversion factor"),
            about=gettext(
                "Proportion of forest edge biomass that is elemental carbon. Required if"
                " Compute Forest Edge Effects is selected."
            ),
            required="compute_forest_edge_effects",
            allowed="compute_forest_edge_effects",
            units=None
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="carbon_map",
            path="carbon_map.tif",
            about=gettext(
                "A map of carbon stock per hectare, with the amount in forest derived"
                " from the regression based on distance to forest edge, and the amount in"
                " non-forest classes according to the biophysical table. "
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.VectorOutput(
            id="aggregated_carbon_stocks",
            path="aggregated_carbon_stocks.shp",
            about=gettext("AOI map with aggregated carbon statistics."),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="c_sum",
                    about=gettext("Total carbon in the area."),
                    units=u.metric_ton
                ),
                spec.NumberOutput(
                    id="c_ha_mean",
                    about=gettext("Mean carbon density in the area."),
                    units=u.metric_ton / u.hectare
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="c_above_carbon_stocks",
            path="intermediate_outputs/c_above_carbon_stocks.tif",
            about=gettext(
                "Carbon stored in the aboveground biomass carbon pool."
            ),
            data_type=float,
            units=u.metric_ton
        ),
        spec.SingleBandRasterOutput(
            id="c_below_carbon_stocks",
            path="intermediate_outputs/c_below_carbon_stocks.tif",
            about=gettext(
                "Carbon stored in the belowground biomass carbon pool."
            ),
            data_type=float,
            units=u.metric_ton
        ),
        spec.SingleBandRasterOutput(
            id="c_dead_carbon_stocks",
            path="intermediate_outputs/c_dead_carbon_stocks.tif",
            about=gettext(
                "Carbon stored in the dead matter biomass carbon pool."
            ),
            data_type=float,
            units=u.metric_ton
        ),
        spec.SingleBandRasterOutput(
            id="c_soil_carbon_stocks",
            path="intermediate_outputs/c_soil_carbon_stocks.tif",
            about=gettext("Carbon stored in the soil biomass carbon pool."),
            data_type=float,
            units=u.metric_ton
        ),
        spec.VectorOutput(
            id="local_carbon_shape",
            path="intermediate_outputs/local_carbon_shape.shp",
            about=gettext(
                "The regression parameters reprojected to match your study area."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="edge_distance",
            path="intermediate_outputs/edge_distance.tif",
            about=gettext(
                "The distance of each forest pixel to the nearest forest edge"
            ),
            data_type=float,
            units=u.pixel
        ),
        spec.SingleBandRasterOutput(
            id="tropical_forest_edge_carbon_stocks",
            path="intermediate_outputs/tropical_forest_edge_carbon_stocks.tif",
            about=gettext(
                "A map of carbon in the forest only, according to the regression"
                " method."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.VectorOutput(
            id="regression_model_params_clipped",
            path="intermediate_outputs/regression_model_params_clipped.shp",
            about=gettext(
                "The Global Regression Models shapefile clipped to the study"
                " area."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.SingleBandRasterOutput(
            id="non_forest_mask",
            path="intermediate_outputs/non_forest_mask.tif",
            about=gettext(
                "A mask raster where non-forest pixels are marked as 1 and forest pixels as 0."
            ),
            data_type=int,
            units=None
        ),
        spec.FileOutput(
            id="spatial_index_pickle",
            path="intermediate_outputs/spatial_index.pickle",
            about=gettext(
                "A pickle file containing the spatial index (kd-tree and model parameters) for forest edge regression."
            )
        ),
        spec.TASKGRAPH_CACHE
    ]
)


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
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    # just check that the AOI exists since it wouldn't crash until the end of
    # the whole model run if it didn't.
    if args['aoi_vector_path']:
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

    # Map non-forest landcover codes to carbon biomasses
    LOGGER.info('Calculating direct mapped carbon stocks')
    carbon_maps = []
    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(
        args['biophysical_table_path'])
    pool_list = [('c_above', True)]
    if args['pools_to_calculate'] == 'all':
        pool_list.extend([
            ('c_below', False), ('c_soil', False), ('c_dead', False)])
    for carbon_pool_type, ignore_tropical_type in pool_list:
        if carbon_pool_type in biophysical_df.columns:
            carbon_maps.append(
                file_registry[f'{carbon_pool_type}_carbon_stocks'])
            task_graph.add_task(
                func=_calculate_lulc_carbon_map,
                args=(args['lulc_raster_path'],
                      args['biophysical_table_path'],
                      carbon_pool_type, ignore_tropical_type,
                      args['compute_forest_edge_effects'],
                      file_registry[f'{carbon_pool_type}_carbon_stocks']),
                target_path_list=[
                    file_registry[f'{carbon_pool_type}_carbon_stocks']],
                task_name=f'calculate_lulc_{carbon_pool_type}_map')

    if args['compute_forest_edge_effects']:
        # generate a map of pixel distance to forest edge from the landcover
        # map
        LOGGER.info('Calculating distance from forest edge')
        map_distance_task = task_graph.add_task(
            func=_map_distance_from_tropical_forest_edge,
            args=(args['lulc_raster_path'],
                  args['biophysical_table_path'],
                  file_registry['edge_distance'],
                  file_registry['non_forest_mask']),
            target_path_list=[file_registry['edge_distance'],
                              file_registry['non_forest_mask']],
            task_name='map_distance_from_forest_edge')

        # Clip global regression model vector to LULC raster bounding box
        LOGGER.info('Clipping global forest carbon edge regression models vector')
        clip_forest_edge_carbon_vector_task = task_graph.add_task(
            func=_clip_global_regression_models_vector,
            args=(args['lulc_raster_path'],
                  args['tropical_forest_edge_carbon_model_vector_path'],
                  file_registry['regression_model_params_clipped']),
            target_path_list=[file_registry['regression_model_params_clipped']],
            task_name='clip_forest_edge_carbon_vector')

        # Build spatial index for gridded global model for closest 3 points
        LOGGER.info('Building spatial index for forest edge models.')
        build_spatial_index_task = task_graph.add_task(
            func=_build_spatial_index,
            args=(args['lulc_raster_path'],
                  file_registry['local_carbon_shape'],
                  file_registry['regression_model_params_clipped'],
                  file_registry['spatial_index_pickle']),
            target_path_list=[file_registry['spatial_index_pickle']],
            task_name='build_spatial_index',
            dependent_task_list=[clip_forest_edge_carbon_vector_task])

        # calculate the carbon edge effect on forests
        LOGGER.info('Calculating forest edge carbon')
        task_graph.add_task(
            func=_calculate_tropical_forest_edge_carbon_map,
            args=(file_registry['edge_distance'],
                  file_registry['spatial_index_pickle'],
                  args['n_nearest_model_points'],
                  args['biomass_to_carbon_conversion_factor'],
                  file_registry['tropical_forest_edge_carbon_stocks']),
            target_path_list=[
                file_registry['tropical_forest_edge_carbon_stocks']],
            task_name='calculate_forest_edge_carbon_map',
            dependent_task_list=[map_distance_task, build_spatial_index_task])

        # This is also a carbon stock
        carbon_maps.append(
            file_registry['tropical_forest_edge_carbon_stocks'])

    # combine maps into a single output
    LOGGER.info('combining carbon maps into single raster')
    carbon_maps_band_list = [(path, 1) for path in carbon_maps]

    # Join here since the raster calculation depends on the target datasets
    # from all the tasks above
    task_graph.join()

    combine_carbon_maps_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(carbon_maps_band_list, combine_carbon_maps,
              file_registry['carbon_map'], gdal.GDT_Float32,
              NODATA_VALUE),
        target_path_list=[file_registry['carbon_map']],
        task_name='combine_carbon_maps')

    # generate report (optional) by aoi if they exist
    if args['aoi_vector_path']:
        LOGGER.info('aggregating carbon map by aoi')
        task_graph.add_task(
            func=_aggregate_carbon_map,
            args=(args['aoi_vector_path'],
                  file_registry['carbon_map'],
                  file_registry['aggregated_carbon_stocks']),
            target_path_list=[
                file_registry['aggregated_carbon_stocks']],
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
        valid_mask = ~pygeoprocessing.array_equals_nodata(carbon_map, NODATA_VALUE)
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

    # Since pixel values are Mg/ha, raster sum is (Mg•px)/ha.
    # To convert to Mg, multiply by ha/px.
    raster_info = pygeoprocessing.get_raster_info(carbon_map_path)
    pixel_area = abs(numpy.prod(raster_info['pixel_size']))
    ha_per_px = pixel_area / 10000

    for poly_feat in target_aggregate_layer:
        poly_fid = poly_feat.GetFID()
        poly_feat.SetField(
            'c_sum', float(serviceshed_stats[poly_fid]['sum'] * ha_per_px))
        # calculates mean pixel value per ha in for each feature in AOI
        poly_geom = poly_feat.GetGeometryRef()
        poly_area_ha = poly_geom.GetArea() / 1e4  # converts m^2 to hectare
        poly_geom = None
        poly_feat.SetField(
            'c_ha_mean',
            float(serviceshed_stats[poly_fid]['sum'] / poly_area_ha
                  * ha_per_px))

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
    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(biophysical_table_path)

    lucode_to_per_cell_carbon = {}

    # Build a lookup table
    for lucode, row in biophysical_df.iterrows():
        if compute_forest_edge_effects:
            is_tropical_forest = row['is_tropical_forest']
        else:
            is_tropical_forest = False
        if ignore_tropical_type and is_tropical_forest:
            # if tropical forest above ground, lookup table is nodata
            lucode_to_per_cell_carbon[lucode] = NODATA_VALUE
        else:
            if pandas.isna(row[carbon_pool_type]):
                raise ValueError(
                    "Could not interpret carbon pool value as a number. "
                    f"lucode: {lucode}, pool_type: {carbon_pool_type}, "
                    f"value: {row[carbon_pool_type]}")
            lucode_to_per_cell_carbon[lucode] = row[carbon_pool_type]

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
    biophysical_df = MODEL_SPEC.get_input(
        'biophysical_table_path').get_validated_dataframe(
        biophysical_table_path)
    forest_codes = biophysical_df[biophysical_df['is_tropical_forest']].index.values

    # Make a raster where 1 is non-forest landcover types and 0 is forest
    lulc_nodata = pygeoprocessing.get_raster_info(
        base_lulc_raster_path)['nodata']

    pygeoprocessing.raster_map(
        op=lambda lulc_array: ~numpy.isin(lulc_array, forest_codes),
        rasters=[base_lulc_raster_path],
        target_path=target_non_forest_mask_path,
        target_dtype=numpy.uint8,
        target_nodata=255)

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

    for offset_dict in pygeoprocessing.iterblocks((base_lulc_raster_path, 1),
            offset_only=True):
        # where LULC has nodata, overwrite edge distance with nodata value
        lulc_block = lulc_band.ReadAsArray(**offset_dict)
        distance_block = edge_distance_band.ReadAsArray(**offset_dict)
        nodata_mask = pygeoprocessing.array_equals_nodata(lulc_block, lulc_nodata)
        distance_block[nodata_mask] = lulc_nodata
        edge_distance_band.WriteArray(
            distance_block,
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])


def _clip_global_regression_models_vector(
        lulc_raster_path, source_vector_path, target_vector_path):
    """Clip the global carbon edge model shapefile

    Clip the shapefile containing the global carbon edge model parameters
    to the bounding box of the LULC raster (representing the target study area)
    plus a buffer to account for the kd-tree lookup DISTANCE_UPPER_BOUND

    Args:
        lulc_raster_path (string): path to a raster that is used to define the
            bounding box to use for clipping.
        source_vector_path (string): a path to an OGR shapefile to be clipped.
        target_vector_path (string): a path to an OGR shapefile to store the
            clipped vector.

    Returns:
        None

    """
    raster_info = pygeoprocessing.get_raster_info(lulc_raster_path)
    vector_info = pygeoprocessing.get_vector_info(source_vector_path)

    bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = raster_info['bounding_box']
    buffered_bb = [
        bbox_minx - DISTANCE_UPPER_BOUND,
        bbox_miny - DISTANCE_UPPER_BOUND,
        bbox_maxx + DISTANCE_UPPER_BOUND,
        bbox_maxy + DISTANCE_UPPER_BOUND
    ]

    # Reproject the LULC bounding box to the vector's projection for clipping
    mask_bb = pygeoprocessing.transform_bounding_box(buffered_bb,
        raster_info['projection_wkt'], vector_info['projection_wkt'])
    shapely_mask = shapely.prepared.prep(shapely.geometry.box(*mask_bb))

    base_vector = gdal.OpenEx(source_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()
    base_layer_defn = base_layer.GetLayerDefn()
    base_geom_type = base_layer.GetGeomType()

    target_driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = target_driver.Create(
        target_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    target_layer = target_vector.CreateLayer(
        base_layer_defn.GetName(), base_layer.GetSpatialRef(), base_geom_type)
    target_layer.CreateFields(base_layer.schema)

    target_layer.StartTransaction()
    invalid_feature_count = 0
    for feature in base_layer:
        invalid = False
        geometry = feature.GetGeometryRef()
        try:
            shapely_geom = shapely.wkb.loads(bytes(geometry.ExportToWkb()))
        # Catch invalid geometries that cannot be loaded by Shapely;
        # e.g. polygons with too few points for their type
        except shapely.errors.ShapelyError:
            invalid = True
        else:
            if shapely_geom.is_valid:
                # Check for intersection rather than use gdal.Layer.Clip()
                # to preserve the shape of the polygons (we use the centroid
                # when constructing the kd-tree)
                if shapely_mask.intersects(shapely_geom):
                    new_feature = ogr.Feature(target_layer.GetLayerDefn())
                    new_feature.SetGeometry(ogr.CreateGeometryFromWkb(
                        shapely_geom.wkb))
                    for field_name, field_value in feature.items().items():
                        new_feature.SetField(field_name, field_value)
                    target_layer.CreateFeature(new_feature)
            else:
                invalid = True
        finally:
            if invalid:
                invalid_feature_count += 1
                LOGGER.warning(
                    f"The geometry at feature {feature.GetFID()} is invalid "
                    "and will be skipped.")

    target_layer.CommitTransaction()

    if invalid_feature_count:
        LOGGER.warning(
            f"{invalid_feature_count} features in {source_vector_path} "
            "were found to be invalid during clipping and were skipped.")


def _build_spatial_index(
        base_raster_path, carbon_model_reproject_path,
        tropical_forest_edge_carbon_model_vector_path,
        target_spatial_index_pickle_path):
    """Build a kd-tree index.

    Build a kd-tree index of the locally projected globally georeferenced
    carbon edge model parameters.

    Args:
        base_raster_path (string): path to a raster that is used to define the
            bounding box and projection of the local model.
        carbon_model_reproject_path (string): path at which to create the
            shapefile of the locally projected global data model grid.
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
    with open(spatial_index_pickle_path, 'rb') as spatial_index_pickle_file:
        kd_tree, theta_model_parameters, method_model_parameter = pickle.load(
            spatial_index_pickle_file)

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
        )

        # logarithmic model
        # biomass_2 = t1 + t2 * numpy.log(edge_dist_km)
        biomass[mask_2] = (
            thetas[mask_2][:, 0] + thetas[mask_2][:, 1] * numpy.log(
                valid_edge_distances_km[mask_2]))

        # linear regression
        # biomass_3 = t1 + t2 * edge_dist_km
        biomass[mask_3] = (
            thetas[mask_3][:, 0] + thetas[mask_3][:, 1] *
            valid_edge_distances_km[mask_3])

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
    model_spec = copy.deepcopy(MODEL_SPEC)
    if 'pools_to_calculate' in args and args['pools_to_calculate'] == 'all':
        model_spec.get_input('biophysical_table_path').get_column('c_below').required = True
        model_spec.get_input('biophysical_table_path').get_column('c_soil').required = True
        model_spec.get_input('biophysical_table_path').get_column('c_dead').required = True
    return validation.validate(args, model_spec)
