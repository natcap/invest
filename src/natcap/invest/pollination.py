"""Pollinator service model for InVEST."""
import collections
import itertools
import logging
import os
import re

import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import taskgraph
from osgeo import gdal
from osgeo import ogr

from . import gettext
from . import spec
from . import utils
from . import validation
from .file_registry import FileRegistry
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

MODEL_SPEC = spec.ModelSpec(
    model_id="pollination",
    model_title=gettext("Crop Pollination"),
    userguide="croppollination.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["landcover_raster_path", "landcover_biophysical_table_path"],
        ["guild_table_path", "farm_vector_path"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="landcover_raster_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "Map of LULC codes. All values in this raster must have corresponding"
                " entries in the Biophysical Table."
            ),
            data_type=int,
            units=None,
            projected=True
        ),
        spec.CSVInput(
            id="guild_table_path",
            name=gettext("Guild Table"),
            about=gettext(
                "A table mapping each pollinator species or guild of interest to its"
                " pollination-related parameters."
            ),
            columns=[
                spec.StringInput(
                    id="species",
                    about=gettext(
                        "Unique name or identifier for each pollinator species or guild"
                        " of interest."
                    ),
                    regexp=None
                ),
                spec.RatioInput(
                    id="nesting_suitability_[SUBSTRATE]_index",
                    about=gettext(
                        "Utilization of the substrate by this species, where 1 indicates"
                        " the nesting substrate is fully utilized and 0 indicates it is"
                        " not utilized at all. Replace [SUBSTRATE] with substrate names"
                        " matching those in the Biophysical Table, so that there is a"
                        " column for each substrate."
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="foraging_activity_[SEASON]_index",
                    about=gettext(
                        "Pollinator activity for this species/guild in each season. 1"
                        " indicates maximum activity for the species/guild, and 0"
                        " indicates no activity. Replace [SEASON] with season names"
                        " matching those in the biophysical table, so that there is a"
                        " column for each season."
                    ),
                    units=None
                ),
                spec.NumberInput(
                    id="alpha",
                    about=gettext(
                        "Average distance that this species or guild travels to forage on"
                        " flowers."
                    ),
                    units=u.meter
                ),
                spec.RatioInput(
                    id="relative_abundance",
                    about=gettext(
                        "The proportion of total pollinator abundance that consists of"
                        " this species/guild."
                    ),
                    units=None
                )
            ],
            index_col="species"
        ),
        spec.CSVInput(
            id="landcover_biophysical_table_path",
            name=gettext("biophysical table"),
            about=gettext(
                "A table mapping each LULC class to nesting availability and floral"
                " abundance data for each substrate and season in that LULC class. All"
                " values in the LULC raster must have corresponding entries in this"
                " table."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.RatioInput(
                    id="nesting_[SUBSTRATE]_availability_index",
                    about=gettext(
                        "Index of availability of the given substrate in this LULC class."
                        " Replace [SUBSTRATE] with substrate names matching those in the"
                        " Guild Table, so that there is a column for each substrate."
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="floral_resources_[SEASON]_index",
                    about=gettext(
                        "Abundance of flowers during the given season in this LULC class."
                        " This is the proportion of land area covered by flowers,"
                        " multiplied by the proportion of the season for which there is"
                        " that coverage. Replace [SEASON] with season names matching"
                        " those in the Guild Table, so that there is a column for each"
                        " season."
                    ),
                    units=None
                )
            ],
            index_col="lucode"
        ),
        spec.VectorInput(
            id="farm_vector_path",
            name=gettext("farms map"),
            about=gettext(
                "Map of farm sites to be analyzed, with pollination data specific to each"
                " farm."
            ),
            required=False,
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.StringInput(
                    id="crop_type",
                    about=(
                        "Name of the crop grown on each polygon, e.g. 'blueberries',"
                        " 'almonds', etc."
                    ),
                    regexp=None
                ),
                spec.RatioInput(
                    id="half_sat",
                    about=gettext(
                        "The half saturation coefficient for the crop grown in this area."
                        " This is the wild pollinator abundance (i.e. the proportion of"
                        " all pollinators that are wild) needed to reach half of the"
                        " total potential pollinator-dependent yield."
                    ),
                    units=None
                ),
                spec.StringInput(
                    id="season",
                    about=gettext(
                        "The season in which the crop is pollinated. Season names must"
                        " match those in the Guild Table and Biophysical Table."
                    ),
                    regexp=None
                ),
                spec.RatioInput(
                    id="fr_[SEASON]",
                    about=gettext(
                        "The floral resources available at this farm for the given"
                        " season. Replace [SEASON] with season names matching those in"
                        " the Guild Table and Biophysical Table, so that there is one"
                        " field for each season."
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="n_[SUBSTRATE]",
                    about=gettext(
                        "The nesting suitability for the given substrate at this farm."
                        " given substrate. Replace [SUBSTRATE] with substrate names"
                        " matching those in the Guild Table and Biophysical Table, so"
                        " that there is one field for each substrate."
                    ),
                    units=None
                ),
                spec.RatioInput(
                    id="p_dep",
                    about=gettext("The proportion of crop dependent on pollinators."),
                    units=None
                ),
                spec.RatioInput(
                    id="p_managed",
                    about=gettext(
                        "The proportion of pollination required on the farm that is"
                        " provided by managed pollinators."
                    ),
                    units=None
                )
            ],
            projected=None
        )
    ],
    outputs=[
        spec.VectorOutput(
            id="farm_results",
            path="farm_results.shp",
            about=gettext(
                "A copy of the input farm polygon vector file with additional fields"
            ),
            created_if="farm_vector_path",
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.RatioOutput(
                    id="p_abund",
                    about=gettext(
                        "Average pollinator abundance on the farm for the active season"
                    )
                ),
                spec.RatioOutput(
                    id="y_tot",
                    about=gettext(
                        "Total yield index, including wild and managed pollinators and"
                        " pollinator independent yield."
                    )
                ),
                spec.RatioOutput(
                    id="pdep_y_w",
                    about=gettext(
                        "Proportion of potential pollination-dependent yield attributable"
                        " to wild pollinators."
                    )
                ),
                spec.RatioOutput(
                    id="y_wild",
                    about=gettext(
                        "Proportion of the total yield attributable to wild pollinators."
                    )
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="farm_pollinators",
            path="farm_pollinators.tif",
            about=gettext(
                "Total pollinator abundance across all species per season, clipped to the"
                " geometry of the farm vector’s polygons."
            ),
            created_if="farm_vector_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="pollinator_abundance_[SPECIES]_[SEASON]",
            path="pollinator_abundance_[SPECIES]_[SEASON].tif",
            about=gettext("Abundance of pollinator SPECIES in season SEASON."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="pollinator_supply_[SPECIES]",
            path="pollinator_supply_[SPECIES].tif",
            about=gettext(
                "Index of pollinator SPECIES that could be on a pixel given its arbitrary"
                " abundance factor from the table, multiplied by the habitat suitability"
                " for that species at that pixel, multiplied by the available floral"
                " resources that a pollinator could fly to from that pixel."
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="total_pollinator_abundance_[SEASON]",
            path="total_pollinator_abundance_[SEASON].tif",
            about=gettext("Total pollinator abundance across all species per season."),
            created_if="farm_vector_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="total_pollinator_yield",
            path="total_pollinator_yield.tif",
            about=gettext(
                "Total pollinator yield index for pixels that overlap farms, including"
                " wild and managed pollinators."
            ),
            created_if="farm_vector_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="wild_pollinator_yield",
            path="wild_pollinator_yield.tif",
            about=gettext(
                "Pollinator yield index for pixels that overlap farms, for wild"
                " pollinators only."
            ),
            created_if="farm_vector_path",
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="blank_raster",
            path="intermediate_outputs/blank_raster.tif",
            about=gettext(
                "Blank raster used for rasterizing all the farm parameters/fields"
                " later"
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="convolve_ps_[SPECIES]",
            path="intermediate_outputs/convolve_ps_[SPECIES].tif",
            about=gettext("Convolved pollinator supply"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="farm_nesting_substrate_index_[SUBSTRATE]",
            path="intermediate_outputs/farm_nesting_substrate_index_[SUBSTRATE].tif",
            about=gettext("Rasterized substrate availability"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="farm_pollinator_[SEASON]",
            path="intermediate_outputs/farm_pollinator_[SEASON].tif",
            about=gettext("On-farm pollinator abundance"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="farm_relative_floral_abundance_index_[SEASON]",
            path="intermediate_outputs/farm_relative_floral_abundance_index_[SEASON].tif",
            about=gettext("On-farm relative floral abundance"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="floral_resources_[SPECIES]",
            path="intermediate_outputs/floral_resources_[SPECIES].tif",
            about=gettext("Floral resources available to the species"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="foraged_flowers_index_[SPECIES]_[SEASON]",
            path="intermediate_outputs/foraged_flowers_index_[SPECIES]_[SEASON].tif",
            about=gettext(
                "Foraged flowers index for the given species and season"
            ),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="habitat_nesting_index_[SPECIES]",
            path="intermediate_outputs/habitat_nesting_index_[SPECIES].tif",
            about=gettext("Habitat nesting index for the given species"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="half_saturation_[SEASON]",
            path="intermediate_outputs/half_saturation_[SEASON].tif",
            about=gettext("Half saturation constant for the given season"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="kernel_[ALPHA]",
            path="intermediate_outputs/kernel_[ALPHA].tif",
            about=gettext("Exponential decay kernel for the given radius"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="local_foraging_effectiveness_[SPECIES]",
            path="intermediate_outputs/local_foraging_effectiveness_[SPECIES].tif",
            about=gettext("Foraging effectiveness for the given species"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="managed_pollinators",
            path="intermediate_outputs/managed_pollinators.tif",
            about=gettext("Managed pollinators rasterized from the farm vector"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="nesting_substrate_index_[SUBSTRATE]",
            path="intermediate_outputs/nesting_substrate_index_[SUBSTRATE].tif",
            about=gettext("Nesting substrate index for the given substrate"),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="relative_floral_abundance_index_[SEASON]",
            path="intermediate_outputs/relative_floral_abundance_index_[SEASON].tif",
            about=gettext("Floral abundance index in the given season"),
            data_type=float,
            units=None
        ),
        spec.VectorOutput(
            id="reprojected_farm_vector",
            path="intermediate_outputs/reprojected_farm_vector.shp",
            about=gettext("Farm vector reprojected to the LULC projection"),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[]
        ),
        spec.TASKGRAPH_CACHE
    ]
)

_INDEX_NODATA = -1

# These patterns are expected in the biophysical table
_NESTING_SUBSTRATE_PATTERN = 'nesting_([^_]+)_availability_index'
_FLORAL_RESOURCES_AVAILABLE_PATTERN = 'floral_resources_([^_]+)_index'
_EXPECTED_BIOPHYSICAL_HEADERS = [
    _NESTING_SUBSTRATE_PATTERN, _FLORAL_RESOURCES_AVAILABLE_PATTERN]

# These are patterns expected in the guilds table
_NESTING_SUITABILITY_PATTERN = 'nesting_suitability_([^_]+)_index'
# replace with season
_FORAGING_ACTIVITY_PATTERN = 'foraging_activity_%s_index'
_FORAGING_ACTIVITY_RE_PATTERN = _FORAGING_ACTIVITY_PATTERN % '([^_]+)'
_RELATIVE_SPECIES_ABUNDANCE_FIELD = 'relative_abundance'
_ALPHA_HEADER = 'alpha'
_EXPECTED_GUILD_HEADERS = [
    _NESTING_SUITABILITY_PATTERN, _FORAGING_ACTIVITY_RE_PATTERN,
    _ALPHA_HEADER, _RELATIVE_SPECIES_ABUNDANCE_FIELD]

# output field on target shapefile if farms are enabled
_TOTAL_FARM_YIELD_FIELD_ID = 'y_tot'
# output field for wild pollinators on farms if farms are enabled
_WILD_POLLINATOR_FARM_YIELD_FIELD_ID = 'y_wild'
# output field for proportion of wild pollinators over the pollinator
# dependent part of the yield
_POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID = 'pdep_y_w'
# output field for pollinator abundance on farm for the season of pollination
_POLLINATOR_ABUNDANCE_FARM_FIELD_ID = 'p_abund'
# expected pattern for seasonal floral resources in input shapefile (season)
_FARM_FLORAL_RESOURCES_HEADER_PATTERN = 'fr_%s'
# regular expression version of _FARM_FLORAL_RESOURCES_PATTERN
_FARM_FLORAL_RESOURCES_PATTERN = (
    _FARM_FLORAL_RESOURCES_HEADER_PATTERN % '([^_]+)')
# expected pattern for nesting substrate in input shapfile (substrate)
_FARM_NESTING_SUBSTRATE_HEADER_PATTERN = 'n_%s'
# regular expression version of _FARM_NESTING_SUBSTRATE_HEADER_PATTERN
_FARM_NESTING_SUBSTRATE_RE_PATTERN = (
    _FARM_NESTING_SUBSTRATE_HEADER_PATTERN % '([^_]+)')
_HALF_SATURATION_FARM_HEADER = 'half_sat'
_CROP_POLLINATOR_DEPENDENCE_FIELD = 'p_dep'
_MANAGED_POLLINATORS_FIELD = 'p_managed'
_FARM_SEASON_FIELD = 'season'
_EXPECTED_FARM_HEADERS = [
    _FARM_SEASON_FIELD, 'crop_type', _HALF_SATURATION_FARM_HEADER,
    _MANAGED_POLLINATORS_FIELD, _FARM_FLORAL_RESOURCES_PATTERN,
    _FARM_NESTING_SUBSTRATE_RE_PATTERN, _CROP_POLLINATOR_DEPENDENCE_FIELD]


def execute(args):
    """Pollination.

    Args:
        args['workspace_dir'] (string): a path to the output workspace folder.
            Will overwrite any files that exist if the path already exists.
        args['results_suffix'] (string): string appended to each output
            file path.
        args['landcover_raster_path'] (string): file path to a landcover
            raster.
        args['guild_table_path'] (string): file path to a table indicating
            the bee species to analyze in this model run.  Table headers
            must include:

                * 'species': a bee species whose column string names will
                    be referred to in other tables and the model will output
                    analyses per species.
                * one or more columns matching _NESTING_SUITABILITY_PATTERN
                    with values in the range [0, 1] indicating the
                    suitability of the given species to nest in a particular
                    substrate.
                * one or more columns matching _FORAGING_ACTIVITY_RE_PATTERN
                    with values in the range [0, 1] indicating the
                    relative level of foraging activity for that species
                    during a particular season.
                * _ALPHA_HEADER the sigma average flight distance of that bee
                    species in meters.
                * 'relative_abundance': a weight indicating the relative
                    abundance of the particular species with respect to the
                    sum of all relative abundance weights in the table.

        args['landcover_biophysical_table_path'] (string): path to a table
            mapping landcover codes in `args['landcover_path']` to indexes of
            nesting availability for each nesting substrate referenced in
            guilds table as well as indexes of abundance of floral resources
            on that landcover type per season in the bee activity columns of
            the guild table.

            All indexes are in the range [0, 1].

            Columns in the table must be at least
                * 'lucode': representing all the unique landcover codes in
                    the raster ast `args['landcover_path']`
                * For every nesting matching _NESTING_SUITABILITY_PATTERN
                  in the guild stable, a column matching the pattern in
                  `_LANDCOVER_NESTING_INDEX_HEADER`.
                * For every season matching _FORAGING_ACTIVITY_RE_PATTERN
                  in the guilds table, a column matching
                  the pattern in `_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`.
        args['farm_vector_path'] (string): (optional) path to a single layer
            polygon shapefile representing farms. If present will trigger the
            farm yield component of the model.

            The layer must have at least the following fields:

            * season (string): season in which the farm needs pollination
            * crop_type (string): a text field to identify the crop type for
                summary statistics.
            * half_sat (float): a real in the range [0, 1] representing
                the proportion of wild pollinators to achieve a 50% yield
                of that crop.
            * p_dep (float): a number in the range [0, 1]
                representing the proportion of yield dependent on pollinators.
            * p_managed (float): proportion of pollinators that come from
                non-native/managed hives.
            * fr_[season] (float): one or more fields that match this pattern
                such that `season` also matches the season headers in the
                biophysical and guild table.  Any areas that overlap the
                landcover map will replace seasonal floral resources with
                this value.  Ranges from 0..1.
            * n_[substrate] (float): One or more fields that match this
                pattern such that `substrate` also matches the nesting
                substrate headers in the biophysical and guild table.  Any
                areas that overlap the landcover map will replace nesting
                substrate suitability with this value.  Ranges from 0..1.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    # parse out the scenario variables from a complicated set of two tables
    # and possibly a farm polygon.  This function will also raise an exception
    # if any of the inputs are malformed.
    scenario_variables = _parse_scenario_variables(args)
    landcover_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])

    if args['farm_vector_path']:
        # ensure farm vector is in the same projection as the landcover map
        reproject_farm_task = task_graph.add_task(
            task_name='reproject_farm_task',
            func=pygeoprocessing.reproject_vector,
            args=(
                args['farm_vector_path'],
                landcover_raster_info['projection_wkt'],
                file_registry['reprojected_farm_vector']),
            target_path_list=[file_registry['reprojected_farm_vector']])

    # calculate nesting_substrate_index[substrate] substrate maps
    # N(x, n) = ln(l(x), n)
    scenario_variables['nesting_substrate_index_path'] = {}
    landcover_substrate_index_tasks = {}
    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}
    for substrate in scenario_variables['substrate_list']:
        scenario_variables['nesting_substrate_index_path'][substrate] = file_registry[
            'nesting_substrate_index_[SUBSTRATE]', substrate]
        landcover_substrate_index_tasks[substrate] = task_graph.add_task(
            task_name=f'reclassify_to_substrate_{substrate}',
            func=utils.reclassify_raster,
            args=(
                (args['landcover_raster_path'], 1),
                scenario_variables['landcover_substrate_index'][substrate],
                file_registry['nesting_substrate_index_[SUBSTRATE]', substrate],
                gdal.GDT_Float32, _INDEX_NODATA, reclass_error_details),
            target_path_list=[file_registry[
                'nesting_substrate_index_[SUBSTRATE]', substrate]])

    # calculate farm_nesting_substrate_index[substrate] substrate maps
    # dependent on farm substrate rasterized over N(x, n)
    if args['farm_vector_path']:
        scenario_variables['farm_nesting_substrate_index_path'] = (
            collections.defaultdict(dict))
        farm_substrate_rasterize_task_list = []
        for substrate in scenario_variables['substrate_list']:
            farm_substrate_id = _FARM_NESTING_SUBSTRATE_HEADER_PATTERN % substrate
            scenario_variables['farm_nesting_substrate_index_path'][substrate] = file_registry[
                'farm_nesting_substrate_index_[SUBSTRATE]', substrate]
            farm_substrate_rasterize_task_list.append(
                task_graph.add_task(
                    task_name=f'rasterize_nesting_substrate_{substrate}',
                    func=_rasterize_vector_onto_base,
                    args=(
                        file_registry['nesting_substrate_index_[SUBSTRATE]', substrate],
                        file_registry['reprojected_farm_vector'], farm_substrate_id,
                        file_registry['farm_nesting_substrate_index_[SUBSTRATE]', substrate]),
                    target_path_list=[
                        file_registry['farm_nesting_substrate_index_[SUBSTRATE]', substrate]],
                    dependent_task_list=[
                        landcover_substrate_index_tasks[substrate],
                        reproject_farm_task]))

    habitat_nesting_tasks = {}
    for species in scenario_variables['species_list']:
        # calculate habitat_nesting_index[species] HN(x, s) = max_n(N(x, n) ns(s,n))
        if args['farm_vector_path']:
            dependent_task_list = farm_substrate_rasterize_task_list
            substrate_path_map = scenario_variables['farm_nesting_substrate_index_path']
        else:
            dependent_task_list = list(landcover_substrate_index_tasks.values())
            substrate_path_map = scenario_variables['nesting_substrate_index_path']

        habitat_nesting_tasks[species] = task_graph.add_task(
            task_name=f'calculate_habitat_nesting_{species}',
            func=_calculate_habitat_nesting_index,
            args=(
                substrate_path_map,
                scenario_variables['species_substrate_index'][species],
                file_registry['habitat_nesting_index_[SPECIES]', species]),
            dependent_task_list=dependent_task_list,
            target_path_list=[
                file_registry['habitat_nesting_index_[SPECIES]', species]])

    relative_floral_abudance_task_map = {}
    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Biophysical'}
    for season in scenario_variables['season_list']:
        # calculate relative_floral_abundance_index[season] per season
        # RA(l(x), j)
        relative_floral_abudance_task = task_graph.add_task(
            task_name=f'reclassify_to_floral_abundance_{season}',
            func=utils.reclassify_raster,
            args=(
                (args['landcover_raster_path'], 1),
                scenario_variables['landcover_floral_resources'][season],
                file_registry['relative_floral_abundance_index_[SEASON]', season],
                gdal.GDT_Float32, _INDEX_NODATA, reclass_error_details),
            target_path_list=[
                file_registry['relative_floral_abundance_index_[SEASON]', season]])

        # if there's a farm, rasterize floral resources over the top
        if args['farm_vector_path']:
            farm_floral_resources_id = _FARM_FLORAL_RESOURCES_HEADER_PATTERN % season
            # override the relative floral task because we'll need this one
            relative_floral_abudance_task = task_graph.add_task(
                task_name=f'relative_floral_abudance_task_{season}',
                func=_rasterize_vector_onto_base,
                args=(
                    file_registry['relative_floral_abundance_index_[SEASON]', season],
                    file_registry['reprojected_farm_vector'], farm_floral_resources_id,
                    file_registry['farm_relative_floral_abundance_index_[SEASON]', season]),
                target_path_list=[
                    file_registry['farm_relative_floral_abundance_index_[SEASON]', season]],
                dependent_task_list=[
                    relative_floral_abudance_task, reproject_farm_task])

        relative_floral_abudance_task_map[season] = relative_floral_abudance_task

    foraged_flowers_index_task_map = {}
    for species in scenario_variables['species_list']:
        for season in scenario_variables['season_list']:
            if args['farm_vector_path']:
                relative_abundance_path = file_registry[
                    'farm_relative_floral_abundance_index_[SEASON]', season]
            else:
                relative_abundance_path = file_registry[
                    'relative_floral_abundance_index_[SEASON]', season]

            # calculate foraged_flowers_species_season = RA(l(x),j)*fa(s,j)
            foraged_flowers_index_task_map[(species, season)] = task_graph.add_task(
                task_name=f'calculate_foraged_flowers_{species}_{season}',
                func=_multiply_by_scalar,
                args=(
                    relative_abundance_path,
                    scenario_variables['species_foraging_activity'][(species, season)],
                    file_registry['foraged_flowers_index_[SPECIES]_[SEASON]', species, season]),
                dependent_task_list=[
                    relative_floral_abudance_task_map[season]],
                target_path_list=[
                    file_registry['foraged_flowers_index_[SPECIES]_[SEASON]', species, season]])

    pollinator_abundance_task_map = {}
    floral_resources_index_task_map = {}
    alpha_kernel_map = {}
    for species in scenario_variables['species_list']:
        # calculate foraging_effectiveness[species]
        # FE(x, s) = sum_j [RA(l(x), j) * fa(s, j)]
        foraged_flowers_path_band_list = [
            (file_registry['foraged_flowers_index_[SPECIES]_[SEASON]', species, season], 1)
            for season in scenario_variables['season_list']]
        local_foraging_effectiveness_task = task_graph.add_task(
            task_name=f'local_foraging_effectiveness_{species}',
            func=pygeoprocessing.raster_calculator,
            args=(
                foraged_flowers_path_band_list, _sum_arrays,
                file_registry['local_foraging_effectiveness_[SPECIES]', species],
                gdal.GDT_Float32, _INDEX_NODATA),
            target_path_list=[
                file_registry['local_foraging_effectiveness_[SPECIES]', species]],
            dependent_task_list=[
                foraged_flowers_index_task_map[(species, season)]
                for season in scenario_variables['season_list']])

        landcover_pixel_size_tuple = landcover_raster_info['pixel_size']
        try:
            landcover_mean_pixel_size = utils.mean_pixel_size_and_area(
                landcover_pixel_size_tuple)[0]
        except ValueError:
            landcover_mean_pixel_size = numpy.min(numpy.absolute(
                landcover_pixel_size_tuple))
            LOGGER.debug(
                'Land Cover Raster has unequal x, y pixel sizes: '
                f'{landcover_pixel_size_tuple}. Using'
                f'{landcover_mean_pixel_size} as the mean pixel size.')
        # create a convolution kernel for the species flight range
        alpha = scenario_variables['alpha_value'][species] / landcover_mean_pixel_size
        kernel_path = file_registry['kernel_[ALPHA]', f'{alpha:.6f}']
        # to avoid creating duplicate kernel rasters check to see if an
        # adequate kernel task has already been submitted
        try:
            alpha_kernel_raster_task = alpha_kernel_map[kernel_path]
        except KeyError:
            alpha_kernel_raster_task = task_graph.add_task(
                task_name=f'decay_kernel_raster_{alpha}',
                func=pygeoprocessing.kernels.exponential_decay_kernel,
                kwargs=dict(
                    target_kernel_path=kernel_path,
                    max_distance=alpha * 5,
                    expected_distance=alpha),
                target_path_list=[kernel_path])
            alpha_kernel_map[kernel_path] = alpha_kernel_raster_task

        # convolve FE with alpha_s
        floral_resources_task = task_graph.add_task(
            task_name=f'convolve_{species}',
            func=pygeoprocessing.convolve_2d,
            args=(
                (file_registry['local_foraging_effectiveness_[SPECIES]', species], 1),
                (kernel_path, 1),
                file_registry['floral_resources_[SPECIES]', species]),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': False,
                },
            dependent_task_list=[
                alpha_kernel_raster_task, local_foraging_effectiveness_task],
            target_path_list=[file_registry['floral_resources_[SPECIES]', species]])

        floral_resources_index_task_map[species] = floral_resources_task

        # calculate
        # pollinator_supply_index[species] PS(x,s) = FR(x,s) * HN(x,s) * sa(s)
        pollinator_supply_task = task_graph.add_task(
            task_name=f'calculate_pollinator_supply_{species}',
            func=_calculate_pollinator_supply_index,
            args=(
                file_registry['habitat_nesting_index_[SPECIES]', species],
                file_registry['floral_resources_[SPECIES]', species],
                scenario_variables['species_abundance'][species],
                file_registry['pollinator_supply_[SPECIES]', species]),
            dependent_task_list=[
                floral_resources_task, habitat_nesting_tasks[species]],
            target_path_list=[file_registry['pollinator_supply_[SPECIES]', species]])

        # calc convolved_PS PS over alpha_s
        convolve_ps_task = task_graph.add_task(
            task_name=f'convolve_ps_{species}',
            func=pygeoprocessing.convolve_2d,
            args=(
                (file_registry['pollinator_supply_[SPECIES]', species], 1),
                (kernel_path, 1),
                file_registry['convolve_ps_[SPECIES]', species]),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': False,
                },
            dependent_task_list=[
                alpha_kernel_raster_task, pollinator_supply_task],
            target_path_list=[file_registry['convolve_ps_[SPECIES]', species]])

        for season in scenario_variables['season_list']:
            # calculate pollinator activity as
            # PA(x,s,j)=RA(l(x),j)fa(s,j) convolve(ps, alpha_s)
            pollinator_abundance_task_map[(species, season)] = task_graph.add_task(
                task_name=f'calculate_poll_abudance_{species}',
                func=pygeoprocessing.raster_map,
                kwargs=dict(
                    op=pollinator_supply_op,
                    rasters=[
                        file_registry['foraged_flowers_index_[SPECIES]_[SEASON]', species, season],
                        file_registry['floral_resources_[SPECIES]', species],
                        file_registry['convolve_ps_[SPECIES]', species]],
                    target_path=file_registry['pollinator_abundance_[SPECIES]_[SEASON]', species, season],
                    target_dtype=numpy.float32,
                    target_nodata=_INDEX_NODATA),
                dependent_task_list=[
                    foraged_flowers_index_task_map[(species, season)],
                    floral_resources_index_task_map[species],
                    convolve_ps_task],
                target_path_list=[file_registry['pollinator_abundance_[SPECIES]_[SEASON]', species, season]])

    # calculate total abundance of all pollinators for each season
    total_pollinator_abundance_task = {}
    for season in scenario_variables['season_list']:
        pollinator_abundance_season_path_band_list = [
            (file_registry['pollinator_abundance_[SPECIES]_[SEASON]', species, season], 1)
            for species in scenario_variables['species_list']]

        total_pollinator_abundance_task[season] = task_graph.add_task(
            task_name=f'calculate_poll_abudnce_{species}_{season}',
            func=pygeoprocessing.raster_calculator,
            args=(
                pollinator_abundance_season_path_band_list, _sum_arrays,
                file_registry['total_pollinator_abundance_[SEASON]', season],
                gdal.GDT_Float32, _INDEX_NODATA),
            dependent_task_list=[
                pollinator_abundance_task_map[(species, season)]
                for species in scenario_variables['species_list']],
            target_path_list=[
                file_registry['total_pollinator_abundance_[SEASON]', season]])

    # next step is farm vector calculation, if no farms then okay to quit
    if not args['farm_vector_path']:
        task_graph.close()
        task_graph.join()
        return

    # blank raster used for rasterizing all the farm parameters/fields later
    blank_raster_task = task_graph.add_task(
        task_name='create_blank_raster',
        func=pygeoprocessing.new_raster_from_base,
        args=(
            args['landcover_raster_path'], file_registry['blank_raster'],
            gdal.GDT_Float32, [_INDEX_NODATA]),
        kwargs={'fill_value_list': [_INDEX_NODATA]},
        target_path_list=[file_registry['blank_raster']])

    farm_pollinator_season_task_list = []
    for season in scenario_variables['season_list']:
        half_saturation_task = task_graph.add_task(
            task_name=f'half_saturation_rasterize_{season}',
            func=_rasterize_vector_onto_base,
            args=(
                file_registry['blank_raster'],
                file_registry['reprojected_farm_vector'],
                _HALF_SATURATION_FARM_HEADER,
                file_registry['half_saturation_[SEASON]', season]),
            kwargs={'filter_string': f"{_FARM_SEASON_FIELD}='{season}'"},
            dependent_task_list=[blank_raster_task],
            target_path_list=[file_registry['half_saturation_[SEASON]', season]])
        # calc on farm pollinator abundance i.e. FP_season
        farm_pollinator_season_task_list.append(task_graph.add_task(
            task_name=f'farm_pollinator_{season}',
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=on_farm_pollinator_abundance_op,
                rasters=[
                    file_registry['half_saturation_[SEASON]', season],
                    file_registry['total_pollinator_abundance_[SEASON]', season]],
                target_path=file_registry['farm_pollinator_[SEASON]', season],
                target_dtype=numpy.float32,
                target_nodata=_INDEX_NODATA),
            dependent_task_list=[
                half_saturation_task, total_pollinator_abundance_task[season]],
            target_path_list=[file_registry['farm_pollinator_[SEASON]', season]]))

    # sum farm pollinators
    farm_pollinator_task = task_graph.add_task(
        task_name='sum_farm_pollinators',
        func=pygeoprocessing.raster_calculator,
        args=(
            [(file_registry['farm_pollinator_[SEASON]', season], 1)
                for season in scenario_variables['season_list']],
            _sum_arrays, file_registry['farm_pollinators'], gdal.GDT_Float32,
            _INDEX_NODATA),
        dependent_task_list=farm_pollinator_season_task_list,
        target_path_list=[file_registry['farm_pollinators']])

    # rasterize managed pollinators
    managed_pollinator_task = task_graph.add_task(
        task_name='rasterize_managed_pollinators',
        func=_rasterize_vector_onto_base,
        args=(
            file_registry['blank_raster'],
            file_registry['reprojected_farm_vector'],
            _MANAGED_POLLINATORS_FIELD,
            file_registry['managed_pollinators']),
        dependent_task_list=[reproject_farm_task, blank_raster_task],
        target_path_list=[file_registry['managed_pollinators']])

    # calculate PYT
    pyt_task = task_graph.add_task(
        task_name='calculate_total_pollinators',
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=pyt_op,
            rasters=[file_registry['managed_pollinators'], file_registry['farm_pollinators']],
            target_path=file_registry['total_pollinator_yield'],
            target_dtype=numpy.float32,
            target_nodata=_INDEX_NODATA),
        dependent_task_list=[farm_pollinator_task, managed_pollinator_task],
        target_path_list=[file_registry['total_pollinator_yield']])

    # calculate PYW
    wild_pollinator_task = task_graph.add_task(
        task_name='calculate_wild_pollinators',
        func=pygeoprocessing.raster_map,
        kwargs=dict(
            op=pyw_op,
            rasters=[
                file_registry['managed_pollinators'],
                file_registry['total_pollinator_yield']],
            target_path=file_registry['wild_pollinator_yield'],
            target_dtype=numpy.float32,
            target_nodata=_INDEX_NODATA),
        dependent_task_list=[pyt_task, managed_pollinator_task],
        target_path_list=[file_registry['wild_pollinator_yield']])

    # aggregate yields across farms
    if os.path.exists(file_registry['farm_results']):
        os.remove(file_registry['farm_results'])
    reproject_farm_task.join()
    _create_farm_result_vector(
        file_registry['reprojected_farm_vector'], file_registry['farm_results'])

    # aggregate wild pollinator yield over farm
    wild_pollinator_task.join()
    wild_pollinator_yield_aggregate = pygeoprocessing.zonal_statistics(
        (file_registry['wild_pollinator_yield'], 1), file_registry['farm_results'])

    # aggregate yield over a farm
    pyt_task.join()
    total_farm_results = pygeoprocessing.zonal_statistics(
        (file_registry['total_pollinator_yield'], 1), file_registry['farm_results'])

    # aggregate the pollinator abundance results over the farms
    pollinator_abundance_results = {}
    for season in scenario_variables['season_list']:
        total_pollinator_abundance_task[season].join()
        pollinator_abundance_results[season] = (
            pygeoprocessing.zonal_statistics(
                (file_registry['total_pollinator_abundance_[SEASON]', season], 1),
                file_registry['farm_results']))

    target_farm_vector = gdal.OpenEx(file_registry['farm_results'], 1)
    target_farm_layer = target_farm_vector.GetLayer()

    # aggregate results per farm
    for farm_feature in target_farm_layer:
        nu = float(farm_feature.GetField(_CROP_POLLINATOR_DEPENDENCE_FIELD))
        fid = farm_feature.GetFID()
        if total_farm_results[fid]['count'] > 0:
            # total pollinator farm yield is 1-*nu(1-tot_pollination_coverage)
            # this is YT from the user's guide (y_tot)
            farm_feature.SetField(
                _TOTAL_FARM_YIELD_FIELD_ID,
                float(1 - nu * (
                    1 - total_farm_results[fid]['sum'] /
                    float(total_farm_results[fid]['count']))))

            # this is PYW ('pdep_y_w')
            farm_feature.SetField(
                _POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID,
                float(wild_pollinator_yield_aggregate[fid]['sum'] /
                 float(wild_pollinator_yield_aggregate[fid]['count'])))

            # this is YW ('y_wild')
            farm_feature.SetField(
                _WILD_POLLINATOR_FARM_YIELD_FIELD_ID,
                float(nu * (wild_pollinator_yield_aggregate[fid]['sum'] /
                      float(wild_pollinator_yield_aggregate[fid]['count']))))

            # this is PAT ('p_abund')
            farm_season = farm_feature.GetField(_FARM_SEASON_FIELD)
            farm_feature.SetField(
                _POLLINATOR_ABUNDANCE_FARM_FIELD_ID,
                float(pollinator_abundance_results[farm_season][fid]['sum'] /
                float(pollinator_abundance_results[farm_season][fid]['count'])))

        target_farm_layer.SetFeature(farm_feature)
    target_farm_layer.SyncToDisk()
    target_farm_layer = None
    target_farm_vector = None

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def pollinator_supply_op(
        foraged_flowers_array, floral_resources_array,
        convolve_ps_array):
    """raster_map equation: calculate (RA*fa)/FR * convolve(PS)."""
    return numpy.where(
        floral_resources_array == 0, 0,
        foraged_flowers_array / floral_resources_array * convolve_ps_array)

def on_farm_pollinator_abundance_op(h, pat):
    """raster_map equation: return (pat * (1 - h)) / (h * (1 - 2*pat)+pat))"""
    return (pat * (1 - h)) / (h * (1 - 2 * pat) + pat)

# raster_map equation: return min(mp_array+FP_array, 1)
def pyt_op(mp_array, FP_array): return numpy.minimum(mp_array + FP_array, 1)

# raster_map equation: return max(0,PYT_array-mp_array)
def pyw_op(mp_array, PYT_array): return numpy.maximum(PYT_array - mp_array, 0)


def _rasterize_vector_onto_base(
        base_raster_path, base_vector_path, attribute_id,
        target_raster_path, filter_string=None):
    """Rasterize attribute from vector onto a copy of base.

    Args:
        base_raster_path (string): path to a base raster file
        attribute_id (string): id in `base_vector_path` to rasterize.
        target_raster_path (string): a copy of `base_raster_path` with
            `base_vector_path[attribute_id]` rasterized on top.
        filter_string (string): filtering string to select from farm layer

    Returns:
        None.
    """
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    raster_driver = gdal.GetDriverByName('GTiff')
    target_raster = raster_driver.CreateCopy(target_raster_path, base_raster)
    base_raster = None

    vector = gdal.OpenEx(base_vector_path)
    layer = vector.GetLayer()

    if filter_string is not None:
        layer.SetAttributeFilter(str(filter_string))
    gdal.RasterizeLayer(
        target_raster, [1], layer,
        options=[f'ATTRIBUTE={attribute_id}'])
    target_raster.FlushCache()
    target_raster = None
    layer = None
    vector = None


def _create_farm_result_vector(
        base_vector_path, target_vector_path):
    """Create a copy of `base_vector_path` and add FID field to it.

    Args:
        base_vector_path (string): path to vector to copy
        target_vector_path (string): path to target vector. This path must
            not already exist. Vector will be created at this path that is
            a copy of the base vector with result fields added:
                pollination._POLLINATOR_ABUNDANCE_FARM_FIELD_ID,
                pollination._TOTAL_FARM_YIELD_FIELD_ID,
                pollination._POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID,
                pollination._WILD_POLLINATOR_FARM_YIELD_FIELD_ID

    Returns:
        None.

    """
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)

    driver = gdal.GetDriverByName('ESRI Shapefile')
    target_vector = driver.CreateCopy(target_vector_path, base_vector)
    target_layer = target_vector.GetLayer()

    farm_pollinator_abundance_defn = ogr.FieldDefn(
        _POLLINATOR_ABUNDANCE_FARM_FIELD_ID, ogr.OFTReal)
    farm_pollinator_abundance_defn.SetWidth(25)
    farm_pollinator_abundance_defn.SetPrecision(11)
    target_layer.CreateField(farm_pollinator_abundance_defn)

    total_farm_yield_field_defn = ogr.FieldDefn(
        _TOTAL_FARM_YIELD_FIELD_ID, ogr.OFTReal)
    total_farm_yield_field_defn.SetWidth(25)
    total_farm_yield_field_defn.SetPrecision(11)
    target_layer.CreateField(total_farm_yield_field_defn)

    pol_proportion_farm_yield_field_defn = ogr.FieldDefn(
        _POLLINATOR_PROPORTION_FARM_YIELD_FIELD_ID, ogr.OFTReal)
    pol_proportion_farm_yield_field_defn.SetWidth(25)
    pol_proportion_farm_yield_field_defn.SetPrecision(11)
    target_layer.CreateField(pol_proportion_farm_yield_field_defn)

    wild_pol_farm_yield_field_defn = ogr.FieldDefn(
        _WILD_POLLINATOR_FARM_YIELD_FIELD_ID, ogr.OFTReal)
    wild_pol_farm_yield_field_defn.SetWidth(25)
    wild_pol_farm_yield_field_defn.SetPrecision(11)
    target_layer.CreateField(wild_pol_farm_yield_field_defn)

    target_layer = None
    target_vector.FlushCache()
    target_vector = None


def _parse_scenario_variables(args):
    """Parse out scenario variables from input parameters.

    This function parses through the guild table, biophysical table, and
    farm polygons (if available) to generate

    Parameter:
        args (dict): this is the args dictionary passed in to the `execute`
            function, requires a 'guild_table_path', and
            'landcover_biophysical_table_path' key and optional
            'farm_vector_path' key.

    Returns:
        A dictionary with the keys:
            * season_list (list of string)
            * substrate_list (list of string)
            * species_list (list of string)
            * alpha_value[species] (float)
            * landcover_substrate_index[substrate][landcover] (float)
            * landcover_floral_resources[season][landcover] (float)
            * species_abundance[species] (string->float)
            * species_foraging_activity[(species, season)] (string->float)
            * species_substrate_index[(species, substrate)] (tuple->float)
            * foraging_activity_index[(species, season)] (tuple->float)
    """
    guild_df = MODEL_SPEC.get_input(
        'guild_table_path').get_validated_dataframe(args['guild_table_path'])

    LOGGER.info('Checking to make sure guild table has all expected headers')
    for header in _EXPECTED_GUILD_HEADERS:
        matches = re.findall(header, " ".join(guild_df.columns))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in guild table that matched the pattern "
                f"'{header}' but was unable to find one. Here are all the "
                f"headers from {args['guild_table_path']}: "
                f"{', '.join(guild_df.columns)}")

    landcover_biophysical_df = MODEL_SPEC.get_input(
        'landcover_biophysical_table_path').get_validated_dataframe(
        args['landcover_biophysical_table_path'])
    biophysical_table_headers = landcover_biophysical_df.columns
    for header in _EXPECTED_BIOPHYSICAL_HEADERS:
        matches = re.findall(header, " ".join(biophysical_table_headers))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in biophysical table that matched the "
                f"pattern '{header}' but was unable to find one. Here are all "
                f"the headers from {args['landcover_biophysical_table_path']}: "
                f"{', '.join(biophysical_table_headers)}")

    # this dict to dict will map seasons to guild/biophysical headers
    # ex season_to_header['spring']['guilds']
    season_to_header = collections.defaultdict(dict)
    # this dict to dict will map substrate types to guild/biophysical headers
    # ex substrate_to_header['cavity']['biophysical']
    substrate_to_header = collections.defaultdict(dict)
    for header in guild_df.columns:
        match = re.match(_FORAGING_ACTIVITY_RE_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]['guild'] = match.group()
        match = re.match(_NESTING_SUITABILITY_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]['guild'] = match.group()

    farm_vector = None
    if args['farm_vector_path']:
        LOGGER.info('Checking that farm polygon has expected headers')
        farm_vector = gdal.OpenEx(args['farm_vector_path'])
        farm_layer = farm_vector.GetLayer()
        if farm_layer.GetGeomType() not in [
                ogr.wkbPolygon, ogr.wkbMultiPolygon]:
            farm_layer = None
            farm_vector = None
            raise ValueError("Farm layer not a polygon type")
        farm_layer_defn = farm_layer.GetLayerDefn()
        farm_headers = [
            farm_layer_defn.GetFieldDefn(i).GetName()
            for i in range(farm_layer_defn.GetFieldCount())]
        for header in _EXPECTED_FARM_HEADERS:
            matches = re.findall(header, " ".join(farm_headers))
            if not matches:
                raise ValueError(
                    f"Missing expected header(s) '{header}' from "
                    f"{args['farm_vector_path']}.\n"
                    f"Got these headers instead: {farm_headers}")

        for header in farm_headers:
            match = re.match(_FARM_FLORAL_RESOURCES_PATTERN, header)
            if match:
                season = match.group(1)
                season_to_header[season]['farm'] = match.group()
            match = re.match(_FARM_NESTING_SUBSTRATE_RE_PATTERN, header)
            if match:
                substrate = match.group(1)
                substrate_to_header[substrate]['farm'] = match.group()

    for header in biophysical_table_headers:
        match = re.match(_FLORAL_RESOURCES_AVAILABLE_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]['biophysical'] = match.group()
        match = re.match(_NESTING_SUBSTRATE_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]['biophysical'] = match.group()

    for table_type, lookup_table in itertools.chain(
            season_to_header.items(), substrate_to_header.items()):
        if len(lookup_table) != 3 and farm_vector is not None:
            raise ValueError(
                "Expected a biophysical, guild, and farm entry for "
                f"'{table_type}' but instead found only "
                f"{list(lookup_table.keys())}. Ensure there are "
                f"corresponding entries of '{table_type}' in both the "
                "guilds, biophysical table, and farm fields.")
        elif len(lookup_table) != 2 and farm_vector is None:
            raise ValueError(
                f"Expected a biophysical and guild entry for '{table_type}' "
                f"but instead found only {list(lookup_table.keys())}. "
                "Ensure there are corresponding entries of '{table_type}' in "
                "both the guilds and biophysical table.")

    if args['farm_vector_path']:
        farm_season_set = set()
        for farm_feature in farm_layer:
            farm_season_set.add(farm_feature.GetField(_FARM_SEASON_FIELD))

        if len(farm_season_set.difference(season_to_header)) > 0:
            raise ValueError(
                "Found seasons in farm polygon that were not specified in the "
                "biophysical table: "
                f"{', '.join(farm_season_set.difference(season_to_header))}. "
                f"Expected only these: {', '.join(season_to_header.keys())}")

    result = {}
    # * season_list (list of string)
    result['season_list'] = sorted(season_to_header)
    # * substrate_list (list of string)
    result['substrate_list'] = sorted(substrate_to_header)
    # * species_list (list of string)
    result['species_list'] = sorted(guild_df.index)

    result['alpha_value'] = dict()
    for species in result['species_list']:
        result['alpha_value'][species] = guild_df[_ALPHA_HEADER][species]

    # * species_abundance[species] (string->float)
    total_relative_abundance = guild_df[_RELATIVE_SPECIES_ABUNDANCE_FIELD].sum()
    result['species_abundance'] = {}
    for species in result['species_list']:
        result['species_abundance'][species] = (
            guild_df[_RELATIVE_SPECIES_ABUNDANCE_FIELD][species] /
            total_relative_abundance)

    # map the relative foraging activity of a species during a certain season
    # (species, season)
    result['species_foraging_activity'] = dict()
    for species in result['species_list']:
        total_activity = numpy.sum([
            guild_df[_FORAGING_ACTIVITY_PATTERN % season][species]
            for season in result['season_list']])
        for season in result['season_list']:
            result['species_foraging_activity'][(species, season)] = (
                guild_df[_FORAGING_ACTIVITY_PATTERN % season][species] /
                total_activity)

    # * landcover_substrate_index[substrate][landcover] (float)
    result['landcover_substrate_index'] = collections.defaultdict(dict)
    for landcover_id, row in landcover_biophysical_df.iterrows():
        for substrate in result['substrate_list']:
            substrate_biophysical_header = (
                substrate_to_header[substrate]['biophysical'])
            result['landcover_substrate_index'][substrate][landcover_id] = (
                row[substrate_biophysical_header])

    # * landcover_floral_resources[season][landcover] (float)
    result['landcover_floral_resources'] = collections.defaultdict(dict)
    for landcover_id, row in landcover_biophysical_df.iterrows():
        for season in result['season_list']:
            floral_rources_header = season_to_header[season]['biophysical']
            result['landcover_floral_resources'][season][landcover_id] = (
                row[floral_rources_header])

    # * species_substrate_index[(species, substrate)] (tuple->float)
    result['species_substrate_index'] = collections.defaultdict(dict)
    for species in result['species_list']:
        for substrate in result['substrate_list']:
            substrate_guild_header = substrate_to_header[substrate]['guild']
            result['species_substrate_index'][species][substrate] = (
                guild_df[substrate_guild_header][species])

    # * foraging_activity_index[(species, season)] (tuple->float)
    result['foraging_activity_index'] = {}
    for species in result['species_list']:
        for season in result['season_list']:
            key = (species, season)
            foraging_biophyiscal_header = season_to_header[season]['guild']
            result['foraging_activity_index'][key] = (
                guild_df[foraging_biophyiscal_header][species])

    return result


def _sum_arrays(*array_list):
    """Calculate sum of array_list and account for nodata."""
    valid_mask = numpy.zeros(array_list[0].shape, dtype=bool)
    result = numpy.empty_like(array_list[0])
    result[:] = 0
    for array in array_list:
        local_valid_mask = ~pygeoprocessing.array_equals_nodata(
            array, _INDEX_NODATA)
        result[local_valid_mask] += array[local_valid_mask]
        valid_mask |= local_valid_mask
    result[~valid_mask] = _INDEX_NODATA
    return result


def _calculate_habitat_nesting_index(
        substrate_path_map, species_substrate_index_map,
        target_habitat_nesting_index_path):
    """Calculate HN(x, s) = max_n(N(x, n) ns(s,n))."""

    substrate_path_list = [
        substrate_path_map[substrate_id]
        for substrate_id in sorted(substrate_path_map)]

    species_substrate_suitability_index_array = numpy.array([
        species_substrate_index_map[substrate_id]
        for substrate_id in sorted(substrate_path_map)]).reshape(
            (len(species_substrate_index_map), 1))

    def max_op(*substrate_index_arrays):
        """Return the max of index_array[n] * ns[n]."""
        return numpy.max(
            numpy.stack([x.flatten() for x in substrate_index_arrays]) *
            species_substrate_suitability_index_array, axis=0
        ).reshape(substrate_index_arrays[0].shape)

    pygeoprocessing.raster_map(
        op=max_op,
        rasters=substrate_path_list,
        target_dtype=numpy.float32,
        target_path=target_habitat_nesting_index_path)


def _multiply_by_scalar(raster_path, scalar, target_path):
    """Multiply a raster by a scalar and write out result."""
    pygeoprocessing.raster_map(
        op=lambda array: array * scalar,
        rasters=[raster_path],
        target_path=target_path,
        target_dtype=numpy.float32,
        target_nodata=_INDEX_NODATA
    )


def _calculate_pollinator_supply_index(
        habitat_nesting_suitability_path, floral_resources_path,
        species_abundance, target_path):
    """Calculate pollinator supply index..

    Args:
        habitat_nesting_suitability_path (str): path to habitat nesting
            suitability raster
        floral_resources_path (str): path to floral resources raster
        species_abundance (float): species abundance value
        target_path (str): path to write out result raster

    Returns:
        None.
    """
    pygeoprocessing.raster_map(
        op=lambda f_r, h_n: species_abundance * f_r * h_n,
        rasters=[habitat_nesting_suitability_path, floral_resources_path],
        target_path=target_path,
        target_dtype=numpy.float32,
        target_nodata=_INDEX_NODATA
    )


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
    # Deliberately not validating the interrelationship of the columns between
    # the biophysical table and the guilds table as the model itself already
    # does extensive checking for this.
    return validation.validate(args, MODEL_SPEC)
