"""Pollinator service model for InVEST."""
import multiprocessing
import tempfile
import itertools
import collections
import re
import os
import logging
import uuid

from osgeo import gdal
from osgeo import ogr
import pygeoprocessing
import numpy
import taskgraph

from . import utils

LOGGER = logging.getLogger('natcap.invest.pollination')


_OUTPUT_BASE_FILES = {
    }

_INTERMEDIATE_BASE_FILES = {
    }

_TMP_BASE_FILES = {
    }

_INDEX_NODATA = -1.0

_MANAGED_BEES_RASTER_FILE_PATTERN = r'managed_bees_%s'
_SPECIES_ALPHA_KERNEL_FILE_PATTERN = r'alpha_kernel_%s'
_ACCESSIBLE_FLORAL_RESOURCES_FILE_PATTERN = r'accessible_floral_resources_%s'
_LOCAL_POLLINATOR_SUPPLY_FILE_PATTERN = r'local_pollinator_supply_%s_index'
_POLLINATOR_ABUNDANCE_FILE_PATTERN = r'pollinator_abundance_%s_index'
_SEASONAL_POLLINATOR_ABUNDANCE_FILE_PATTERN = (
    r'seasonal_pollinator_abundance_%s_%s_index')
_TOTAL_SEASONAL_POLLINATOR_ABUNDANCE_FILE_PATTERN = (
    r'total_seasonal_pollinator_abundance_%s_index')
_RAW_POLLINATOR_ABUNDANCE_FILE_PATTERN = r'raw_pollinator_abundance_%s_index'
_LOCAL_FLORAL_RESOURCE_AVAILABILITY_FILE_PATTERN = (
    r'local_floral_resource_availability_%s_index')
_NESTING_SUITABILITY_SPECIES_PATTERN = r'nesting_suitability_%s_index'
_PROJECTED_FARM_VECTOR_FILE_PATTERN = 'projected_farm_vector%s.shp'

# These patterns are expected in the biophysical table
_NESTING_SUBSTRATE_PATTERN = 'nesting_([^_]+)_availability_index'
_FLORAL_RESOURCES_AVAILABLE_PATTERN = 'floral_resources_([^_]+)_index'
_EXPECTED_BIOPHYSICAL_HEADERS = [
    'lucode', _NESTING_SUBSTRATE_PATTERN, _FLORAL_RESOURCES_AVAILABLE_PATTERN]

# These are patterns expected in the guilds table
_NESTING_SUITABILITY_PATTERN = 'nesting_suitability_([^_]+)_index'
_FORAGING_ACTIVITY_PATTERN = 'foraging_activity_([^_]+)_index'
_RELATIVE_SPECIES_ABUNDANCE_FIELD = 'relative_abundance'
_EXPECTED_GUILD_HEADERS = [
    'species', _NESTING_SUITABILITY_PATTERN, _FORAGING_ACTIVITY_PATTERN,
    'alpha', _RELATIVE_SPECIES_ABUNDANCE_FIELD]

_HALF_SATURATION_SEASON_FILE_PATTERN = 'half_saturation_%s'
_FARM_POLLINATORS_FILE_PATTERN = 'farm_pollinators_%s'
_FARM_FLORAL_RESOURCES_PATTERN = 'fr_([^_]+)'
_FARM_NESTING_SUBSTRATE_PATTERN = 'n_([^_]+)'
_HALF_SATURATION_FARM_HEADER = 'half_sat'
_CROP_POLLINATOR_DEPENDENCE_FIELD = 'p_dep'
_EXPECTED_FARM_HEADERS = [
    'season', 'crop_type', _HALF_SATURATION_FARM_HEADER, 'p_managed',
    _FARM_FLORAL_RESOURCES_PATTERN, _FARM_NESTING_SUBSTRATE_PATTERN,
    _CROP_POLLINATOR_DEPENDENCE_FIELD]

_SEASONAL_POLLINATOR_YIELD_FILE_PATTERN = 'seasonal_pollinator_yield_%s'
_TOTAL_POLLINATOR_YIELD_FILE_PATTERN = 'total_pollinator_yield'
_TARGET_AGGREGATE_FARM_VECTOR_FILE_PATTERN = 'farm_yield'
_POLLINATOR_FARM_YIELD_FIELD_ID = 'p_av_yield'
_TOTAL_FARM_YIELD_FIELD_ID = 't_av_yield'

_MAX_POLLINATED_SPECIES_FILE_PATTERN = 'max_pollinated_%s'


def execute(args):
    """InVEST Pollination Model.

    Parameters:
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
                    with values in the range [0.0, 1.0] indicating the
                    suitability of the given species to nest in a particular
                    substrate.
                * one or more columns matching _FORAGING_ACTIVITY_PATTERN
                    with values in the range [0.0, 1.0] indicating the
                    relative level of foraging activity for that species
                    during a particular season.
                * 'alpha': the sigma average flight distance of that bee
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

            All indexes are in the range [0.0, 1.0].

            Columns in the table must be at least
                * 'lucode': representing all the unique landcover codes in
                    the raster ast `args['landcover_path']`
                * For every nesting matching _NESTING_SUITABILITY_PATTERN
                  in the guild stable, a column matching the pattern in
                  `_LANDCOVER_NESTING_INDEX_HEADER`.
                * For every season matching _FORAGING_ACTIVITY_PATTERN
                  in the guilds table, a column matching
                  the pattern in `_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`.
        args['farm_vector_path'] (string): (optional) path to a single layer
            polygon shapefile representing farms. If present will trigger the
            farm yield component of the model.

            The layer must have at least the following fields:

            * season (string): season in which the farm needs pollination
            * crop_type (string): a text field to identify the crop type for
                summary statistics.
            * half_sat (float): a real in the range [0.0, 1.0] representing
                the proportion of wild pollinators to achieve a 50% yield
                of that crop.
            * p_dep (float): a number in the range [0.0, 1.0]
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

    Returns:
        None
    """
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    work_token_dir = os.path.join(
        intermediate_output_dir, '_tmp_work_tokens')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories(
        [output_dir, intermediate_output_dir])

    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)
    temp_file_set = set()  # to keep track of temporary files to delete

    guild_table = utils.build_lookup_from_csv(
        args['guild_table_path'], 'species', to_lower=True,
        numerical_cast=True)

    LOGGER.info('Checking to make sure guild table has all expected headers')
    guild_headers = guild_table.itervalues().next().keys()

    # normalize relative species abundances
    total_relative_abundance = numpy.sum([
        guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD]
        for species in guild_table])
    for species in guild_table:
        guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD] /= (
            total_relative_abundance)
    # we need to match at least one of each of expected
    for header in _EXPECTED_GUILD_HEADERS:
        matches = re.findall(header, " ".join(guild_headers))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in guild table that matched the pattern "
                "'%s' but was unable to find one.  Here are all the headers "
                "from %s: %s" % (
                    header, args['guild_table_path'],
                    guild_headers))

    # this dict to dict will map seasons to guild/biophysical headers
    # ex season_to_header['spring']['guilds']
    season_to_header = collections.defaultdict(dict)
    # this dict to dict will map substrate types to guild/biophysical headers
    # ex substrate_to_header['cavity']['biophysical']
    substrate_to_header = collections.defaultdict(dict)
    for header in guild_headers:
        match = re.match(_FORAGING_ACTIVITY_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]['guild'] = match.group()
        match = re.match(_NESTING_SUITABILITY_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]['guild'] = match.group()

    landcover_biophysical_table = utils.build_lookup_from_csv(
        args['landcover_biophysical_table_path'], 'lucode', to_lower=True,
        numerical_cast=True)
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    biophysical_table_headers = (
        landcover_biophysical_table.itervalues().next().keys())
    for header in _EXPECTED_BIOPHYSICAL_HEADERS:
        matches = re.findall(header, " ".join(biophysical_table_headers))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in biophysical table that matched the "
                "pattern '%s' but was unable to find one.  Here are all the "
                "headers from %s: %s" % (
                    header, args['landcover_biophysical_table_path'],
                    biophysical_table_headers))

    farm_vector = None
    if 'farm_vector_path' in args and args['farm_vector_path'] != '':
        LOGGER.info('Checking that farm polygon has expected headers')
        farm_vector = ogr.Open(args['farm_vector_path'])
        if farm_vector.GetLayerCount() != 1:
            raise ValueError(
                "Farm polygon at %s has %d layers when expecting only 1." % (
                    args['farm_vector_path'], farm_vector.GetLayerCount()))
        farm_layer = farm_vector.GetLayer()
        if farm_layer.GetGeomType() not in [
                ogr.wkbPolygon, ogr.wkbMultiPolygon]:
            farm_layer = None
            farm_vector = None
            raise ValueError("Farm layer not a polygon type")
        farm_layer_defn = farm_layer.GetLayerDefn()
        farm_headers = [
            farm_layer_defn.GetFieldDefn(i).GetName()
            for i in xrange(farm_layer_defn.GetFieldCount())]
        for header in _EXPECTED_FARM_HEADERS:
            matches = re.findall(header, " ".join(farm_headers))
            if len(matches) == 0:
                raise ValueError(
                    "Missing an expected headers '%s'from %s.\n"
                    "Got these headers instead %s" % (
                        header, args['farm_vector_path'], farm_headers))

        for header in farm_headers:
            match = re.match(_FARM_FLORAL_RESOURCES_PATTERN, header)
            if match:
                season = match.group(1)
                season_to_header[season]['farm'] = match.group()
            match = re.match(_FARM_NESTING_SUBSTRATE_PATTERN, header)
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
            season_to_header.iteritems(), substrate_to_header.iteritems()):
        if len(lookup_table) != 3 and farm_vector is not None:
            raise ValueError(
                "Expected a biophysical, guild, and farm entry for '%s' but "
                "instead found only %s. Ensure there are corresponding "
                "entries of '%s' in both the guilds, biophysical "
                "table, and farm fields." % (
                    table_type, lookup_table, table_type))
        elif len(lookup_table) != 2 and farm_vector is None:
            raise ValueError(
                "Expected a biophysical, and guild entry for '%s' but "
                "instead found only %s. Ensure there are corresponding "
                "entries of '%s' in both the guilds and biophysical "
                "table." % (
                    table_type, lookup_table, table_type))

    task_graph = taskgraph.TaskGraph(
        work_token_dir, 0)#multiprocessing.cpu_count())

    # farms can be optional
    reproject_farm_task = None
    if farm_vector is not None:
        farm_season_set = set()
        for farm_feature in farm_layer:
            farm_season_set.add(farm_feature.GetField('season'))

        if len(farm_season_set.difference(season_to_header)) > 0:
            raise ValueError(
                "Found seasons in farm polygon that were not specified in the"
                "biophysical table: %s.  Expected only these: %s" % (
                    farm_season_set.difference(season_to_header),
                    season_to_header))

        # ensure the farm vector is in the same projection as the landcover map
        projected_farm_vector_path = os.path.join(
            intermediate_output_dir,
            _PROJECTED_FARM_VECTOR_FILE_PATTERN % file_suffix)
        reproject_farm_task = task_graph.add_task(
            target=pygeoprocessing.reproject_vector,
            args=(
                args['farm_vector_path'], lulc_raster_info['projection'],
                projected_farm_vector_path),
            target_path_list=[projected_farm_vector_path])

    nesting_substrate_task_list = []
    for nesting_substrate in substrate_to_header:
        LOGGER.info(
            "Mapping landcover to nesting substrate %s", nesting_substrate)
        nesting_id = substrate_to_header[nesting_substrate]['biophysical']
        landcover_to_nesting_suitability_table = dict([
            (lucode, landcover_biophysical_table[lucode][nesting_id]) for
            lucode in landcover_biophysical_table])

        f_reg[nesting_id] = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (nesting_id, file_suffix))

        landcover_nesting_reclass_task = task_graph.add_task(
            target=pygeoprocessing.reclassify_raster,
            args=(
                (args['landcover_raster_path'], 1),
                landcover_to_nesting_suitability_table, f_reg[nesting_id],
                gdal.GDT_Float32, _INDEX_NODATA),
            target_path_list=[f_reg[nesting_id]])

        nesting_substrate_task_list.append(landcover_nesting_reclass_task)

        if farm_vector is not None:
            LOGGER.info(
                "Overriding landcover nesting substrates where a farm "
                " polygon is available.")

            farm_substrate_rasterize_task = task_graph.add_task(
                target=pygeoprocessing.rasterize,
                args=(
                    projected_farm_vector_path, f_reg[nesting_id],
                    None, ['ATTRIBUTE=%s' % (
                        _FARM_NESTING_SUBSTRATE_PATTERN.replace(
                            '([^_]+)', nesting_substrate))]),
                target_path_list=[f_reg[nesting_id]],
                dependent_task_list=[
                    landcover_nesting_reclass_task, reproject_farm_task])
            nesting_substrate_task_list.append(farm_substrate_rasterize_task)

    nesting_substrate_path_list = [
        f_reg[substrate_to_header[nesting_substrate]['biophysical']]
        for nesting_substrate in sorted(substrate_to_header)]

    species_nesting_suitability_index = None
    species_nesting_task_lookup = {}
    for species_id in guild_table.iterkeys():
        LOGGER.info("Calculate species nesting index for %s", species_id)
        species_nesting_suitability_index = numpy.array([
            guild_table[species_id][substrate_to_header[substrate]['guild']]
            for substrate in sorted(substrate_to_header)])
        species_nesting_suitability_id = (
            _NESTING_SUITABILITY_SPECIES_PATTERN % species_id)

        f_reg[species_nesting_suitability_id] = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                species_nesting_suitability_id, file_suffix))

        species_nesting_task = task_graph.add_task(
            target=pygeoprocessing.raster_calculator,
            args=(
                [(path, 1) for path in nesting_substrate_path_list],
                _HabitatSuitabilityIndexOp(species_nesting_suitability_index),
                f_reg[species_nesting_suitability_id], gdal.GDT_Float32,
                _INDEX_NODATA),
            kwargs={'calc_raster_stats': False},
            target_path_list=[f_reg[species_nesting_suitability_id]],
            dependent_task_list=nesting_substrate_task_list)
        species_nesting_task_lookup[species_id] = species_nesting_task

    floral_resources_task_lookup = {}
    farm_floral_resources_task_lookup = {}
    for season_id in season_to_header:
        LOGGER.info(
            "Mapping landcover to available floral resources for season %s",
            season_id)
        relative_floral_resources_id = (
            season_to_header[season_id]['biophysical'])
        f_reg[relative_floral_resources_id] = os.path.join(
            intermediate_output_dir,
            "%s%s.tif" % (relative_floral_resources_id, file_suffix))

        landcover_to_floral_resources_table = dict([
            (lucode, landcover_biophysical_table[lucode][
                relative_floral_resources_id]) for lucode in
            landcover_biophysical_table])

        floral_resources_reclassify_task = task_graph.add_task(
            target=pygeoprocessing.reclassify_raster,
            args=(
                (args['landcover_raster_path'], 1),
                landcover_to_floral_resources_table,
                f_reg[relative_floral_resources_id], gdal.GDT_Float32,
                _INDEX_NODATA),
            target_path_list=[f_reg[relative_floral_resources_id]])
        floral_resources_task_lookup[season_id] = (
            floral_resources_reclassify_task)

        # farm vector is optional
        if farm_vector is not None:
            LOGGER.info(
                "Overriding landcover floral resources with a farm's floral "
                "resources.")

            farm_floral_resources_raster_task = task_graph.add_task(
                target=pygeoprocessing.rasterize,
                args=(
                    projected_farm_vector_path,
                    f_reg[relative_floral_resources_id], None,
                    ['ATTRIBUTE=%s' % (_FARM_FLORAL_RESOURCES_PATTERN.replace(
                        '([^_]+)', season_id))]),
                target_path_list=[f_reg[relative_floral_resources_id]],
                dependent_task_list=[
                    floral_resources_reclassify_task, reproject_farm_task])
            farm_floral_resources_task_lookup[season_id] = (
                farm_floral_resources_raster_task)

    floral_resources_path_list = [
        f_reg[season_to_header[season_id]['biophysical']]
        for season_id in sorted(season_to_header)]

    species_foraging_activity_per_season = None
    species_abundance = None
    raw_abundance_nodata = None
    seasonal_pollinator_abundance_task_list_lookup = collections.defaultdict(
        list)
    seasonal_pollinator_abundance_path_band_list_lookup = (
        collections.defaultdict(list))
    for species_id in guild_table:
        LOGGER.info(
            "Making local floral resources map for species %s", species_id)
        species_foraging_activity_per_season = numpy.array([
            guild_table[species_id][
                season_to_header[season_id]['guild']]
            for season_id in sorted(season_to_header)])
        # normalize the species foraging activity so it sums to 1.0
        species_foraging_activity_per_season = (
            species_foraging_activity_per_season / sum(
                species_foraging_activity_per_season))
        local_floral_resource_availability_id = (
            _LOCAL_FLORAL_RESOURCE_AVAILABILITY_FILE_PATTERN % species_id)
        f_reg[local_floral_resource_availability_id] = os.path.join(
            intermediate_output_dir, "%s%s.tif" % (
                local_floral_resource_availability_id, file_suffix))

        species_floral_resources_task = task_graph.add_task(
            target=pygeoprocessing.raster_calculator,
            args=(
                [(path, 1) for path in floral_resources_path_list],
                _SpeciesFloralAbudanceOp(
                    species_foraging_activity_per_season),
                f_reg[local_floral_resource_availability_id],
                gdal.GDT_Float32, _INDEX_NODATA),
            kwargs={'calc_raster_stats': False},
            target_path_list=[f_reg[local_floral_resource_availability_id]],
            dependent_task_list=floral_resources_task_lookup.values())

        alpha = (
            guild_table[species_id]['alpha'] /
            float(lulc_raster_info['mean_pixel_size']))
        species_file_kernel_id = (
            _SPECIES_ALPHA_KERNEL_FILE_PATTERN % species_id)
        f_reg[species_file_kernel_id] = os.path.join(
            output_dir, species_file_kernel_id + '%s.tif' % file_suffix)

        alpha_kernel_raster_task = task_graph.add_task(
            target=utils.exponential_decay_kernel_raster,
            args=(alpha, f_reg[species_file_kernel_id]),
            target_path_list=[f_reg[species_file_kernel_id]])

        accessible_floral_resouces_id = (
            _ACCESSIBLE_FLORAL_RESOURCES_FILE_PATTERN % species_id)
        f_reg[accessible_floral_resouces_id] = os.path.join(
            intermediate_output_dir, accessible_floral_resouces_id +
            '%s.tif' % file_suffix)
        temp_file_set.add(f_reg[species_file_kernel_id])
        LOGGER.info(
            "Calculating available floral resources for %s", species_id)

        convolve_2d_nodata = numpy.finfo(numpy.float32).min
        convolve_local_floral_resource_task = task_graph.add_task(
            target=_normalized_convolve_2d,
            args=(
                (f_reg[local_floral_resource_availability_id], 1),
                (f_reg[species_file_kernel_id], 1),
                f_reg[accessible_floral_resouces_id],
                gdal.GDT_Float32, convolve_2d_nodata, args['workspace_dir']),
            target_path_list=[f_reg[accessible_floral_resouces_id]],
            ignore_path_list=[args['workspace_dir']],
            dependent_task_list=[
                species_floral_resources_task, alpha_kernel_raster_task])

        LOGGER.info("Calculating local pollinator supply for %s", species_id)
        pollinator_supply_id = (
            _LOCAL_POLLINATOR_SUPPLY_FILE_PATTERN % species_id)
        f_reg[pollinator_supply_id] = os.path.join(
            intermediate_output_dir,
            pollinator_supply_id + "%s.tif" % file_suffix)

        species_abundance = guild_table[species_id][
            _RELATIVE_SPECIES_ABUNDANCE_FIELD]

        species_nesting_suitability_id = (
            _NESTING_SUITABILITY_SPECIES_PATTERN % species_id)

        pollinator_supply_task = task_graph.add_task(
            target=pygeoprocessing.raster_calculator,
            args=(
                [(f_reg[accessible_floral_resouces_id], 1),
                 (f_reg[species_nesting_suitability_id], 1)],
                _PollinatorSupplyOp(species_abundance),
                f_reg[pollinator_supply_id],
                gdal.GDT_Float32, _INDEX_NODATA),
            kwargs={'calc_raster_stats': False},
            target_path_list=[f_reg[pollinator_supply_id]],
            dependent_task_list=[convolve_local_floral_resource_task])

        LOGGER.info("Calculating raw pollinator abundance for %s", species_id)
        raw_pollinator_abundance_id = (
            _RAW_POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id)
        f_reg[raw_pollinator_abundance_id] = os.path.join(
            intermediate_output_dir, "%s%s.tif" % (
                raw_pollinator_abundance_id, file_suffix))
        raw_pollinator_abundance_task = task_graph.add_task(
            target=_normalized_convolve_2d,
            args=(
                (f_reg[pollinator_supply_id], 1),
                (f_reg[species_file_kernel_id], 1),
                f_reg[raw_pollinator_abundance_id],
                gdal.GDT_Float32, convolve_2d_nodata, args['workspace_dir']),
            dependent_task_list=[
                pollinator_supply_task, alpha_kernel_raster_task])

        for season_index, season_id in enumerate(sorted(season_to_header)):
            LOGGER.info(
                "Calculating seasonal pollinator abundance by scaling the raw by "
                "floral resources available for %s during season %s",
                species_id, season_id)
            seasonal_pollinator_abundance_id = (
                _SEASONAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                    species_id, season_id))
            f_reg[seasonal_pollinator_abundance_id] = os.path.join(
                output_dir,
                seasonal_pollinator_abundance_id + "%s.tif" % file_suffix)

            seasonal_pollinator_abundance_task = task_graph.add_task(
                target=pygeoprocessing.raster_calculator,
                args=(
                    [(f_reg[raw_pollinator_abundance_id], 1),
                     (f_reg[local_floral_resource_availability_id], 1)],
                    _PollinatorAbudanceOp(
                        species_foraging_activity_per_season[season_index],
                        convolve_2d_nodata),
                    f_reg[seasonal_pollinator_abundance_id], gdal.GDT_Float32,
                    _INDEX_NODATA),
                kwargs={'calc_raster_stats': False},
                target_path_list=[f_reg[seasonal_pollinator_abundance_id]],
                dependent_task_list=[
                    raw_pollinator_abundance_task,
                    convolve_local_floral_resource_task])
            seasonal_pollinator_abundance_path_band_list_lookup[season_id].append(
                (f_reg[seasonal_pollinator_abundance_id], 1))
            seasonal_pollinator_abundance_task_list_lookup[season_id].append(
                seasonal_pollinator_abundance_task)

    total_seasonal_pollinator_abundance_task_lookup = {}
    total_seasonal_pollinator_abundance_path_lookup = {}
    for season_id in season_to_header:
        total_seasonal_pollinator_abundance_id = (
            _TOTAL_SEASONAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % season_id)
        f_reg[total_seasonal_pollinator_abundance_id] = os.path.join(
            output_dir, total_seasonal_pollinator_abundance_id +
            "%s.tif" % file_suffix)
        total_seasonal_pollinator_abundance_task = task_graph.add_task(
            target=pygeoprocessing.raster_calculator,
            args=(
                seasonal_pollinator_abundance_path_band_list_lookup[
                    season_id], _AddRasterOp(_INDEX_NODATA),
                f_reg[total_seasonal_pollinator_abundance_id],
                gdal.GDT_Float32, _INDEX_NODATA),
            target_path_list=[f_reg[total_seasonal_pollinator_abundance_id]],
            dependent_task_list=(
                seasonal_pollinator_abundance_task_list_lookup[season_id]))
        total_seasonal_pollinator_abundance_task_lookup[season_id] = (
            total_seasonal_pollinator_abundance_task)
        total_seasonal_pollinator_abundance_path_lookup[season_id] = (
            f_reg[total_seasonal_pollinator_abundance_id])

    if farm_vector is None:
        LOGGER.info("All done, no farm polygon to process!")
        task_graph.join()
        for path in temp_file_set:
            try:
                os.remove(path)
            except OSError:
                pass  # we might have deleted it on another task
        return

    LOGGER.info("Calculating farm yields")
    target_farm_path = os.path.join(
        output_dir, '%s%s.shp' % (
            _TARGET_AGGREGATE_FARM_VECTOR_FILE_PATTERN, file_suffix))
    farm_fid_field = str(uuid.uuid4())[-8:-1]
    create_target_farm_task = task_graph.add_task(
        target=_create_fid_vector_copy,
        args=(projected_farm_vector_path, farm_fid_field, target_farm_path),
        target_path_list=[target_farm_path],
        dependent_task_list=[reproject_farm_task])

    wild_pollinator_activity = None
    foraging_activity_index = None
    farm_yield_task_list = []
    for season_id in season_to_header:
        LOGGER.info("Calculating total pollinator abundance")
        seasonal_pollinator_abundance_id = (
            _SEASONAL_POLLINATOR_ABUNDANCE_FILE_PATTERN % (
                species_id, season_id))
        f_reg[seasonal_pollinator_abundance_id]

        LOGGER.info("Rasterizing half saturation for season %s", season_id)
        half_saturation_file_path = os.path.join(
            intermediate_output_dir,
            _HALF_SATURATION_SEASON_FILE_PATTERN % season_id + (
                '%s.tif' % file_suffix))

        rasterize_half_saturation_task = task_graph.add_task(
            target=_rasterize_half_saturation,
            args=(
                args['landcover_raster_path'], season_id, target_farm_path,
                half_saturation_file_path),
            dependent_task_list=[create_target_farm_task])

        LOGGER.info("Rasterizing managed farm pollinators for season %s")
        # rasterize farm managed pollinators on landscape first
        managed_bees_raster_path = os.path.join(
            intermediate_output_dir, "%s%s.tif" % (
                _MANAGED_BEES_RASTER_FILE_PATTERN % season_id, file_suffix))
        rasterize_farm_pollinators_task = task_graph.add_task(
            target=_rasterize_managed_farm_pollinators,
            args=(
                args['landcover_raster_path'], target_farm_path,
                managed_bees_raster_path))

        LOGGER.info("Calculating farm pollinators for season %s", season_id)
        wild_pollinator_activity = [
            guild_table[species_id][season_to_header[season_id]['guild']]
            for species_id in sorted(guild_table)]

        wild_pollinator_abundance_band_paths = [
            (f_reg[_POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id], 1)
            for species_id in sorted(guild_table)]
        farm_pollinators_path = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                _FARM_POLLINATORS_FILE_PATTERN % season_id, file_suffix))

        farm_pollinators_task = task_graph.add_task(
            target=pygeoprocessing.raster_calculator,
            args=(
                [(managed_bees_raster_path, 1)] +
                wild_pollinator_abundance_band_paths, _FarmPollinatorsOp(
                    wild_pollinator_activity),
                farm_pollinators_path, gdal.GDT_Float32, _INDEX_NODATA),
            kwargs={'calc_raster_stats': False},
            seasonaldent_task_list=final_pollinator_abundance_task_list + [
                rasterize_farm_pollinators_task])

        species_equal_task_list = []
        for species_id in guild_table:
            max_pollinated_species_path = os.path.join(
                intermediate_output_dir, "%s%s.tif" % (
                    _MAX_POLLINATED_SPECIES_FILE_PATTERN % species_id,
                    file_suffix))
            foraging_activity_index = (
                guild_table[species_id][season_to_header[season_id]['guild']])

            species_equal_task = task_graph.add_task(
                target=pygeoprocessing.raster_calculator,
                args=(
                    [(farm_pollinators_path, 1),
                     (f_reg[_POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id],
                      1)],
                    _EqualOp(foraging_activity_index),
                    max_pollinated_species_path, gdal.GDT_Byte,
                    _INDEX_NODATA),
                kwargs={'calc_raster_stats': False},
                seasonaldent_task_list=final_pollinator_abundance_task_list + [
                    farm_pollinators_task])
            species_equal_task_list.append(species_equal_task)

        LOGGER.info("Calculating farm yield.")

        pollinator_yield_path = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                _SEASONAL_POLLINATOR_YIELD_FILE_PATTERN % season_id,
                file_suffix))

        farm_yield_task = task_graph.add_task(
            target=pygeoprocessing.raster_calculator,
            args=(
                [(half_saturation_file_path, 1), (farm_pollinators_path, 1)],
                _FarmYieldOp(), pollinator_yield_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            kwargs={'calc_raster_stats': True},
            dependent_task_list=[
                rasterize_half_saturation_task, farm_pollinators_task])
        farm_yield_task_list.append(farm_yield_task)

    # add the yield pollinators, shouldn't be conflicts since we don't have
    # overlapping farms

    seasonal_pollinator_yield_path_band_list = [
        (os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                _SEASONAL_POLLINATOR_YIELD_FILE_PATTERN % season_id,
                file_suffix)), 1)
        for season_id in season_to_header]
    total_pollinator_yield_path = os.path.join(
        output_dir, '%s%s.tif' % (
            _TOTAL_POLLINATOR_YIELD_FILE_PATTERN, file_suffix))

    _combine_yields_task = task_graph.add_task(
        target=pygeoprocessing.raster_calculator,
        args=(
            seasonal_pollinator_yield_path_band_list, _CombineYieldsOp(),
            total_pollinator_yield_path, gdal.GDT_Float32, _INDEX_NODATA),
        kwargs={'calc_raster_stats': True},
        dependent_task_list=farm_yield_task_list)

    _combine_yields_task.join()

    farm_stats = pygeoprocessing.zonal_statistics(
        (total_pollinator_yield_path, 1), target_farm_path, farm_fid_field)

    # after much debugging, I could only get the first feature to write.
    # closing and re-opening the vector seemed to reset it enough to work.
    farm_vector.SyncToDisk()
    farm_layer = None
    farm_vector = None
    farm_vector = ogr.Open(target_farm_path, 1)
    farm_layer = farm_vector.GetLayer()
    for feature in farm_layer:
        fid = feature.GetField(farm_fid_field)
        pollinator_dependence = feature.GetField(
            _CROP_POLLINATOR_DEPENDENCE_FIELD)
        pollinator_dependent_yield = float(
            farm_stats[fid]['sum'] / farm_stats[fid]['count'] *
            pollinator_dependence)
        feature.SetField(
            _POLLINATOR_FARM_YIELD_FIELD_ID, pollinator_dependent_yield)
        total_yield = (1 - pollinator_dependence) + pollinator_dependent_yield
        feature.SetField(
            _TOTAL_FARM_YIELD_FIELD_ID, total_yield)
        farm_layer.SetFeature(feature)
        feature = None

    farm_layer.DeleteField(
        farm_layer.GetLayerDefn().GetFieldIndex(farm_fid_field))
    for path in temp_file_set:
        try:
            os.remove(path)
        except OSError:
            pass
            # it's possible this was removed in an earlier run


def _normalized_convolve_2d(
        signal_path_band, kernel_path_band, target_raster_path,
        target_datatype, target_nodata, workspace_dir):
    """Perform a normalized 2D convolution.

    Convolves the raster in `kernel_path_band` over `signal_path_band` and
    divides the result by a convolution of the kernel over a non-nodata mask
    of the signal.

    Parameters:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index).
        target_path (string): filepath to target raster that's the convolution
            of signal with kernel.  Output will be a single band raster of
            same size and projection as `signal_path_band`. Any nodata pixels
            that align with `signal_path_band` will be set to nodata.
        target_datatype (GDAL type): a GDAL raster type to set the output
            raster type to, as well as the type to calculate the convolution
            in.
        target_nodata (int/float): target_path's nodata value.
        workspace_dir (string): path to a directory that exists where
            threadsafe non-colliding temporary files can be written.

    Returns:
        None
    """
    with tempfile.NamedTemporaryFile(
            prefix='mask_path_', dir=workspace_dir,
            delete=False, suffix='.tif') as mask_file:
        mask_path = mask_file.name

    with tempfile.NamedTemporaryFile(
            prefix='base_convolve_path_', dir=workspace_dir,
            delete=False, suffix='.tif') as base_convolve_file:
        base_convolve_path = base_convolve_file.name

    with tempfile.NamedTemporaryFile(
            prefix='mask_convolve_path_', dir=workspace_dir,
            delete=False, suffix='.tif') as mask_convolve_file:
        mask_convolve_path = mask_convolve_file.name

    signal_info = pygeoprocessing.get_raster_info(signal_path_band[0])
    signal_nodata = signal_info['nodata'][signal_path_band[1]-1]
    pygeoprocessing.raster_calculator(
        [signal_path_band], lambda x: x != signal_nodata,
        mask_path, gdal.GDT_Byte, None,
        calc_raster_stats=False)

    pygeoprocessing.convolve_2d(
        signal_path_band, kernel_path_band, base_convolve_path,
        target_datatype=target_datatype,
        target_nodata=target_nodata)
    pygeoprocessing.convolve_2d(
        (mask_path, 1), kernel_path_band, mask_convolve_path,
        target_datatype=target_datatype)

    def _divide_op(base_convolve, normalization):
        """Divide base_convolve by normalization + handle nodata/div by 0."""
        result = numpy.empty(base_convolve.shape, dtype=numpy.float32)
        valid_mask = (base_convolve != target_nodata)
        nonzero_mask = normalization != 0.0
        result[:] = target_nodata
        result[valid_mask] = base_convolve[valid_mask]
        result[valid_mask & nonzero_mask] /= normalization[
            valid_mask & nonzero_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(base_convolve_path, 1), (mask_convolve_path, 1)], _divide_op,
        target_raster_path, target_datatype, target_nodata,
        calc_raster_stats=False)

    for path in [mask_path, base_convolve_path, mask_convolve_path]:
        os.remove(path)


def _add_fid_field(base_vector_path, target_vector_path, fid_id):
    """Make a copy of base vector and an FID field to identify features.

    Parameters:
        base_vector_path (string): path to a single layer vector
        target_vector_path (string): path to desired output vector, the
            directory to the file must exist.
        fid_id (string): field ID to add to base vector.  Must not already
            be defined in base_vector_path.  Raises a ValueError if so.

    Returns:
        None
    """
    esri_driver = ogr.GetDriverByName("ESRI Shapefile")

    base_vector = ogr.Open(base_vector_path)
    base_layer = base_vector.GetLayer()
    base_defn = base_layer.GetLayerDefn()

    if base_defn.GetFieldIndex(fid_id) != -1:
        raise ValueError(
            "Tried to add a new field %s, but is already defined in %s." % (
                fid_id, base_vector_path))
    if os.path.exists(target_vector_path):
        os.remove(target_vector_path)
    target_vector = esri_driver.CopyDataSource(
        base_vector, target_vector_path)
    target_layer = target_vector.GetLayer()
    target_layer.CreateField(ogr.FieldDefn(fid_id, ogr.OFTInteger))
    for feature in target_layer:
        feature.SetField(fid_id, feature.GetFID())
        target_layer.SetFeature(feature)
    target_layer = None
    target_vector.SyncToDisk()
    target_vector = None


def _rasterize_half_saturation(
        base_raster_path, season_id, target_farm_path,
        half_saturation_file_path):
    #TODO: document
    farm_vector = ogr.Open(target_farm_path)
    farm_layer = farm_vector.GetLayer()

    pygeoprocessing.new_raster_from_base(
        base_raster_path, half_saturation_file_path,
        gdal.GDT_Float32, [_INDEX_NODATA],
        fill_value_list=[_INDEX_NODATA])
    farm_layer.SetAttributeFilter(str("season='%s'" % season_id))
    half_saturation_raster = gdal.Open(
        half_saturation_file_path, gdal.GA_Update)
    gdal.RasterizeLayer(
        half_saturation_raster, [1], farm_layer,
        options=['ATTRIBUTE=%s' % _HALF_SATURATION_FARM_HEADER])
    gdal.Dataset.__swig_destroy__(half_saturation_raster)
    half_saturation_raster = None
    farm_layer = None
    farm_vector = None


def _rasterize_managed_farm_pollinators(
        base_raster_path, target_farm_path, managed_bees_raster_path):
    #TODO: document
    farm_vector = ogr.Open(target_farm_path)
    farm_layer = farm_vector.GetLayer()

    pygeoprocessing.new_raster_from_base(
        base_raster_path, managed_bees_raster_path,
        gdal.GDT_Float32, [_INDEX_NODATA])

    managed_raster = gdal.Open(managed_bees_raster_path, gdal.GA_Update)
    gdal.RasterizeLayer(
        managed_raster, [1], farm_layer, options=['ATTRIBUTE=p_managed'])
    gdal.Dataset.__swig_destroy__(managed_raster)
    del managed_raster
    farm_layer = None
    farm_vector = None


def _create_fid_vector_copy(
        base_vector_path, fid_field, target_vector_path):
    """Create a copy of `base_vector_path` and add FID field to it."""
    # make a random string to use as an FID field.  The chances of this
    # colliding with an existing field name are so astronomical we aren't
    # going to test if that happens.
    esri_driver = ogr.GetDriverByName("ESRI Shapefile")
    base_vector = ogr.Open(base_vector_path)
    base_layer = base_vector.GetLayer()
    base_defn = base_layer.GetLayerDefn()

    if base_defn.GetFieldIndex(fid_field) != -1:
        raise ValueError(
            "Tried to add a new field %s, but is already defined in %s." % (
                fid_field, base_vector_path))
    if os.path.exists(target_vector_path):
        os.remove(target_vector_path)
    target_vector = esri_driver.CopyDataSource(
        base_vector, target_vector_path)
    target_layer = target_vector.GetLayer()
    target_layer.CreateField(ogr.FieldDefn(fid_field, ogr.OFTInteger))
    for feature in target_layer:
        feature.SetField(fid_field, feature.GetFID())
        target_layer.SetFeature(feature)

    target_layer.CreateField(ogr.FieldDefn(
        _POLLINATOR_FARM_YIELD_FIELD_ID, ogr.OFTReal))
    target_layer.CreateField(ogr.FieldDefn(
        _TOTAL_FARM_YIELD_FIELD_ID, ogr.OFTReal))

    target_layer = None
    target_vector.SyncToDisk()
    target_vector = None


class _AddRasterOp(object):
    """Closure for adding rasters together."""

    def __init__(self, raster_stack_nodata):
        self.raster_stack_nodata = raster_stack_nodata

    def __call__(self, *raster_stack):
        result = numpy.empty(raster_stack[0].shape)
        valid_mask = raster_stack[0] != self.raster_stack_nodata
        result[:] = self.raster_stack_nodata
        result[valid_mask] = (
            numpy.sum(numpy.stack(raster_stack, axis=2)[valid_mask], axis=1))
        return result


class _HabitatSuitabilityIndexOp(object):
    """Closure for nesting suitability raster calculator."""

    def __init__(self, species_nesting_suitability_index):
        self.species_nesting_suitability_index = (
            species_nesting_suitability_index)

    def __call__(self, *nesting_suitability_index):
        result = numpy.empty(
            nesting_suitability_index[0].shape, dtype=numpy.float32)
        valid_mask = nesting_suitability_index[0] != _INDEX_NODATA
        result[:] = _INDEX_NODATA
        # the species' nesting suitability index is the maximum value
        # of all nesting substrates multiplied by the species'
        # suitability index for that substrate
        result[valid_mask] = numpy.max(
            [nsi[valid_mask] * snsi for nsi, snsi in zip(
                nesting_suitability_index,
                self.species_nesting_suitability_index)],
            axis=0)
        return result


class _SpeciesFloralAbudanceOp(object):
    """Closure for species floral abundance raster calculator."""

    def __init__(self, species_foraging_activity_per_season):
        self.species_foraging_activity_per_season = (
            species_foraging_activity_per_season)

    def __call__(self, *floral_resources_index_array):
        """Calculate species floral abudance."""
        result = numpy.empty(
            floral_resources_index_array[0].shape, dtype=numpy.float32)
        result[:] = _INDEX_NODATA
        valid_mask = floral_resources_index_array[0] != _INDEX_NODATA
        result[valid_mask] = numpy.sum(
            [fri[valid_mask] * sfa for fri, sfa in zip(
                floral_resources_index_array,
                self.species_foraging_activity_per_season)], axis=0)
        return result


class _PollinatorSupplyOp(object):
    """Closure for pollinator supply raster calculator."""

    def __init__(self, species_abundance):
        self.species_abundance = species_abundance

    def __call__(self, accessible_floral_resources, species_nesting_index):
        """Supply is floral resources * nesting index * abundance."""
        result = numpy.empty(
            accessible_floral_resources.shape, dtype=numpy.float32)
        result[:] = _INDEX_NODATA
        valid_mask = (species_nesting_index != _INDEX_NODATA)
        result[valid_mask] = (
            accessible_floral_resources[valid_mask] *
            species_nesting_index[valid_mask] * self.species_abundance)
        return result


class _PollinatorAbudanceOp(object):
    """Closure for pollinator abundance operation."""

    def __init__(
            self, species_seasonal_foraging_activity, raw_abundance_nodata):
        self.raw_abundance_nodata = raw_abundance_nodata
        self.species_seasonal_foraging_activity = (
            species_seasonal_foraging_activity)

    def __call__(self, raw_abundance, floral_resources):
        """Multiply raw_abundance by floral_resources and skip nodata."""
        result = numpy.empty(raw_abundance.shape, dtype=numpy.float32)
        result[:] = _INDEX_NODATA
        valid_mask = (
            (self.raw_abundance_nodata != raw_abundance) &
            (floral_resources != _INDEX_NODATA))
        result[valid_mask] = (
            self.species_seasonal_foraging_activity *
            raw_abundance[valid_mask] * floral_resources[valid_mask])
        return result


class _FarmPollinatorsOp(object):
    """Closure for calculating farm pollinators."""

    def __init__(self, wild_pollinator_activity):
        self.wild_pollinator_activity = wild_pollinator_activity

    def __call__(
            self, managed_pollinator_abundance, *wild_pollinator_abundance):
        """Calculate the max of all pollinators.

        Wild pollinators need to be scaled by their seasonal foraging
        activity index included in the closure.
        """
        result = numpy.empty(
            managed_pollinator_abundance.shape, dtype=numpy.float32)
        valid_mask = wild_pollinator_abundance[0] != _INDEX_NODATA
        result[:] = _INDEX_NODATA
        result[valid_mask] = numpy.clip(
            (managed_pollinator_abundance[valid_mask] +
             numpy.sum([
                 activity * abundance[valid_mask]
                 for abundance, activity in zip(
                     wild_pollinator_abundance,
                     self.wild_pollinator_activity)], axis=0)), 0, 1)
        return result


class _EqualOp(object):
    """Closure for determining max species in a pixel."""

    def __init__(self, foraging_activity_index):
        self.foraging_activity_index = foraging_activity_index

    def __call__(
            self, farm_pollinator_index, species_pollinator_abudance_index):
        """Return 1 if FP == SP*FA."""
        result = numpy.empty(
            farm_pollinator_index.shape, dtype=numpy.int8)
        result[:] = _INDEX_NODATA
        valid_mask = farm_pollinator_index != _INDEX_NODATA
        result[valid_mask] = farm_pollinator_index[valid_mask] == (
            species_pollinator_abudance_index[valid_mask] *
            self.foraging_activity_index)
        return result


class _FarmYieldOp(object):
    """Farm yield closure."""

    def __init__(self):
        pass

    def __call__(self, half_sat, farm_pollinators):
        result = numpy.empty(half_sat.shape, dtype=numpy.float32)
        result[:] = _INDEX_NODATA
        valid_mask = (
            farm_pollinators != _INDEX_NODATA) & (
                half_sat != _INDEX_NODATA)
        # the following is a tunable half-saturation half-sigmoid
        # FP(x,j) = (1-h(x,j))h(x,j) / ((1-2FP(x,j))+FP(x,j))
        result[valid_mask] = (
            farm_pollinators[valid_mask] * (1 - half_sat[valid_mask]) / (
                half_sat[valid_mask] * (
                    1 - 2 * farm_pollinators[valid_mask]) +
                farm_pollinators[valid_mask]))
        return result


class _CombineYieldsOp(object):
    """Combine all farm yields."""

    def __init__(self):
        pass

    def __call__(self, *pollinator_yields):
        """Set output to defined pixel in the stack of pollinator_yields."""
        result = numpy.empty(pollinator_yields[0].shape, dtype=numpy.float32)
        result[:] = _INDEX_NODATA
        for pollinator_yield in pollinator_yields:
            valid_mask = pollinator_yield != _INDEX_NODATA
            result[valid_mask] = pollinator_yield[valid_mask]
        return result
