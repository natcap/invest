"""Pollinator service model for InVEST."""
import itertools
import collections
import re
import os
import logging

from osgeo import gdal
import pygeoprocessing
import numpy

from . import utils

LOGGER = logging.getLogger('natcap.invest.pollination')


_OUTPUT_BASE_FILES = {
    }

_INTERMEDIATE_BASE_FILES = {
    }

_TMP_BASE_FILES = {
    }

_INDEX_NODATA = -1.0

_SPECIES_ALPHA_KERNEL_FILE_PATTERN = r'alpha_kernel_%s'
_ACCESSABLE_FLORAL_RESOURCES_FILE_PATTERN = r'accessable_floral_resources_%s'
_LOCAL_POLLINATOR_SUPPLY_FILE_PATTERN = r'local_pollinator_supply_%s_index'
_POLLINATOR_ABUNDANCE_FILE_PATTERN = r'pollinator_abundance_%s_index'
_LOCAL_FLORAL_RESOURCE_AVAILABILITY_FILE_PATTERN = (
    r'local_floral_resource_availability_%s_index')

_NESTING_SUITABILITY_SPECIES_PATTERN = r'nesting_suitability_%s_index'

# These patterns are expected in the biophysical table
_NESTING_SUBSTRATE_PATTERN = 'nesting_([^_]+)_availability_index'
_FLORAL_RESOURCES_AVAILABLE_PATTERN = 'floral_resources_([^_]+)_index'
_EXPECTED_BIOPHYSICAL_HEADERS = [
    'lucode', _NESTING_SUBSTRATE_PATTERN, _FLORAL_RESOURCES_AVAILABLE_PATTERN]

# These are patterns expected in the guilds table
_NESTING_SUITABILITY_PATTERN = 'nesting_suitability_([^_]+)_index'
_FORAGING_ACTIVITY_PATTERN = 'foraging_activity_([^_]+)_index'
_EXPECTED_GUILD_HEADERS = [
    'species', _NESTING_SUITABILITY_PATTERN, _FORAGING_ACTIVITY_PATTERN,
    'alpha']


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
            the bee species to analyize in this model run.  Table headers
            must include:
                * 'species': a bee species whose column string names will
                    be refered to in other tables and the model will output
                    analyses per species.
                * 'nesting_cavity' and 'nesting_ground': a number in the range
                    [0.0, 1.0] indicating the suitability of the given species
                    to nest in cavity or ground types.
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
        args['farm_vector_path'] (string): path to a single layer polygon
            shapefile representing farms. The layer will have at least the
            following fields:

            * season (string): season in which the farm needs pollination
            * half_sat (float): a real in the range [0.0, 1.0] representing
                the proportion of wild pollinators to achieve a 50% yield
                of that crop.
            * p_wild_dep (float): a number in the range [0.0, 1.0]
                representing the proportion of yield dependant on pollinators.
            * p_managed (float): proportion of pollinators that come from
                non-native/managed hives.

    Returns:
        None
    """
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories(
        [output_dir, intermediate_output_dir])

    f_reg = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir),
         (_TMP_BASE_FILES, output_dir)], file_suffix)
    temp_file_set = set()  # to keep track of temporary files to delete

    guild_table = utils.build_lookup_from_csv(
        args['guild_table_path'], 'species', to_lower=True,
        numerical_cast=True)

    LOGGER.debug('Checking to make sure guild table has all expected headers')
    guild_headers = guild_table.itervalues().next().keys()

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

    LOGGER.debug(
        'TODO: grab the seasons from guild table and compare against '
        'habitat nesting suitability seasons.  If different, then report '
        'otherwise make an index mapping season to an ID and report it in '
        'an output table and log')

    landcover_biophysical_table = utils.build_lookup_from_csv(
        args['landcover_biophysical_table_path'], 'lucode', to_lower=True,
        numerical_cast=True)
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    LOGGER.debug(
        'TODO: make sure landcover biophysical table has all expected '
        'headers')
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

    for header in biophysical_table_headers:
        match = re.match(_FLORAL_RESOURCES_AVAILABLE_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]['biophysical'] = match.group()
        match = re.match(_NESTING_SUBSTRATE_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]['biophysical'] = match.group()

    LOGGER.debug(substrate_to_header)
    for header_type, lookup_table in itertools.chain(
            substrate_to_header.iteritems(), season_to_header.iteritems()):
        if len(lookup_table) != 2:
            raise ValueError(
                "Expected both a biophysical and guild entry for '%s' but "
                "instead found only %s. Ensure there are corresponding "
                "entries of '%s' in both the guilds and biophysical "
                "table." % (header_type, lookup_table, header_type))

    for nesting_substrate in substrate_to_header:
        LOGGER.info(
            "Mapping landcover to nesting substrate %s", nesting_substrate)
        nesting_id = substrate_to_header[nesting_substrate]['biophysical']
        landcover_to_nesting_suitability_table = dict([
            (lucode, landcover_biophysical_table[lucode][nesting_id]) for
            lucode in landcover_biophysical_table])

        f_reg[nesting_id] = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (nesting_id, file_suffix))

        pygeoprocessing.reclassify_raster(
            (args['landcover_raster_path'], 1),
            landcover_to_nesting_suitability_table, f_reg[nesting_id],
            gdal.GDT_Float32, _INDEX_NODATA, exception_flag='values_required')

    nesting_substrate_path_list = [
        f_reg[substrate_to_header[nesting_substrate]['biophysical']]
        for nesting_substrate in sorted(substrate_to_header)]

    species_nesting_suitability_index = None
    for species_id in guild_table.iterkeys():
        LOGGER.info("Calculate species nesting index for %s", species_id)
        species_nesting_suitability_index = numpy.array([
            guild_table[species_id][substrate_to_header[substrate]['guild']]
            for substrate in sorted(substrate_to_header)])
        species_nesting_suitability_id = (
            _NESTING_SUITABILITY_SPECIES_PATTERN % species_id)

        def _habitat_suitability_index_op(*nesting_suitability_index):
            """Calculate habitat suitability per species."""
            result = numpy.empty(
                nesting_suitability_index[0].shape, dtype=numpy.float32)
            valid_mask = nesting_suitability_index[0] != _INDEX_NODATA
            result[:] = _INDEX_NODATA
            # the species' nesting suitability index is the maximum value of
            # all nesting substrates multiplied by the species' suitability
            # index for that substrate
            result[valid_mask] = numpy.max(
                [nsi[valid_mask] * snsi for nsi, snsi in zip(
                    nesting_suitability_index,
                    species_nesting_suitability_index)],
                axis=0)
            return result

        f_reg[species_nesting_suitability_id] = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                species_nesting_suitability_id, file_suffix))
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in nesting_substrate_path_list],
            _habitat_suitability_index_op,
            f_reg[species_nesting_suitability_id], gdal.GDT_Float32,
            _INDEX_NODATA, calc_raster_stats=False)

    for season_id in season_to_header:
        LOGGER.info(
            "Mapping landcover to available floral resources for season %s",
            season_id)
        relative_floral_abudance_id = (
            season_to_header[season_id]['biophysical'])
        f_reg[relative_floral_abudance_id] = os.path.join(
            intermediate_output_dir,
            "%s%s.tif" % (relative_floral_abudance_id, file_suffix))

        landcover_to_floral_abudance_table = dict([
            (lucode, landcover_biophysical_table[lucode][
                relative_floral_abudance_id]) for lucode in
            landcover_biophysical_table])

        pygeoprocessing.reclassify_raster(
            (args['landcover_raster_path'], 1),
            landcover_to_floral_abudance_table,
            f_reg[relative_floral_abudance_id], gdal.GDT_Float32,
            _INDEX_NODATA, exception_flag='values_required')

    floral_resources_path_list = [
        f_reg[season_to_header[season_id]['biophysical']]
        for season_id in sorted(season_to_header)]

    LOGGER.debug(
        "TODO: consider making species season floral weight relative "
        "rather than absolute.")
    species_foraging_activity_per_season = None
    for species_id in guild_table:
        LOGGER.info(
            "Making local floral resources map for species %s", species_id)
        species_foraging_activity_per_season = numpy.array([
            guild_table[species_id][
                season_to_header[season_id]['guild']]
            for season_id in sorted(season_to_header)])
        local_floral_resource_availability_id = (
            _LOCAL_FLORAL_RESOURCE_AVAILABILITY_FILE_PATTERN % species_id)
        f_reg[local_floral_resource_availability_id] = os.path.join(
            intermediate_output_dir, "%s%s.tif" % (
                local_floral_resource_availability_id, file_suffix))

        def _species_floral_abudance_op(*floral_resources_index_array):
            """Calculate species floral abudance."""
            result = numpy.empty(
                floral_resources_index_array[0].shape, dtype=numpy.float32)
            result[:] = _INDEX_NODATA
            valid_mask = floral_resources_index_array[0] != _INDEX_NODATA
            result[valid_mask] = numpy.sum(
                [fri[valid_mask] * sfa for fri, sfa in zip(
                    floral_resources_index_array,
                    species_foraging_activity_per_season)], axis=0)
            return result

        pygeoprocessing.raster_calculator(
            [(path, 1) for path in floral_resources_path_list],
            _species_floral_abudance_op,
            f_reg[local_floral_resource_availability_id], gdal.GDT_Float32,
            _INDEX_NODATA, calc_raster_stats=False)

        LOGGER.warn("TODO: consider case where cell size is not square.")
        alpha = (
            guild_table[species_id]['alpha'] /
            float(lulc_raster_info['mean_pixel_size']))
        species_file_kernel_id = (
            _SPECIES_ALPHA_KERNEL_FILE_PATTERN % species_id)
        f_reg[species_file_kernel_id] = os.path.join(
            output_dir, species_file_kernel_id + '%s.tif' % file_suffix)
        utils.exponential_decay_kernel_raster(
            alpha, f_reg[species_file_kernel_id])

        accessable_floral_resouces_id = (
            _ACCESSABLE_FLORAL_RESOURCES_FILE_PATTERN % species_id)
        f_reg[accessable_floral_resouces_id] = os.path.join(
            intermediate_output_dir, accessable_floral_resouces_id +
            '%s.tif' % file_suffix)
        temp_file_set.add(f_reg[species_file_kernel_id])
        LOGGER.info(
            "Calculating available floral resources for %s", species_id)
        pygeoprocessing.convolve_2d(
            (f_reg[local_floral_resource_availability_id], 1),
            (f_reg[species_file_kernel_id], 1),
            f_reg[accessable_floral_resouces_id],
            target_datatype=gdal.GDT_Float32)
        LOGGER.info("Calculating local pollinator supply for %s", species_id)
        pollinator_supply_id = (
            _LOCAL_POLLINATOR_SUPPLY_FILE_PATTERN % species_id)
        f_reg[pollinator_supply_id] = os.path.join(
            intermediate_output_dir,
            pollinator_supply_id + "%s.tif" % file_suffix)

        def _pollinator_supply_op(
                accessable_floral_resources, species_nesting_index):
            """Multiply accesable floral resources by nesting index."""
            result = numpy.empty(
                accessable_floral_resources.shape, dtype=numpy.float32)
            result[:] = _INDEX_NODATA
            valid_mask = (species_nesting_index != _INDEX_NODATA)
            result[valid_mask] = (
                accessable_floral_resources[valid_mask] *
                species_nesting_index[valid_mask])
            return result

        species_nesting_suitability_id = (
            _NESTING_SUITABILITY_SPECIES_PATTERN % species_id)
        pygeoprocessing.raster_calculator(
            [(f_reg[accessable_floral_resouces_id], 1),
             (f_reg[species_nesting_suitability_id], 1)],
            _pollinator_supply_op, f_reg[pollinator_supply_id],
            gdal.GDT_Float32, _INDEX_NODATA, calc_raster_stats=False)

        LOGGER.info("Calculating pollinator abundance for %s")
        pollinator_abudanance_id = (
            _POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id)
        f_reg[pollinator_abudanance_id] = os.path.join(
            output_dir,
            pollinator_abudanance_id + "%s.tif" % file_suffix)
        pygeoprocessing.convolve_2d(
            (f_reg[pollinator_supply_id], 1),
            (f_reg[species_file_kernel_id], 1),
            f_reg[pollinator_abudanance_id],
            target_datatype=gdal.GDT_Float32)

    LOGGER.info("Calculating farm polinator index.")
    # rasterize farm managed pollinators on landscape first
    # rasterize each farm as a season
    # raster calcualtor pass managed polinators, farm/season masks, pollinator abundance per season
    # FP(f,x)=MP(f)+sSPA(x,s)FA(s,j(f))

    #pygeoprocessing.zonal_statistics(
    #    base_raster_path_band, aggregating_vector_path,
    #    aggregate_field_name, aggregate_layer_name=None,
    #    ignore_nodata=True, all_touched=False, polygons_might_overlap=True)

    for path in temp_file_set:
        os.remove(path)
