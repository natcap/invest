"""Pollinator service model for InVEST."""
import tempfile
import itertools
import collections
import re
import os
import logging

from osgeo import gdal
from osgeo import ogr
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

_MANAGED_BEES_RASTER_FILE_PATTERN = r'managed_bees_%s'
_SPECIES_ALPHA_KERNEL_FILE_PATTERN = r'alpha_kernel_%s'
_ACCESSABLE_FLORAL_RESOURCES_FILE_PATTERN = r'accessable_floral_resources_%s'
_LOCAL_POLLINATOR_SUPPLY_FILE_PATTERN = r'local_pollinator_supply_%s_index'
_POLLINATOR_ABUNDANCE_FILE_PATTERN = r'pollinator_abundance_%s_index'
_RAW_POLLINATOR_ABUNDANCE_FILE_PATTERN = r'raw_pollinator_abundance_%s_index'
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

_HALF_SATURATION_SEASON_FILE_PATTERN = 'half_saturation_%s'
_FARM_POLLINATORS_FILE_PATTERN = 'farm_pollinators_%s'
_FARM_FLORAL_RESOURCES_PATTERN = 'fr_([^_]+)'
_FARM_NESTING_SUBSTRATE_PATTERN = 'n_([^_]+)'
_HALF_SATURATION_FARM_HEADER = 'half_sat'
_EXPECTED_FARM_HEADERS = [
    'season', 'crop_type', _HALF_SATURATION_FARM_HEADER, 'p_managed',
    _FARM_FLORAL_RESOURCES_PATTERN, _FARM_NESTING_SUBSTRATE_PATTERN]

_POLLINATOR_YIELD_FILE_PATTERN = 'pollinator_yield_%s'

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
            * f_[season] (float): any number of fields that match this pattern
                such that `season` also matches the season headers in the
                biophysical and guild table.  Any areas that overlap the
                landcover map will replace seasonal floral resources with
                this value.  Ranges from 0..1.
            * n_[substrate] (float): any number of fields that match this
                pattern such that `substrate` also matches the nesting
                substrate headers in the biophysical and guild table.  Any
                areas that overlap the landcover map will replace nesting
                substrate suitability with this value.  Ranges from 0..1.

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

    LOGGER.info('Checking that farm polygon has expected headers')
    farm_vector = ogr.Open(args['farm_vector_path'])
    if farm_vector.GetLayerCount() != 1:
        raise ValueError(
            "Farm polygon at %s has %d layers when expecting only 1." % (
                args['farm_vector_path'], farm_vector.GetLayerCount()))
    farm_layer = farm_vector.GetLayer()
    if farm_layer.GetGeomType() not in [
            ogr.wkbPolygon, ogr.wkbMultiPolygon]:
        raise ValueError(
            "Farm layer not a polygon type, instead type %s" % (
                farm_layer.GetGeomType()))
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

    for header in biophysical_table_headers:
        match = re.match(_FLORAL_RESOURCES_AVAILABLE_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]['biophysical'] = match.group()
        match = re.match(_NESTING_SUBSTRATE_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]['biophysical'] = match.group()

    for header in farm_headers:
        match = re.match(_FARM_FLORAL_RESOURCES_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]['farm'] = match.group()
        match = re.match(_FARM_NESTING_SUBSTRATE_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]['farm'] = match.group()

    LOGGER.debug(substrate_to_header)
    for table_type, lookup_table in itertools.chain(
            season_to_header.iteritems(), substrate_to_header.iteritems()):
        if len(lookup_table) != 3:
            raise ValueError(
                "Expected a biophysical, guild, and farm entry for '%s' but "
                "instead found only %s. Ensure there are corresponding "
                "entries of '%s' in both the guilds, biophysical "
                "table, and farm fields." % (
                    table_type, lookup_table, table_type))

    farm_season_set = set()
    for farm_feature in farm_layer:
        farm_season_set.add(farm_feature.GetField('season'))

    if len(farm_season_set.difference(season_to_header)) > 0:
        raise ValueError(
            "Found seasons in farm polygon that were not specified in the"
            "biophysical table: %s.  Expected only these: %s" % (
                farm_season_set.difference(season_to_header),
                season_to_header))

    season_to_rasterize_id = [
        (season, season_id) for season_id, season in enumerate(
            sorted(farm_season_set))]

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

        LOGGER.info(
            "Overriding landcover nesting substrates where a farm polygon is "
            "available.")
        nesting_substrate_raster = gdal.Open(
            f_reg[nesting_id], gdal.GA_Update)
        farm_vector = ogr.Open(args['farm_vector_path'])
        farm_layer = farm_vector.GetLayer()
        gdal.RasterizeLayer(
            nesting_substrate_raster, [1], farm_layer,
            options=['ATTRIBUTE=%s' % (
                _FARM_NESTING_SUBSTRATE_PATTERN.replace(
                    '([^_]+)', nesting_substrate))])
        del farm_layer
        del farm_vector
        gdal.Dataset.__swig_destroy__(nesting_substrate_raster)
        del nesting_substrate_raster

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
        relative_floral_resources_id = (
            season_to_header[season_id]['biophysical'])
        f_reg[relative_floral_resources_id] = os.path.join(
            intermediate_output_dir,
            "%s%s.tif" % (relative_floral_resources_id, file_suffix))

        landcover_to_floral_resources_table = dict([
            (lucode, landcover_biophysical_table[lucode][
                relative_floral_resources_id]) for lucode in
            landcover_biophysical_table])

        pygeoprocessing.reclassify_raster(
            (args['landcover_raster_path'], 1),
            landcover_to_floral_resources_table,
            f_reg[relative_floral_resources_id], gdal.GDT_Float32,
            _INDEX_NODATA, exception_flag='values_required')

        LOGGER.info(
            "Overriding landcover floral resources where a farm polygon is "
            "available.")
        floral_resources_raster = gdal.Open(
            f_reg[relative_floral_resources_id], gdal.GA_Update)
        farm_vector = ogr.Open(args['farm_vector_path'])
        farm_layer = farm_vector.GetLayer()
        gdal.RasterizeLayer(
            floral_resources_raster, [1], farm_layer,
            options=['ATTRIBUTE=%s' % (
                _FARM_FLORAL_RESOURCES_PATTERN.replace(
                    '([^_]+)', season_id))])
        del farm_layer
        del farm_vector
        gdal.Dataset.__swig_destroy__(floral_resources_raster)
        del floral_resources_raster

    floral_resources_path_list = [
        f_reg[season_to_header[season_id]['biophysical']]
        for season_id in sorted(season_to_header)]

    species_foraging_activity_per_season = None
    raw_abundance_nodata = None
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
        _normalized_convolve_2d(
            (f_reg[local_floral_resource_availability_id], 1),
            (f_reg[species_file_kernel_id], 1),
            f_reg[accessable_floral_resouces_id],
            gdal.GDT_Float32, args['workspace_dir'])
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

        LOGGER.info("Calculating raw pollinator abundance for %s")
        raw_pollinator_abundance_id = (
            _RAW_POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id)
        f_reg[raw_pollinator_abundance_id] = os.path.join(
            intermediate_output_dir, "%s%s.tif" % (
                raw_pollinator_abundance_id, file_suffix))
        _normalized_convolve_2d(
            (f_reg[pollinator_supply_id], 1),
            (f_reg[species_file_kernel_id], 1),
            f_reg[raw_pollinator_abundance_id],
            gdal.GDT_Float32, args['workspace_dir'])

        LOGGER.info(
            "Calculating pollinator abundance by scaling the raw by floral "
            "resources available for %s")
        pollinator_abudanance_id = (
            _POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id)
        f_reg[pollinator_abudanance_id] = os.path.join(
            output_dir,
            pollinator_abudanance_id + "%s.tif" % file_suffix)

        raw_abundance_nodata = pygeoprocessing.get_raster_info(
            f_reg[raw_pollinator_abundance_id])['nodata'][0]

        def _pollinator_abudance_op(raw_abundance, floral_resources):
            """Multiply raw_abundance by floral_resources and skip nodata."""
            result = numpy.empty(raw_abundance.shape, dtype=numpy.float32)
            result[:] = _INDEX_NODATA
            valid_mask = (
                (raw_abundance_nodata != raw_abundance) &
                (floral_resources != _INDEX_NODATA))
            result[valid_mask] = (
                raw_abundance[valid_mask] * floral_resources[valid_mask])
            return result

        pygeoprocessing.raster_calculator(
            [(f_reg[raw_pollinator_abundance_id], 1),
             (f_reg[local_floral_resource_availability_id], 1)],
            _pollinator_abudance_op, f_reg[pollinator_abudanance_id],
            gdal.GDT_Float32, _INDEX_NODATA, calc_raster_stats=False)

    LOGGER.info("Calculating farm yields")
    farm_vector = ogr.Open(args['farm_vector_path'])
    farm_layer = farm_vector.GetLayer()
    for season_id in season_to_header:
        LOGGER.info("Rasterizing half saturation for season %s")
        half_saturation_file_path = os.path.join(
            intermediate_output_dir,
            _HALF_SATURATION_SEASON_FILE_PATTERN % season_id + (
                '%s.tif' % file_suffix))
        pygeoprocessing.new_raster_from_base(
            args['landcover_raster_path'], half_saturation_file_path,
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

        LOGGER.info("Rasterizing managed farm pollinators for season %s")
        # rasterize farm managed pollinators on landscape first
        managed_bees_raster_path = os.path.join(
            intermediate_output_dir, "%s%s.tif" % (
                _MANAGED_BEES_RASTER_FILE_PATTERN % season_id, file_suffix))
        pygeoprocessing.new_raster_from_base(
            args['landcover_raster_path'], managed_bees_raster_path,
            gdal.GDT_Float32, [_INDEX_NODATA])

        managed_raster = gdal.Open(managed_bees_raster_path, gdal.GA_Update)
        gdal.RasterizeLayer(
            managed_raster, [1], farm_layer, options=['ATTRIBUTE=p_managed'])
        gdal.Dataset.__swig_destroy__(managed_raster)
        del managed_raster
        LOGGER.info("Calculating farm pollinators for season %s")

        wild_pollinator_activity = [
            guild_table[species_id][season_to_header[season_id]['guild']]
            for species_id in sorted(guild_table)]

        def _farm_pollinators_op(
                managed_pollinator_abundance, *wild_pollinator_abundance):
            """Calculate the max of all pollinators.

                Wild pollinators need to be scaled by their seasonal foraging
                activity index included in the closure.
            """
            result = numpy.empty(
                managed_pollinator_abundance.shape, dtype=numpy.float32)
            valid_mask = wild_pollinator_abundance[0] != _INDEX_NODATA
            result[:] = _INDEX_NODATA
            result[valid_mask] = numpy.max(
                (managed_pollinator_abundance[valid_mask],
                 numpy.max([
                    activity * abundance[valid_mask]
                    for abundance, activity in zip(
                        wild_pollinator_abundance, wild_pollinator_activity)],
                    axis=0)), axis=0)
            return result

        wild_pollinator_abundance_band_paths = [
            (f_reg[_POLLINATOR_ABUNDANCE_FILE_PATTERN % species_id], 1)
            for species_id in sorted(guild_table)]

        farm_pollinators_path = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                _FARM_POLLINATORS_FILE_PATTERN % season_id, file_suffix))

        pygeoprocessing.raster_calculator(
            [(managed_bees_raster_path, 1)] +
            wild_pollinator_abundance_band_paths, _farm_pollinators_op,
            farm_pollinators_path, gdal.GDT_Float32, _INDEX_NODATA,
            calc_raster_stats=False)

        LOGGER.info("Calculating farm yield.")

        def _farm_yield_op(half_sat, farm_pollinators):
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

        pollinator_yield_path = os.path.join(
            output_dir, '%s%s.tif' % (
                _POLLINATOR_YIELD_FILE_PATTERN % season_id, file_suffix))
        pygeoprocessing.raster_calculator(
            [(half_saturation_file_path, 1), (farm_pollinators_path, 1)],
            _farm_yield_op, pollinator_yield_path, gdal.GDT_Float32,
            _INDEX_NODATA, calc_raster_stats=True)

        farm_stats = pygeoprocessing.zonal_statistics(
            (pollinator_yield_path, 1), args['farm_vector_path'], None)
        print farm_stats

    for path in temp_file_set:
        os.remove(path)


def _normalized_convolve_2d(
        signal_path_band, kernel_path_band, target_raster_path,
        target_datatype, workspace_dir):
    """Perform a normalized 2D convolution.

    Convolves the raster in `kernel_path_band` over `signal_path_band` and
    divides the result by a convolution of the kernerl over a non-nodata mask
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
            in.  Defaults to GDT_Float64.
        workspace_dir (string): path to a directory that exists where
            temporary files can be written

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
        target_datatype=target_datatype)
    pygeoprocessing.convolve_2d(
        (mask_path, 1), kernel_path_band, mask_convolve_path,
        target_datatype=target_datatype)

    base_convolve_nodata = pygeoprocessing.get_raster_info(
        base_convolve_path)['nodata'][0]

    def _divide_op(base_convolve, normalization):
        """Divide base_convolve by normalization + handle nodata/div by 0."""
        result = numpy.empty(base_convolve.shape, dtype=numpy.float32)
        valid_mask = (base_convolve != base_convolve_nodata)
        nonzero_mask = normalization != 0.0
        result[:] = base_convolve_nodata
        result[valid_mask] = base_convolve[valid_mask]
        result[valid_mask & nonzero_mask] /= normalization[
            valid_mask & nonzero_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(base_convolve_path, 1), (mask_convolve_path, 1)], _divide_op,
        target_raster_path, target_datatype, base_convolve_nodata,
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
