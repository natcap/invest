"""Pollinator service model for InVEST."""
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

_NESTING_TYPES = ['cavity', 'ground']
_SEASON_TYPES = ['spring', 'summer']
_LANDCOVER_NESTING_INDEX_HEADER = r'nesting_%s_index'
_SPECIES_NESTING_TYPE_INDEX_HEADER = r'nesting_suitability_%s_index'
_RELATIVE_FLORAL_ABUDANCE_INDEX_HEADER = r'floral_resources_%s_index'
_SPECIES_SEASONAL_FORAGING_ACTIVITY_HEADER = r'foraging_activity_%s_index'
_SPECIES_ALPHA_KERNEL_FILE_PATTERN = r'alpha_kernel_%s'
_ACCESSABLE_FLORAL_RESOURCES_FILE_PATTERN = r'accessable_floral_resources_%s'
_POLLINATOR_SUPPLY_FILE_PATTERN = r'pollinator_supply_%s_index'


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
            nesting availability for each of `_NESTING_TYPES` as well as
            indexes of abundance of floral resources on that landcover type
            per season in `_SEASON_TYPES`.

            All indexes are in the range [0.0, 1.0].

            Columns in the table must be at least
                * 'lucode': representing all the unique landcover codes in
                    the raster ast `args['landcover_path']`
                * For every nesting type in `_NESTING_TYPES`, a column
                  matching the pattern in `_LANDCOVER_NESTING_INDEX_HEADER`.
                * For every season in `_SEASON_TYPES`, a column matching
                  the pattern in `_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`.

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

    LOGGER.debug('TODO: make sure guild table has all expected headers')

    landcover_biophysical_table = utils.build_lookup_from_csv(
        args['landcover_biophysical_table_path'], 'lucode', to_lower=True,
        numerical_cast=True)
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['landcover_raster_path'])
    LOGGER.debug(
        'TODO: make sure landcover biophysical table has all expected '
        'headers')
    for nesting_type in _NESTING_TYPES:
        nesting_id = _LANDCOVER_NESTING_INDEX_HEADER % nesting_type
        landcover_to_nesting_sutability_table = dict([
            (lucode, landcover_biophysical_table[lucode][nesting_id]) for
            lucode in landcover_biophysical_table])

        f_reg[nesting_id] = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (nesting_id, file_suffix))

        pygeoprocessing.reclassify_raster(
            (args['landcover_raster_path'], 1),
            landcover_to_nesting_sutability_table, f_reg[nesting_id],
            gdal.GDT_Float32, _INDEX_NODATA, exception_flag='values_required')

    species_list = guild_table.keys()
    species_nesting_suitability_index = None
    for species_id in species_list:
        species_nesting_id = (
            _SPECIES_NESTING_TYPE_INDEX_HEADER % species_id)
        LOGGER.debug(guild_table)
        species_nesting_suitability_index = numpy.array([
            guild_table[species_id][
                _SPECIES_NESTING_TYPE_INDEX_HEADER % nesting_type]
            for nesting_type in _NESTING_TYPES])

        LOGGER.info("Calculate species nesting index for %s", species_id)

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

        f_reg[species_nesting_id] = os.path.join(
            intermediate_output_dir, '%s%s.tif' % (
                species_nesting_id, file_suffix))
        pygeoprocessing.raster_calculator(
            [(path, 1) for path in [
                f_reg[_LANDCOVER_NESTING_INDEX_HEADER % nesting_type]
                for nesting_type in _NESTING_TYPES]],
            _habitat_suitability_index_op, f_reg[species_nesting_id],
            gdal.GDT_Float32, _INDEX_NODATA, calc_raster_stats=False)

    for season_id in _SEASON_TYPES:
        relative_floral_abudance_id = (
            _RELATIVE_FLORAL_ABUDANCE_INDEX_HEADER % (season_id))
        f_reg[relative_floral_abudance_id] = os.path.join(
            intermediate_output_dir,
            relative_floral_abudance_id + "%s.tif" % file_suffix)

        landcover_to_floral_abudance_table = dict([
            (lucode, landcover_biophysical_table[lucode][
                relative_floral_abudance_id]) for lucode in
            landcover_biophysical_table])

        pygeoprocessing.reclassify_raster(
            (args['landcover_raster_path'], 1),
            landcover_to_floral_abudance_table,
            f_reg[relative_floral_abudance_id], gdal.GDT_Float32,
            _INDEX_NODATA, exception_flag='values_required')

    LOGGER.debug(
        "TODO: consider making species season floral weight relative "
        "rather than absolute.")
    species_season_floral_weight_index = None
    for species_id in species_list:
        species_season_floral_weight_index = numpy.array([
            guild_table[species_id][
                _SPECIES_SEASONAL_FORAGING_ACTIVITY_HEADER % season_id]
            for season_id in _SEASON_TYPES])
        relative_floral_abudance_species_id = (
            _SPECIES_SEASONAL_FORAGING_ACTIVITY_HEADER % species_id)
        f_reg[relative_floral_abudance_species_id] = os.path.join(
            intermediate_output_dir,
            relative_floral_abudance_species_id + "%s.tif" % file_suffix)

        def _species_floral_abudance_op(*floral_abudance_index):
            """Calculate species floral abudance."""
            result = numpy.empty(
                floral_abudance_index[0].shape, dtype=numpy.float32)
            result[:] = _INDEX_NODATA
            valid_mask = floral_abudance_index[0] != _INDEX_NODATA
            result[valid_mask] = numpy.sum(
                [fai[valid_mask] * ssfwi for fai, ssfwi in zip(
                    floral_abudance_index,
                    species_season_floral_weight_index)], axis=0)
            return result

        pygeoprocessing.raster_calculator(
            [(path, 1) for path in [
                f_reg[_RELATIVE_FLORAL_ABUDANCE_INDEX_HEADER % season_id]
                for season_id in _SEASON_TYPES]],
            _species_floral_abudance_op,
            f_reg[relative_floral_abudance_species_id], gdal.GDT_Float32,
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
            output_dir, accessable_floral_resouces_id +
            '%s.tif' % file_suffix)
        temp_file_set.add(f_reg[accessable_floral_resouces_id])
        LOGGER.info(
            "Calculating available floral resources for %s", species_id)
        pygeoprocessing.convolve_2d(
            (f_reg[relative_floral_abudance_species_id], 1),
            (f_reg[species_file_kernel_id], 1),
            f_reg[accessable_floral_resouces_id],
            target_datatype=gdal.GDT_Float32)
        LOGGER.info("Calculating pollinator supply for %s", species_id)
        pollinator_supply_id = (_POLLINATOR_SUPPLY_FILE_PATTERN % species_id)
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

        species_nesting_id = (
            _SPECIES_NESTING_TYPE_INDEX_HEADER % species_id)
        pygeoprocessing.raster_calculator(
            [(f_reg[accessable_floral_resouces_id], 1),
             (f_reg[species_nesting_id], 1)], _pollinator_supply_op,
            f_reg[pollinator_supply_id], gdal.GDT_Float32,
            _INDEX_NODATA, calc_raster_stats=False)

    for path in temp_file_set:
        os.remove(path)
