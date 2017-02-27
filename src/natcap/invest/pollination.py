"""Pollinator service model for InVEST."""
import os
import logging

from osgeo import gdal
import pygeoprocessing

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
                * For every nesting types in `_NESTING_TYPES`, a column
                  named 'N_[nesting_type]'.
                * For every season in `_SEASON_TYPES`, a column named
                  'F_[season_type]'

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

    guild_table = utils.build_lookup_from_csv(
        args['guild_table_path'], 'species', to_lower=True,
        numerical_cast=True)

    LOGGER.debug('TODO: make sure guild table has all expected headers')

    landcover_biophysical_table = utils.build_lookup_from_csv(
        args['landcover_biophysical_table_path'], 'lucode', to_lower=True,
        numerical_cast=True)
    LOGGER.debug(
        'TODO: make sure landcover biophysical table has all expected '
        'headers')
    for nesting_type in _NESTING_TYPES:
        nesting_id = 'n_%s' % nesting_type
        landcover_to_nesting_sutability = dict([
            (lucode, landcover_biophysical_table[lucode][nesting_id]) for
            lucode in landcover_biophysical_table])
        print landcover_to_nesting_sutability

        nesting_suitability_id = 'n_%s_index' % nesting_type
        f_reg[nesting_suitability_id] = os.path.join(
            intermediate_output_dir, 'n_%s_index%s.tif' % (
                nesting_type, file_suffix))

        pygeoprocessing.reclassify_raster(
            (args['landcover_raster_path'], 1),
            landcover_to_nesting_sutability, f_reg[nesting_suitability_id],
            gdal.GDT_Float32, _INDEX_NODATA, exception_flag='values_required')

    species_list = guild_table.keys()
    for species_id in species_list:
        habitat_nesting_raster_id = (
            '%s_habitat_nesting_suitability_path' % species_id)
        f_reg[habitat_nesting_raster_id] = os.path.join(
            output_dir, 'habitat_nesting_suitability_%s%s.tif' % (
                species_id, file_suffix))
