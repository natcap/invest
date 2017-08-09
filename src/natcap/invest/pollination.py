"""Pollinator service model for InVEST."""
import multiprocessing
import tempfile
import itertools
import collections
import re
import os
import logging
import hashlib
import inspect

from osgeo import gdal
from osgeo import ogr
import pygeoprocessing
import numpy
import taskgraph

from . import utils
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('natcap.invest.pollination')

_INDEX_NODATA = -1.0

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

_NESTING_SUBSTRATE_INDEX_FILEPATTERN = 'nesting_substrate_index_%s%s.tif'
# this is used if there is a farm polygon present
_FARM_NESTING_SUBSTRATE_INDEX_FILEPATTERN = (
    'farm_nesting_substrate_index_%s%s.tif')

# replaced by (species, file_suffix)
_HABITAT_NESTING_INDEX_FILE_PATTERN = 'habitat_nesting_index_%s%s.tif'
# replaced by (season, file_suffix)
_RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = 'relative_floral_abundance_index_%s%s.tif'

### old
_HALF_SATURATION_SEASON_FILE_PATTERN = 'half_saturation_%s'
_FARM_POLLINATORS_FILE_PATTERN = 'farm_pollinators_%s'
_FARM_FLORAL_RESOURCES_PATTERN = 'fr_([^_]+)'
_FARM_NESTING_SUBSTRATE_HEADER_PATTERN = 'n_%s'
_FARM_NESTING_SUBSTRATE_RE_PATTERN = (
    _FARM_NESTING_SUBSTRATE_HEADER_PATTERN % '([^_]+)')
_HALF_SATURATION_FARM_HEADER = 'half_sat'
_CROP_POLLINATOR_DEPENDENCE_FIELD = 'p_dep'
_MANAGED_POLLINATORS_FIELD = 'p_managed'
_FARM_SEASON_FIELD = 'season'
_EXPECTED_FARM_HEADERS = [
    'season', 'crop_type', _HALF_SATURATION_FARM_HEADER,
    _MANAGED_POLLINATORS_FIELD, _FARM_FLORAL_RESOURCES_PATTERN,
    _FARM_NESTING_SUBSTRATE_RE_PATTERN, _CROP_POLLINATOR_DEPENDENCE_FIELD]


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
    # create initial working directories and determine file suffixes
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    work_token_dir = os.path.join(
        intermediate_output_dir, '_tmp_work_tokens')
    output_dir = os.path.join(args['workspace_dir'])
    utils.make_directories(
        [output_dir, intermediate_output_dir])
    file_suffix = utils.make_suffix_string(args, 'results_suffix')

    if 'farm_vector_path' in args and args['farm_vector_path'] != '':
        farm_vector_path = args['farm_vector_path']
    else:
        farm_vector_path = None

    # parse out the scenario variables from a complicated set of two tables
    # and possibly a farm polygon.  This function will also raise an exception
    # if any of the inputs are malformed.
    scenario_variables = _parse_scenario_variables(
        args['guild_table_path'], args['landcover_biophysical_table_path'],
        farm_vector_path)

    task_graph = taskgraph.TaskGraph(work_token_dir, 0)

    # calculate nesting_substrate_index[substrate] substrate maps N(x, n) = ln(l(x), n)
    scenario_variables['nesting_substrate_index_path'] = {}
    landcover_substrate_index_tasks = {}
    for substrate in scenario_variables['substrate_list']:
        nesting_substrate_index_path = os.path.join(
            intermediate_output_dir,
            _NESTING_SUBSTRATE_INDEX_FILEPATTERN % (substrate, file_suffix))
        scenario_variables['nesting_substrate_index_path'][substrate] = (
            nesting_substrate_index_path)

        landcover_substrate_index_tasks[substrate] = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (args['landcover_raster_path'], 1),
                scenario_variables['landcover_substrate_index'][substrate],
                nesting_substrate_index_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            kwargs={'values_required': True},
            target_path_list=[nesting_substrate_index_path])

    # calculate farm_nesting_substrate_index[substrate] substrate maps
    # dependent on farm substrate rasterized over N(x, n)
    if farm_vector_path is not None:
        scenario_variables['farm_nesting_substrate_index_path'] = (
            collections.defaultdict(dict))
        farm_substrate_rasterize_task_list = []
        for substrate in scenario_variables['substrate_list']:
            farm_substrate_id = (
                _FARM_NESTING_SUBSTRATE_HEADER_PATTERN % substrate)
            farm_nesting_substrate_index_path = os.path.join(
                intermediate_output_dir,
                _FARM_NESTING_SUBSTRATE_INDEX_FILEPATTERN % (
                    substrate, file_suffix))
            scenario_variables['farm_nesting_substrate_index_path'][
                substrate] = farm_nesting_substrate_index_path
            farm_substrate_rasterize_task_list.append(
                task_graph.add_task(
                    func=_rasterize_vector_onto_base,
                    args=(
                        scenario_variables['nesting_substrate_index_path'][
                            substrate],
                        farm_vector_path, farm_substrate_id,
                        farm_nesting_substrate_index_path),
                    target_path_list=[farm_nesting_substrate_index_path],
                    dependent_task_list=[
                        landcover_substrate_index_tasks[substrate]]))

    # per species
    habitat_nesting_tasks = {}
    scenario_variables['habitat_nesting_index_path'] = {}
    for species in scenario_variables['species_list']:
        # calculate habitat_nesting_index[species] HN(x, s) = max_n(N(x, n) ns(s,n))
        if farm_vector_path is not None:
            dependent_task_list = farm_substrate_rasterize_task_list
            substrate_path_map = scenario_variables[
                'farm_nesting_substrate_index_path']
        else:
            dependent_task_list = landcover_substrate_index_tasks.values()
            substrate_path_map = scenario_variables[
                'nesting_substrate_index_path']

        scenario_variables['habitat_nesting_index_path'][species] = (
            os.path.join(
                intermediate_output_dir,
                _HABITAT_NESTING_INDEX_FILE_PATTERN % (species, file_suffix)))

        calculate_habitat_nesting_index_op = _CalculateHabitatNestingIndex(
            substrate_path_map,
            scenario_variables['species_substrate_index'][species],
            scenario_variables['habitat_nesting_index_path'][species])

        habitat_nesting_tasks['species'] = task_graph.add_task(
            func=calculate_habitat_nesting_index_op,
            dependent_task_list=dependent_task_list,
            target_path_list=[
                scenario_variables['habitat_nesting_index_path'][species]])

    # per season j
    for season in scenario_variables['season_list']:
        relative_floral_abundance_index_path = os.path.join(
            intermediate_output_dir,
            _RELATIVE_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN % (
                season, file_suffix))

        # calculate relative_floral_abundance_index[season] per season RA(l(x), j)
        relative_floral_abudance_task = task_graph.add_task(
            func=pygeoprocessing.reclassify_raster,
            args=(
                (args['landcover_raster_path'], 1),
                scenario_variables['landcover_floral_resources'][season],
                relative_floral_abundance_index_path, gdal.GDT_Float32,
                _INDEX_NODATA),
            kwargs={'values_required': True},
            target_path_list=[relative_floral_abundance_index_path])

        # if there's a farm, rasterize floral resources over the top

        # per species s
            # local foraging effectiveness foraging_effectiveness[species] FE(x, s) = sum_j [RA(l(x), j) * fa(s, j)]


    task_graph.close()
    task_graph.join()
    return

    # per species
        # accessable_floral_resources_index[species] FR(x,s) = convolve(FE(x, s), \alpha_s)
        # pollinator_supply_index[species] PS(x,s) = FR(x,s) * HN(x,s) * sa(s)

    # per species s
        # per season j
            # pollinator_abundance_index[species, season] PA(x,s,j) = RA(l(x),j)fa(s,j) convolve(PS(x',s), \alpha_s)

    # per season j
        # total_pollinator_abundance_index[season] PAT(x,j)=sum_s PA(x,s,j)



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


def _rasterize_vector_onto_base(
        base_raster_path, base_vector_path, attribute_id,
        target_raster_path, filter_string=None):
    """Rasterize attribute from vector onto a copy of base.

    Parameters:
        base_raster_path (string): path to a base raster file
        attribute_id (string): id in `base_vector_path` to rasterize.
        target_raster_path (string): a copy of `base_raster_path` with
            `base_vector_path[attribute_id]` rasterized on top.
        filter_string (string): filtering string to select from farm layer

    Returns:
        None.
    """
    base_raster = gdal.Open(base_raster_path)
    raster_driver = base_raster.GetDriver()
    target_raster = raster_driver.CreateCopy(target_raster_path, base_raster)
    base_raster = None

    vector = ogr.Open(base_vector_path)
    layer = vector.GetLayer()

    if filter_string is not None:
        layer.SetAttributeFilter(str(filter_string))
    gdal.RasterizeLayer(
        target_raster, [1], layer,
        options=['ATTRIBUTE=%s' % attribute_id])
    target_raster.FlushCache()
    target_raster = None
    layer = None
    vector = None


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
    target_layer.CreateField(ogr.FieldDefn(
        _WILD_POLLINATOR_FARM_YIELD_FIELD_ID, ogr.OFTReal))

    target_layer = None
    target_vector.SyncToDisk()
    target_vector = None


def _parse_scenario_variables(
        guild_table_path, landcover_biophysical_table_path, farm_vector_path):
    """Parse out scenario variables from input parameters.

    This function parses through the guild table, biophysical table, and
    farm polygons (if available) to generate

    Returns:
        A dictionary with the keys:
            * season_list (list of string)
            * substrate_list (list of string)
            * species_list (list of string)
            * landcover_substrate_index[substrate][landcover] (float)
            * landcover_floral_resources[season][landcover] (float)
            * species_abundance[species] (string->float)
            * species_substrate_index[(species, substrate)] (tuple->float)
            * foraging_activity_index[(species, season)] (tuple->float)
    """

    guild_table = utils.build_lookup_from_csv(
        guild_table_path, 'species', to_lower=True,
        numerical_cast=True)

    LOGGER.info('Checking to make sure guild table has all expected headers')
    guild_headers = guild_table.itervalues().next().keys()
    for header in _EXPECTED_GUILD_HEADERS:
        matches = re.findall(header, " ".join(guild_headers))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in guild table that matched the pattern "
                "'%s' but was unable to find one.  Here are all the headers "
                "from %s: %s" % (
                    header, guild_table_path,
                    guild_headers))

    landcover_biophysical_table = utils.build_lookup_from_csv(
        landcover_biophysical_table_path, 'lucode', to_lower=True,
        numerical_cast=True)
    biophysical_table_headers = (
        landcover_biophysical_table.itervalues().next().keys())
    for header in _EXPECTED_BIOPHYSICAL_HEADERS:
        matches = re.findall(header, " ".join(biophysical_table_headers))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in biophysical table that matched the "
                "pattern '%s' but was unable to find one.  Here are all the "
                "headers from %s: %s" % (
                    header, landcover_biophysical_table_path,
                    biophysical_table_headers))

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

    farm_vector = None
    if farm_vector_path is not None:
        LOGGER.info('Checking that farm polygon has expected headers')
        farm_vector = ogr.Open(farm_vector_path)
        if farm_vector.GetLayerCount() != 1:
            raise ValueError(
                "Farm polygon at %s has %d layers when expecting only 1." % (
                    farm_vector_path, farm_vector.GetLayerCount()))
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
                        header, farm_vector_path, farm_headers))

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

    result = {}
    # * season_list (list of string)
    result['season_list'] = sorted(season_to_header)
    # * substrate_list (list of string)
    result['substrate_list'] = sorted(substrate_to_header)
    # * species_list (list of string)
    result['species_list'] = sorted(guild_table)

    # * species_abundance[species] (string->float)
    total_relative_abundance = numpy.sum([
        guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD]
        for species in result['species_list']])
    result['species_abundance'] = {}
    for species in result['species_list']:
        result['species_abundance'][species] = (
            guild_table[species][_RELATIVE_SPECIES_ABUNDANCE_FIELD] /
            total_relative_abundance)

    # * landcover_substrate_index[substrate][landcover] (float)
    result['landcover_substrate_index'] = collections.defaultdict(dict)
    for raw_landcover_id in landcover_biophysical_table:
        landcover_id = int(raw_landcover_id)
        for substrate in result['substrate_list']:
            substrate_biophysical_header = (
                substrate_to_header[substrate]['biophysical'])
            result['landcover_substrate_index'][substrate][landcover_id] = (
                landcover_biophysical_table[landcover_id][
                    substrate_biophysical_header])

    # * landcover_floral_resources[season][landcover] (float)
    result['landcover_floral_resources'] = collections.defaultdict(dict)
    for raw_landcover_id in landcover_biophysical_table:
        landcover_id = int(raw_landcover_id)
        for season in result['season_list']:
            floral_rources_header = season_to_header[season]['biophysical']
            result['landcover_floral_resources'][season][landcover_id] = (
                landcover_biophysical_table[landcover_id][
                    floral_rources_header])

    # * species_substrate_index[(species, substrate)] (tuple->float)
    result['species_substrate_index'] = collections.defaultdict(dict)
    for species in result['species_list']:
        for substrate in result['substrate_list']:
            substrate_guild_header = substrate_to_header[substrate]['guild']
            result['species_substrate_index'][species][substrate] = (
                guild_table[species][substrate_guild_header])

    # * foraging_activity_index[(species, season)] (tuple->float)
    result['foraging_activity_index'] = {}
    for species in result['species_list']:
        for season in result['season_list']:
            key = (species, season)
            foraging_biophyiscal_header = season_to_header[season]['guild']
            result['foraging_activity_index'][key] = (
                guild_table[species][foraging_biophyiscal_header])

    return result


class _CalculateHabitatNestingIndex(object):
    """Closure for HN(x, s) = max_n(N(x, n) ns(s,n)) calculation."""

    def __init__(
            self, substrate_path_map, species_substrate_index_map,
            target_habitat_nesting_index_path):
        """Define parameters necessary for HN(x,s) calculation.

        Parameters:
            substrate_path_map (dict): map substrate name to substrate index
                raster path. (N(x, n))
            species_substrate_index_map (dict): map substrate name to
                scalar value of species substrate suitability. (ns(s,n))
            target_habitat_nesting_index_path (string): path to target
                raster
        """
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(inspect.getsource(
                _CalculateHabitatNestingIndex.__call__)).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = _CalculateHabitatNestingIndex.__name__
        self.substrate_path_list = [
            substrate_path_map[substrate_id]
            for substrate_id in sorted(substrate_path_map)]

        self.species_substrate_suitability_index_array = numpy.array([
            species_substrate_index_map[substrate_id]
            for substrate_id in sorted(substrate_path_map)]).reshape(
                (len(species_substrate_index_map), 1))
        LOGGER.debug(self.species_substrate_suitability_index_array)

        self.target_habitat_nesting_index_path = (
            target_habitat_nesting_index_path)

    def __call__(self):

        def max_op(*substrate_index_arrays):
            """Return the max of index_array[n] * ns[n]."""
            result = numpy.max(
                numpy.stack([x.flatten() for x in substrate_index_arrays]) *
                self.species_substrate_suitability_index_array, axis=0)
            result = result.reshape(substrate_index_arrays[0].shape)
            result[substrate_index_arrays[0] == _INDEX_NODATA] = _INDEX_NODATA
            return result

        pygeoprocessing.raster_calculator(
            [(x, 1) for x in self.substrate_path_list], max_op,
            self.target_habitat_nesting_index_path,
            gdal.GDT_Float32, _INDEX_NODATA)
