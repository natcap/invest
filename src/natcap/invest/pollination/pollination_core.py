"""InVEST Pollination model core module"""

import shutil
import os
import logging

from osgeo import gdal
import numpy
import pygeoprocessing.geoprocessing

from .. import utils

LOGGER = logging.getLogger('natcap.invest.pollination.core')


def execute_model(args):
    """Execute the biophysical component of the pollination model.

        args - a python dictionary with at least the following entries:
            'landuse' - a URI to a GDAL dataset
            'landuse_attributes' - A fileio AbstractTableHandler object
            'guilds' - A fileio AbstractTableHandler object
            'ag_classes' - a python list of ints representing agricultural
                classes in the landuse map.  This list may be empty to represent
                the fact that no landuse classes are to be designated as strictly
                agricultural.
            'nesting_fields' - a python list of string nesting fields
            'floral fields' - a python list of string floral fields
            'do_valuation' - a boolean indicating whether to do valuation
            'paths' - a dictionary with the following entries:
                'workspace' - the workspace path
                'intermediate' - the intermediate folder path
                'output' - the output folder path
                'temp' - a temp folder path.

        Additionally, the args dictionary should contain these URIs, which must
        all be python strings of either type str or else utf-8 encoded unicode.
            'ag_map' - a URI
            'foraging_average' - a URI
            'abundance_total' - a URI
            'farm_value_sum' - a URI (Required if do_valuation == True)
            'service_value_sum' - a URI (Required if do_valuation == True)

        The args dictionary must also have a dictionary containing
        species-specific information:
            'species' - a python dictionary with a contained dictionary for each
                species to be considered by the model.  The key to each
                dictionary should be the species name.  For example:

                    args['species']['Apis'] = { ... species_dictionary ... }

                The species-specific dictionary must contain these elements:
                    'floral' - a URI
                    'nesting' - a URI
                    'species_abundance' - a URI
                    'farm_abundance' - a URI

                If do_valuation == True, the following entries are also required
                to be in the species-specific dictionary:
                    'farm_value' - a URI
                    'value_abundance_ratio' - a URI
                    'value_abundance_ratio_blurred' - a URI
                    'service_value' - a URI

        returns nothing."""

    LOGGER.debug('Starting pollination calculations')

    nodata = -1.0
    LOGGER.debug('Using nodata value of %s for internal rasters', nodata)

    reclass_ag_raster(
        args['landuse'], args['ag_map'], args['ag_classes'], nodata)

    # Create the necessary sum rasters by reclassifying the ag map so that all
    # pixels that are not nodata have a value of 0.0.

    output_uri_list = [args['foraging_average'], args['abundance_total']]
    if args['do_valuation']:
        output_uri_list.extend(
            [args['farm_value_sum'], args['service_value_sum']])

    ag_map_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(
        args['ag_map'])
    def mask_op(ag_value):
        """return 0.0 where not original nodata, and 0.0 otherwise"""
        return numpy.where(ag_value == ag_map_nodata, nodata, 0.0)
    pixel_size_out = pygeoprocessing.geoprocessing.get_cell_size_from_uri(
        args['ag_map'])
    for out_uri in output_uri_list:
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [args['ag_map']], mask_op, out_uri, gdal.GDT_Float32, nodata,
            pixel_size_out, 'intersection', vectorize_op=False)

    # Loop through all species and perform the necessary calculations.
    for species, species_dict in args['species'].iteritems():
        LOGGER.debug('Starting %s species', species)

        # We need the guild dictionary for a couple different things later on
        guild_dict = args['guilds'].get_table_row('species', species)

        # Calculate species abundance.  This represents the relative index of
        # how much of a species we can expect to find across the landscape given
        # the floral and nesting patterns (based on land cover) and the
        # specified use of these resources (defined in the guild_dict).
        LOGGER.info('Calculating %s abundance on the landscape', species)
        calculate_abundance(args['landuse'],
            args['landuse_attributes'], guild_dict, args['nesting_fields'],
            args['floral_fields'], uris={
                'nesting': species_dict['nesting'],
                'floral': species_dict['floral'],
                'species_abundance': species_dict['species_abundance'],
                'temp': args['paths']['temp']
            })

        # Add the newly-calculated abundance to the abundance_sum raster.
        LOGGER.info('Adding %s species abundance to the total', species)
        add_two_rasters(args['abundance_total'],
            species_dict['species_abundance'], args['abundance_total'])

        # Calculate the farm abundance.  This takes the species abundance and
        # calculates roughly how much of a species we can expect to find on farm
        # pixels.
        LOGGER.info('Calculating %s abundance on farms ("foraging")', species)
        calculate_farm_abundance(species_dict['species_abundance'],
            args['ag_map'], guild_dict['alpha'], species_dict['farm_abundance'],
            args['paths']['intermediate'])

        # Add the newly calculated farm abundance raster to the total.
        LOGGER.info('Adding %s foraging abundance raster to total', species)
        add_two_rasters(species_dict['farm_abundance'],
            args['foraging_average'], args['foraging_average'])

        if args['do_valuation'] == True:
            LOGGER.info('Starting species-specific valuation for %s', species)

            # Apply the half-saturation yield function from the documentation
            # and write it to its raster
            LOGGER.info('Calculating crop yield due to %s', species)
            calculate_yield(species_dict['farm_abundance'],
                species_dict['farm_value'], args['half_saturation'],
                args['wild_pollination_proportion'], -1.0)

            # Add the new farm_value_matrix to the farm value sum matrix.
            LOGGER.info('Adding crop yield due to %s to the crop yield total',
                species)
            add_two_rasters(args['farm_value_sum'], species_dict['farm_value'],
                args['farm_value_sum'])

            LOGGER.info('Calculating service value for %s', species)
            guild_dict = args['guilds'].get_table_row('species', species)
            sample_raster = gdal.Open(args['farm_value_sum'])
            pixel_size = abs(sample_raster.GetGeoTransform()[1])
            sample_raster = None

            calculate_service(
                rasters={
                    'farm_value': species_dict['farm_value'],
                    'farm_abundance': species_dict['farm_abundance'],
                    'species_abundance': species_dict['species_abundance'],
                    'ag_map': args['ag_map']
                },
                nodata=-1.0,
                alpha=float(guild_dict['alpha']),
                part_wild=args['wild_pollination_proportion'],
                out_uris={
                    'species_value': species_dict['value_abundance_ratio'],
                    'species_value_blurred':\
                        species_dict['value_abundance_ratio_blur'],
                    'service_value': species_dict['service_value'],
                    'temp': args['paths']['temp']
                })

            # Add the new service value to the service value sum matrix
            LOGGER.info('Adding the %s service value raster to the sum',
                species)
            add_two_rasters(args['service_value_sum'],
                species_dict['service_value'], args['service_value_sum'])

    # Calculate the average foraging index based on the total
    # Divide the total pollination foraging index by the number of pollinators
    # to get the mean pollinator foraging index and save that to its raster.
    num_species = float(len(args['species'].values()))
    LOGGER.debug('Number of species: %s', num_species)

    # Calculate the mean foraging values per species.
    LOGGER.debug('Calculating mean foraging score')
    divide_raster(args['foraging_average'], num_species,
        args['foraging_average'])

    # Calculate the mean pollinator supply (pollinator abundance) by taking the
    # abundance_total_matrix and dividing it by the number of pollinators.
    LOGGER.debug('Calculating mean pollinator supply')
    divide_raster(args['abundance_total'], num_species, args['abundance_total'])

    LOGGER.debug('Finished pollination biophysical calculations')


def calculate_abundance(landuse, lu_attr, guild, nesting_fields,
    floral_fields, uris):
    """Calculate pollinator abundance on the landscape.  The calculated
    pollinator abundance raster will be created at uris['species_abundance'].

        landuse - a URI to a GDAL dataset of the LULC.
        lu_attr - a TableHandler
        guild - a dictionary containing information about the pollinator.  All
            entries are required:
            'alpha' - the typical foraging distance in m
            'species_weight' - the relative weight
            resource_n - One entry for each nesting field for each fieldname
                denoted in nesting_fields.  This value must be either 0 and 1,
                indicating whether the pollinator uses this nesting resource for
                nesting sites.
            resource_f - One entry for each floral field denoted in
                floral_fields.  This value must be between 0 and 1,
                representing the liklihood that this species will forage during
                this season.
        nesting_fields - a list of string fieldnames.  Used to extract nesting
            fields from the guild dictionary, so fieldnames here must exist in
            guild.
        floral_fields - a list of string fieldnames.  Used to extract floral
            season fields from the guild dictionary.  Fieldnames here must exist
            in guild.
        uris - a dictionary with these entries:
            'nesting' - a URI to where the nesting raster will be saved.
            'floral' - a URI to where the floral resource raster will be saved.
            'species_abundance' - a URI to where the species abundance raster
                will be saved.
            'temp' - a URI to a folder where temp files will be saved

        Returns nothing."""
    nodata = -1.0


    floral_raster_temp_uri = pygeoprocessing.geoprocessing.temporary_filename()

    LOGGER.debug('Mapping floral attributes to landcover, writing to %s',
        floral_raster_temp_uri)
    map_attribute(landuse, lu_attr, guild, floral_fields,
        floral_raster_temp_uri, sum)
    map_attribute(landuse, lu_attr, guild, nesting_fields, uris['nesting'], max)

    # Now that the per-pixel nesting and floral resources have been
    # calculated, the floral resources still need to factor in
    # neighborhoods.
    lulc_ds = gdal.Open(landuse)
    pixel_size = abs(lulc_ds.GetGeoTransform()[1])
    lulc_ds = None
    expected_distance = guild['alpha'] / pixel_size
    kernel_uri = pygeoprocessing.geoprocessing.temporary_filename()
    utils.exponential_decay_kernel_raster(expected_distance, kernel_uri)
    LOGGER.debug('expected distance: %s ', expected_distance)

    # Fetch the floral resources raster and matrix from the args dictionary
    # apply an exponential convolution filter and save the floral resources raster to the
    # dataset.
    LOGGER.debug('Applying neighborhood mappings to floral resources')
    pygeoprocessing.geoprocessing.convolve_2d_uri(
        floral_raster_temp_uri, kernel_uri, uris['floral'])
    os.remove(kernel_uri)
    # Calculate the pollinator abundance index (using Math! to simplify the
    # equation in the documentation.  We're still waiting on Taylor
    # Rickett's reply to see if this is correct.
    # Once the pollination supply has been calculated, we add it to the
    # total abundance matrix.
    LOGGER.debug('Calculating abundance index')
    species_weight = float(guild['species_weight'])

    pygeoprocessing.geoprocessing.vectorize_datasets(
        [uris['nesting'], uris['floral']],
        lambda x, y: numpy.where(x != nodata, numpy.multiply(numpy.multiply(x,
            y), species_weight), nodata),
        dataset_out_uri=uris['species_abundance'],
        datatype_out=gdal.GDT_Float32, nodata_out=nodata,
        pixel_size_out=pixel_size, bounding_box_mode='intersection',
        vectorize_op=False)


def calculate_farm_abundance(species_abundance, ag_map, alpha, uri, temp_dir):
    """Calculate the farm abundance raster.  The final farm abundance raster
    will be saved to uri.

        species_abundance - a URI to a GDAL dataset of species abundance.
        ag_map - a uri to a GDAL dataset of values where ag pixels are 1
            and non-ag pixels are 0.
        alpha - the typical foraging distance of the current pollinator.
        uri - the output URI for the farm_abundance raster.
        temp_dir- the output folder for temp files

        Returns nothing."""

    LOGGER.debug('Starting to calculate farm abundance')

    farm_abundance_temp_uri = pygeoprocessing.geoprocessing.temporary_filename()
    LOGGER.debug('Farm abundance temp file saved to %s',
        farm_abundance_temp_uri)

    species_abundance_uri = species_abundance
    species_abundance = gdal.Open(species_abundance_uri)

    pixel_size = abs(species_abundance.GetGeoTransform()[1])
    expected_distance = alpha / pixel_size

    kernel_uri = pygeoprocessing.geoprocessing.temporary_filename()
    utils.exponential_decay_kernel_raster(expected_distance, kernel_uri)

    LOGGER.debug('Calculating foraging/farm abundance index')
    pygeoprocessing.geoprocessing.convolve_2d_uri(species_abundance_uri, kernel_uri, farm_abundance_temp_uri)
    os.remove(kernel_uri)

    nodata = species_abundance.GetRasterBand(1).GetNoDataValue()
    LOGGER.debug('Using nodata value %s from species abundance raster', nodata)

    # Mask the farm abundance raster according to whether the pixel is
    # agricultural.  If the pixel is agricultural, the value is preserved.
    # Otherwise, the value is set to nodata.
    LOGGER.debug('Setting all agricultural pixels to 0')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list=[farm_abundance_temp_uri, ag_map],
        dataset_pixel_op=lambda x, y: numpy.where(y == 1.0, x, nodata),
        dataset_out_uri=uri,
        datatype_out=gdal.GDT_Float32,
        nodata_out=nodata,
        pixel_size_out=pixel_size,
        bounding_box_mode='intersection',
        vectorize_op=False)

def reclass_ag_raster(landuse_uri, out_uri, ag_classes, nodata):
    """Reclassify the landuse raster into a raster demarcating the agricultural
        state of a given pixel.  The reclassed ag raster will be saved to uri.

        landuse - a GDAL dataset.  The land use/land cover raster.
        out_uri - the uri of the output, reclassified ag raster.
        ag_classes - a list of landuse classes that are agricultural.  If an
            empty list is provided, all landcover classes are considered to be
            agricultural.
        nodata - an int or float.

        Returns nothing."""

    LOGGER.info(
        'Starting to create an ag raster at %s. Nodata=%s', out_uri, nodata)
    reclass_rules = {}
    default_value = 1.0
    if len(ag_classes) > 0:
        reclass_rules = dict((r, 1) for r in ag_classes)
        default_value = 0.0

    lulc_values = pygeoprocessing.geoprocessing.unique_raster_values_uri(
        landuse_uri)

    for lucode in lulc_values:
        if lucode not in reclass_rules:
            reclass_rules[lucode] = default_value

    LOGGER.debug('Agricultural reclass map=%s', reclass_rules)
    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        landuse_uri, reclass_rules, out_uri, gdal.GDT_Float32, nodata,
        exception_flag='values_required')


def add_two_rasters(raster_1, raster_2, out_uri):
    """Add two rasters where pixels in raster_1 are not nodata.  Pixels are
        considered to have a nodata value iff the pixel value in raster_1 is
        nodata.  Raster_2's pixel value is not checked for nodata.

        raster_1 - a uri to a GDAL dataset
        raster_2 - a uri to a GDAL dataset
        out_uri - the uri at which to save the resulting raster.

        Returns nothing."""

    # If the user wants us to write the output raster to the URI of one of the
    # input files, create a temporary directory and save the output file to the
    # temp folder.
    temp_dir = None
    if out_uri in [raster_1, raster_2]:
        old_out_uri = out_uri
        temp_dir = True
        out_uri = pygeoprocessing.geoprocessing.temporary_filename()
        LOGGER.debug('Sum will be saved to temp file %s', out_uri)

    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(raster_1)
    min_pixel_size = min(map(pygeoprocessing.geoprocessing.get_cell_size_from_uri, [raster_1,
        raster_2]))

    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list=[raster_1, raster_2],
        dataset_pixel_op=lambda x, y: numpy.where(y != nodata, numpy.add(x, y),
            nodata),
        dataset_out_uri=out_uri,
        datatype_out=gdal.GDT_Float32,
        nodata_out=nodata,
        pixel_size_out=min_pixel_size,
        bounding_box_mode='intersection',
        vectorize_op=False)

    # If we saved the output file to a temp folder, remove the file that we're
    # trying to avoid and save the temp file to the old file's location.
    if temp_dir != None:
        os.remove(old_out_uri)
        shutil.move(out_uri, old_out_uri)
        LOGGER.debug('Moved temp sum to %s', old_out_uri)


def calculate_service(rasters, nodata, alpha, part_wild, out_uris):
    """Calculate the service raster.  The finished raster will be saved to
    out_uris['service_value'].

        rasters - a dictionary with these entries:
            'farm_value' - a GDAL dataset.
            'farm_abundance' - a GDAL dataset.
            'species_abundance' - a GDAL dataset.
            'ag_map' - a GDAL dataset.  Values are either nodata, 0 (if not an
                    ag pixel) or 1 (if an ag pixel).
        nodata - the nodata value for output rasters.
        alpha - the expected distance
        part_wild - a number between 0 and 1 representing the proportion of all
            pollination that is done by wild pollinators.
        out_uris - a dictionary with these entries:
            'species_value' - a URI.  The raster created at this URI will
                represent the part of the farm's value that is attributed to the
                current species.
            'species_value_blurred' - the raster created at this URI
                will be a copy of the species_value raster that has had a
                exponential convolution filter applied to it.
            'service_value' - a URI.  The raster created at this URI will be the
                calculated service value raster.
            'temp' - a folder in which to store temp files.

        Returns nothing."""

    LOGGER.debug('Calculating the service value')

    # Open the species foraging matrix and then divide
    # the yield matrix by the foraging matrix for this pollinator.
    LOGGER.debug('Calculating pollinator value to farms')
    min_pixel_size = min(map(pygeoprocessing.geoprocessing.get_cell_size_from_uri,
        [rasters['farm_value'], rasters['farm_abundance']]))

    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list=[rasters['farm_value'], rasters['farm_abundance']],
        dataset_pixel_op=lambda x, y: numpy.where(x != nodata, numpy.divide(x,
            y), nodata),
        dataset_out_uri=out_uris['species_value'],
        datatype_out=gdal.GDT_Float32,
        nodata_out=nodata,
        pixel_size_out=min_pixel_size,
        bounding_box_mode='intersection',
        vectorize_op=False)

    expected_distance = alpha / min_pixel_size
    kernel_uri = pygeoprocessing.geoprocessing.temporary_filename()
    utils.exponential_decay_kernel_raster(expected_distance, kernel_uri)
    LOGGER.debug('Exponetial decay on ratio raster')
    pygeoprocessing.geoprocessing.convolve_2d_uri(
        out_uris['species_value'], kernel_uri, out_uris['species_value_blurred'])
    os.remove(kernel_uri)

    # Vectorize the ps_vectorized function
    LOGGER.debug('Attributing farm value to the current species')

    temp_service_uri = pygeoprocessing.geoprocessing.temporary_filename()

    LOGGER.debug('Saving service value raster to %s', temp_service_uri)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [rasters['species_abundance'], out_uris['species_value_blurred']],
        lambda x, y: numpy.where(x != nodata, numpy.multiply(part_wild,
            numpy.multiply(x, y)), nodata),
        dataset_out_uri=temp_service_uri,
        datatype_out=gdal.GDT_Float32,
        nodata_out=nodata,
        pixel_size_out=min_pixel_size,
        bounding_box_mode='intersection',
        vectorize_op=False)

    # Set all agricultural pixels to 0.  This is according to issue 761.
    LOGGER.debug('Marking the value of all non-ag pixels as 0.0.')
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list=[rasters['ag_map'], temp_service_uri],
        dataset_pixel_op=lambda x, y: numpy.where(x == 0, 0.0, y),
        dataset_out_uri=out_uris['service_value'],
        datatype_out=gdal.GDT_Float32,
        nodata_out=nodata,
        pixel_size_out=min_pixel_size,
        bounding_box_mode='intersection',
        vectorize_op=False)

    LOGGER.debug('Finished calculating service value')


def calculate_yield(in_raster, out_uri, half_sat, wild_poll, out_nodata):
    """Calculate the yield raster.

        in_raster - a uri to a GDAL dataset
        out_uri -a uri for the output (yield) dataset
        half_sat - the half-saturation constant, a python int or float
        wild_poll - the proportion of crops that are pollinated by wild
            pollinators.  An int or float from 0 to 1.
        out_nodata - the nodata value for the output raster

        Returns nothing"""

    LOGGER.debug('Calculating yield')

    # Calculate the yield raster
    kappa_c = float(half_sat)
    nu_c = float(wild_poll)
    nu_c_invert = 1.0 - nu_c
    in_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(in_raster)

    # This function is a vectorize-compatible implementation of the yield
    # function from the documentation.
    def calc_yield(frm_avg):
        """Calculate the yield for a farm pixel.  frm_avg is the average
        foraging score on the landscape on this pixel aross all pollinators.
        This function applies the 'expected yield' function from the
        documentation."""
        return numpy.where(frm_avg == in_nodata, out_nodata,
            nu_c_invert + (nu_c * numpy.divide(frm_avg,
            frm_avg + kappa_c)))

    # Apply the yield calculation to the foraging_average raster
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list=[in_raster],
        dataset_pixel_op=calc_yield,
        dataset_out_uri=out_uri,
        datatype_out=gdal.GDT_Float32,
        nodata_out=out_nodata,
        pixel_size_out=pygeoprocessing.geoprocessing.get_cell_size_from_uri(in_raster),
        bounding_box_mode='intersection',
        vectorize_op=False)


def divide_raster(raster, divisor, uri):
    """Divide all non-nodata values in raster_1 by divisor and save the output
        raster to uri.

        raster - a uri to a GDAL dataset
        divisor - the divisor (a python scalar)
        uri - the uri to which to save the output raster.

        Returns nothing."""

    temp_dir = None
    if raster == uri:
        old_out_uri = uri
        temp_dir = True
        uri = pygeoprocessing.geoprocessing.temporary_filename()
        LOGGER.debug('Quotient raster will be saved to temp file %s', uri)

    nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(raster)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        dataset_uri_list=[raster],
        dataset_pixel_op=lambda x: numpy.where(x == nodata, nodata,
            x / divisor),
        dataset_out_uri=uri,
        datatype_out=gdal.GDT_Float32,
        nodata_out=nodata,
        pixel_size_out=pygeoprocessing.geoprocessing.get_cell_size_from_uri(raster),
        bounding_box_mode='intersection',
        vectorize_op=False)

    raster = None
    if temp_dir != None:
        os.remove(old_out_uri)
        shutil.move(uri, old_out_uri)
        LOGGER.debug('Moved temp quotient to %s', old_out_uri)

def map_attribute(base_raster, attr_table, guild_dict, resource_fields,
                  out_uri, list_op):
    """Make an intermediate raster where values are mapped from the base raster
        according to the mapping specified by key_field and value_field.

        base_raster - a URI to a GDAL dataset
        attr_table - a subclass of fileio.AbstractTableHandler
        guild_dict - a python dictionary representing the guild row for this
            species.
        resource_fields - a python list of string resource fields
        out_uri - a uri for the output dataset
        list_op - a python callable that takes a list of numerical arguments
            and returns a python scalar.  Examples: sum; max

        returns nothing."""

    # Get the input raster's nodata value
    base_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(base_raster)

    # Get the output raster's nodata value

    lu_table_dict = attr_table.get_table_dictionary('lulc')

    value_list = dict((r, guild_dict[r]) for r in resource_fields)

    reclass_rules = {base_nodata: -1}
    for lulc in lu_table_dict.keys():
        resource_values = [
            value_list[r] * lu_table_dict[lulc][r] for r in resource_fields]
        reclass_rules[lulc] = list_op(resource_values)

    pygeoprocessing.geoprocessing.reclassify_dataset_uri(
        base_raster, reclass_rules, out_uri, gdal.GDT_Float32, -1)
