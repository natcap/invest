"""Pollinator service model for InVEST."""

import os
import logging
import re
import sys
import shutil
import struct

from osgeo import gdal
from osgeo import ogr

from natcap.invest import fileio as fileio
from natcap.invest.pollination import pollination_core as pollination_core

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
     %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.pollination.pollination')


class MissingFields(ValueError): pass


def execute(args):
    """
    Execute the pollination model from the topmost, user-accessible level.

    Args:
        workspace_dir (string): a URI to the workspace folder.  Not required to
            exist on disk.  Additional folders will be created inside of
            this folder.  If there are any file name collisions, this model
            will overwrite those files.
        landuse_cur_uri (string): a URI to a GDAL raster on disk.
            'do_valuation' - A boolean.  Indicates whether valuation should be
            performed.  This applies to all scenarios.
        landuse_attributes_uri (string): a URI to a CSV on disk.  See the
            model's documentation for details on the structure of this
            table.
        landuse_fut_uri (string): (Optional) a URI to a GDAL dataset on disk.
            If this args dictionary entry is provided, this model will process
            both the current and future scenarios.
        do_valuation (boolean): Indicates whether the model should include
            valuation
        half_saturation (float): a number between 0 and 1 indicating the
            half-saturation constant. See the pollination documentation for
            more information.
        wild_pollination_proportion (float): a number between 0 and 1
            indicating the proportion of all pollinators that are wild.
            See the pollination documentation for more information.
        guilds_uri (string): a URI to a CSV on disk.  See the model's
            documentation for details on the structure of this table.
        ag_classes (string): (Optional) a space-separated list of land cover
            classes that are to be considered as agricultural.  If this
            input is not provided, all land cover classes are considered to
            be agricultural.
        farms_shapefile (string): (Optional) shapefile containing points
            representing data collection points on the landscape.
        results_suffix (string): inserted into the URI of each file created by
            this model, right before the file extension.

    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir',
            'landuse_cur_uri': 'path/to/raster',
            'landuse_attributes_uri': 'path/to/csv',
            'landuse_fut_uri': 'path/to/raster',
            'do_valuation': 'example',
            'half_saturation': 'example',
            'wild_pollination_proportion': 'example',
            'guilds_uri': 'path/to/csv',
            'ag_classes': 'example',
            'farms_shapefile': 'example',
            'results_suffix': 'example',

        }

    The following args dictionary entries are optional, and
    will affect the behavior of the model if provided:
        1. landuse_fut_uri
        2. ag_classes
        3. results_suffix
        4. farms_shapefile

    If args['do_valuation'] is set to True, the following args dictionary
    entries are also required:
        1. half_saturation
        2. wild_pollination_proportion

    This function has no return value, though it does save a number of
    rasters to disk.  See the user's guide for details.

    """

    workspace = args['workspace_dir']

    # If the user has not provided a results suffix, assume it to be an empty
    # string.
    try:
        suffix = args['results_suffix']
    except KeyError:
        suffix = ''

    # Check to see if each of the workspace folders exists.  If not, create the
    # folder in the filesystem.
    inter_dir = os.path.join(workspace, 'intermediate')
    out_dir = os.path.join(workspace, 'output')
    for folder in [inter_dir, out_dir]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    # Determine which land cover scenarios we should run, and append the
    # appropriate suffix to the landuser_scenarios list as necessary for the
    # scenario.
    landuse_scenarios = ['cur']
    if 'landuse_fut_uri' in args:
        landuse_scenarios.append('fut')

    for scenario in landuse_scenarios:
        LOGGER.info('Starting pollination model for the %s scenario', scenario)

        # Create the args dictionary with a couple of statically defined values.
        biophysical_args = {
            'paths': {
                'workspace': workspace,
                'intermediate': inter_dir,
                'output': out_dir,
                'temp': inter_dir
            },
            'do_valuation': args['do_valuation'],
        }

        # If we're doing valuation, we also require certain other parameters to
        # be present.
        if args['do_valuation']:
            for arg in ['half_saturation', 'wild_pollination_proportion']:
                biophysical_args[arg] = args[arg]

        # Open the landcover raster
        uri = args['landuse_' + scenario +'_uri'].encode('utf-8')
        LOGGER.debug('Opening landuse raster from %s', uri)
        biophysical_args['landuse'] = uri

        # Open a Table Handler for the land use attributes table and a different
        # table handler for the Guilds table.
        LOGGER.info('Opening landuse attributes table')
        att_table_handler = fileio.TableHandler(args['landuse_attributes_uri'])
        biophysical_args['landuse_attributes'] = att_table_handler

        att_table_fields = att_table_handler.get_fieldnames()
        nesting_fields = [f[2:] for f in att_table_fields if re.match('^n_', f)]
        floral_fields = [f[2:] for f in att_table_fields if re.match('^f_', f)]
        LOGGER.debug('Parsed nesting fields: %s', nesting_fields)
        LOGGER.debug('Parsed floral fields: %s', floral_fields)

        fields_to_check = [
            (nesting_fields, 'nesting'),
            (floral_fields, 'floral'),
        ]
        for field_list, field_type in fields_to_check:
            if len(nesting_fields) == 0:
                raise MissingFields(('LULC attribute table must have '
                    ' %s fields but none were found.' % field_type))

        biophysical_args['nesting_fields'] = nesting_fields
        biophysical_args['floral_fields'] = floral_fields

        LOGGER.info('Opening guilds table')
        att_table_handler.set_field_mask('(^n_)|(^f_)', trim=2)
        guilds_handler = fileio.TableHandler(args['guilds_uri'])
        guilds_handler.set_field_mask('(^ns_)|(^fs_)', trim=3)
        biophysical_args['guilds'] = guilds_handler

        biophysical_args['nesting_fields'] = nesting_fields
        biophysical_args['floral_fields'] = floral_fields

        # Convert agricultural classes (a space-separated list of ints) into a
        # list of ints.  If the user has not provided a string list of ints,
        # then use an empty list instead.
        LOGGER.info('Processing agricultural classes')
        try:
            # This approach will create a list with only ints, even if the user
            # has accidentally entered additional spaces.  Any other incorrect
            # input will throw a ValueError exception.
            user_ag_list = args['ag_classes'].split(' ')
            ag_class_list = [int(r) for r in user_ag_list if r != '']
        except KeyError:
            # If the 'ag_classes' key is not present in the args dictionary,
            # use an empty list in its stead.
            ag_class_list = []

        LOGGER.debug('Parsed ag classes: %s', ag_class_list)
        biophysical_args['ag_classes'] = ag_class_list

        # Defined which rasters need to be created at the global level (at the
        # top level of the model dictionary).  the global_rasters list has this
        # structure:
        #   (model_args key, raster_uri base, folder to be saved to)
        global_rasters = [
            ('foraging_total', 'frm_tot', out_dir),
            ('foraging_average', 'frm_avg', out_dir),
            ('farm_value_sum', 'frm_val_sum', inter_dir),
            ('service_value_sum', 'sup_val_sum', out_dir),
            ('abundance_total', 'sup_tot', out_dir),
            ('ag_map', 'agmap', inter_dir)]

        # loop through the global rasters provided and actually create the uris,
        # saving them to the model args dictionary.
        LOGGER.info('Creating top-level raster URIs')
        for key, base, folder in global_rasters:
            raster_uri = build_uri(folder, '%s.tif' % base, [scenario, suffix])
            LOGGER.debug('%s: %s', key, raster_uri)
            biophysical_args[key] = raster_uri

        # Fetch a list of all species from the guilds table.
        species_list = [row['species'] for row in guilds_handler.table]

        # Make new rasters for each species.  In this list of tuples, the first
        # value of each tuple is the args dictionary key, and the second value
        # of each tuple is the raster prefix.  The third value is the folder in
        # which the created raster should exist.
        species_rasters = [
            ('nesting', 'hn', inter_dir),
            ('floral', 'hf', inter_dir),
            ('species_abundance', 'sup', inter_dir),
            ('farm_abundance', 'frm', inter_dir),
            ('farm_value', 'frm_val', inter_dir),
            ('value_abundance_ratio', 'val_sup_ratio', inter_dir),
            ('value_abundance_ratio_blur', 'val_sup_ratio_blur', inter_dir),
            ('service_value', 'sup_val', inter_dir)]

        # Loop through each species and define the necessary raster URIs, as
        # defined by the species_rasters list.
        LOGGER.info('Creating species-specific raster URIs')
        biophysical_args['species'] = {}
        for species in species_list:
            # Casting to UTF-8 so that LOGGER won't crash if species is UTF-8
            species = unicode(species, "utf-8")
            LOGGER.info('Creating rasters for %s', species)
            biophysical_args['species'][species] = {}
            for group, prefix, folder in species_rasters:
                raster_name = prefix + '_' + species + '.tif'
                raster_uri = build_uri(folder, raster_name, [scenario, suffix])
                LOGGER.debug('%s: %s', group, raster_uri)
                biophysical_args['species'][species][group] = raster_uri

        pollination_core.execute_model(biophysical_args)

        # If the user provided a farms shapefile input, aggregate the
        # biophysical species abundance data according to the farms table.
        # This aggregate score will be stored to a scenario-specific shapefile.
        if 'farms_shapefile' in args:
            LOGGER.info('Starting to aggregate by farms')
            encoding = sys.getfilesystemencoding()
            base_shapefile = args['farms_shapefile'].encode(encoding)
            shapefile_folder = build_uri(out_dir, 'farms_abundance', [scenario,
                suffix])

            # Delete the old shapefile folder if it already exists.
            try:
                shutil.rmtree(shapefile_folder)
                LOGGER.debug('Removed old farms folder at %s',
                     shapefile_folder)
            except OSError as exception:
                # The shapefile folder does not exist.  We need to make it
                # anyways, so we really don't care.
                LOGGER.debug('Exception: %s', exception)
                LOGGER.debug('Farms folder did not exist previously: %s',
                    shapefile_folder)

            # Make the shapefile folder to contain the farms shapefile.
            os.makedirs(shapefile_folder)
            LOGGER.debug('Farms shapefile will be saved to %s',
                 shapefile_folder)

            farms_file = ogr.Open(base_shapefile, 0)
            ogr_driver = ogr.GetDriverByName('ESRI Shapefile')
            farms_copy = ogr_driver.CopyDataSource(farms_file,
               shapefile_folder.encode('UTF-8'))

            guilds_crop_fields = [r for r in guilds_handler.get_fieldnames() if
                r[0:4] == 'crp_']
            LOGGER.debug('Crop fields from Guilds table:%s', guilds_crop_fields)

            # Create a new field for each crop in the copied shapefile.
            LOGGER.debug('Starting to create necessary crop sum fields in '
                'the copied shapefile')
            crops = []
            for crop in guilds_crop_fields:
                crop_name = crop[4:]  # Trim off the 'crp_'
                abbrev_crop_name = crop_name[:4]
                field_name = abbrev_crop_name + '_sum'

                # Only create a new field if it doesn't exist already.
                field_index = farms_copy.GetLayer(0).GetFeature(0).GetFieldIndex(field_name)
                if field_index == -1:
                    new_field = ogr.FieldDefn(field_name, ogr.OFTReal)
                    farms_copy.GetLayer(0).CreateField(new_field)
                    LOGGER.debug('Created crop sum field "%s" for "%s"',
                        field_name, crop)

                # Regardless of whether it needs to be created, we still need to
                # keep track of it for later on, when the value of the cell will
                # be set.
                crops.append((crop, field_name))
            LOGGER.debug('Crops/fields: %s', crops)

            LOGGER.info('Starting to aggregate by available farm sites')
            farms_layer = farms_copy.GetLayer(0)
            for farm_site in farms_layer:
                for crop, fieldname in crops:
                    crop_is_present = farm_site.GetField(crop)
                    LOGGER.debug('Crop is present: %s', crop_is_present)
                    LOGGER.debug('Resetting the crop sum to 0')
                    LOGGER.debug('Resetting the visiting species list to []')
                    crop_sum = 0
                    visiting_species = []

                    # The field value is often stored as either 0 or
                    # some other value, but not necessarily 1.  So here,
                    # I need to compare the value against 0, not vs. 1.
                    if crop_is_present != 0:
                        LOGGER.info('Processing crop %s', crop)
                        for species in guilds_handler.table:
                            species_name = species['species']
                            LOGGER.info('Processing species %s', species_name)

                            species_crop = species[crop]
                            if species_crop == 1:
                                visiting_species.append(species_name)
                                supply_uri = biophysical_args['species'][species_name]['species_abundance']
                                LOGGER.debug('Supply raster URI="%s"', supply_uri)
                                pixel_value = get_point(supply_uri, farm_site)[0]
                                LOGGER.debug(('Crop="%s", species="%s"'
                                    'pixel_value=%s'), crop, species_name,
                                    pixel_value)
                                crop_sum += pixel_value

                    LOGGER.info('Sum across %s for crop "%s": %s',
                        visiting_species, crop, crop_sum)

                    # Actually set the correct field value in the copied farms
                    # shapefile using the crop sum.
                    field_index = farm_site.GetFieldIndex(fieldname)
                    farm_site.SetField(field_index, crop_sum)
                    farms_layer.SetFeature(farm_site)
                    LOGGER.info('Sum saved to the shapefile.')

def build_uri(directory, basename, suffix=[]):
    """Take the input directory and basename, inserting the provided suffixes
        just before the file extension.  Each string in the suffix list will be
        underscore-separated.

        directory - a python string folder path
        basename - a python string filename
        suffix='' - a python list of python strings to be separated by
            underscores and concatenated with the basename just before the
            extension.

        returns a python string of the complete path with the correct
        filename."""

    file_base, extension = os.path.splitext(basename)

    # If a siffix is provided, we want the suffix to be prepended with an
    # underscore, so as to separate the file basename and the suffix.  If a
    # suffix is an empty string, ignore it.
    if len(suffix) > 0:
        suffix = '_' + '_'.join([s for s in suffix if s != ''])

    new_filepath = file_base + suffix + extension
    return os.path.join(directory, new_filepath)


def get_point(raster_uri, point):
    """Get the value of the point at the raster located at raster_uri.  This
    operation is completed without using numpy.

        raster_uri - a URI that GDAL can open.
        point - an OGR Feature to use to extract a raster value.

    Returns the value at the point on the raster."""

    raster = gdal.Open(raster_uri)
    raster_gt = raster.GetGeoTransform()
    raster_band = raster.GetRasterBand(1)

    geometry = point.GetGeometryRef()
    geom_x, geom_y = geometry.GetX(), geometry.GetY()  # Coord. in map units

    # Convert from map to pixel coordinates
    pixel_x = int((geom_x - raster_gt[0]) / (raster_gt[1]))
    pixel_y = int((geom_y - raster_gt[3]) / (raster_gt[5]))
    structval = raster_band.ReadRaster(pixel_x, pixel_y, 1, 1,
        buf_type=gdal.GDT_Float32)
    intval = struct.unpack('f', structval)

    return intval
