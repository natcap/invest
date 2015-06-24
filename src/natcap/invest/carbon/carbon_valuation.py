"""InVEST valuation interface module.  Informally known as the URI level."""

import collections
import os
import logging

from osgeo import gdal

from natcap.invest.carbon import carbon_utils
import pygeoprocessing.geoprocessing

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('natcap.invest.carbon.valuation')

def execute(args):
    return execute_30(**args)

def execute_30(**args):
    """This function calculates carbon sequestration valuation.

        args - a python dictionary with at the following *required* entries:

        args['workspace_dir'] - a uri to the directory that will write output
            and other temporary files during calculation. (required)
        args['suffix'] - a string to append to any output file name (optional)
        args['sequest_uri'] - is a uri to a GDAL raster dataset describing the
            amount of carbon sequestered (baseline scenario, if this is REDD)
        args['sequest_redd_uri'] (optional) - uri to the raster dataset for
            sequestration under the REDD policy scenario
        args['conf_uri'] (optional) - uri to the raster dataset indicating
            confident pixels for sequestration or emission
        args['conf_redd_uri'] (optional) - as above, but for the REDD scenario
        args['carbon_price_units'] - a string indicating whether the price is
            in terms of carbon or carbon dioxide. Can value either as
            'Carbon (C)' or 'Carbon Dioxide (CO2)'.
        args['V'] - value of a sequestered ton of carbon or carbon dioxide in
            dollars per metric ton
        args['r'] - the market discount rate in terms of a percentage
        args['c'] - the annual rate of change in the price of carbon
        args['yr_cur'] - the year at which the sequestration measurement
            started
        args['yr_fut'] - the year at which the sequestration measurement ended

        returns a dict with output file URIs."""

    output_directory = carbon_utils.setup_dirs(args['workspace_dir'], 'output')

    if args['carbon_price_units'] == 'Carbon Dioxide (CO2)':
        #Convert to price per unit of Carbon do this by dividing
        #the atomic mass of CO2 (15.9994*2+12.0107) by the atomic
        #mass of 12.0107.  Values gotten from the periodic table of
        #elements.
        args['V'] *= (15.9994*2+12.0107)/12.0107

    LOGGER.info('Constructing valuation formula.')
    n = args['yr_fut'] - args['yr_cur'] - 1
    ratio = 1.0 / ((1 + args['r'] / 100.0) * (1 + args['c'] / 100.0))
    valuation_constant = args['V'] / (args['yr_fut'] - args['yr_cur']) * \
        (1.0 - ratio ** (n + 1)) / (1.0 - ratio)

    nodata_out = -1.0e10

    outputs = _make_outfile_uris(output_directory, args)

    conf_uris = {}
    if args.get('conf_uri'):
        conf_uris['base'] = args['conf_uri']
    if args.get('conf_redd_uri'):
        conf_uris['redd'] = args['conf_redd_uri']

    for scenario_type in ['base', 'redd']:
        try:
            sequest_uri = outputs['sequest_%s' % scenario_type]
        except KeyError:
            # REDD analysis might not be enabled, so just keep going.
            continue

        LOGGER.info('Beginning valuation of %s scenario.', scenario_type)

        sequest_nodata = pygeoprocessing.geoprocessing.get_nodata_from_uri(sequest_uri)

        def value_op(sequest):
            if sequest == sequest_nodata:
                return nodata_out
            return sequest * valuation_constant

        pixel_size_out = pygeoprocessing.geoprocessing.get_cell_size_from_uri(sequest_uri)
        pygeoprocessing.geoprocessing.vectorize_datasets(
            [sequest_uri], value_op, outputs['%s_val' % scenario_type],
            gdal.GDT_Float32, nodata_out, pixel_size_out, "intersection")


        if scenario_type in conf_uris:
            LOGGER.info('Creating masked rasters for %s scenario.', scenario_type)
            # Produce a raster for sequestration, masking out uncertain areas.
            _create_masked_raster(sequest_uri, conf_uris[scenario_type],
                                  outputs['%s_seq_mask' % scenario_type])

            # Produce a raster for value sequestration,
            # again masking out uncertain areas.
            _create_masked_raster(
                outputs['%s_val' % scenario_type],
                conf_uris[scenario_type],
                outputs['%s_val_mask' % scenario_type])

    if 'uncertainty_data' in args:
        uncertainty_data = _compute_uncertainty_data(
            args['uncertainty_data'], valuation_constant)
        if uncertainty_data:
            outputs['uncertainty_data'] = uncertainty_data

    return outputs


def _make_outfile_uris(output_directory, args):
    '''Return a dict with uris for outfiles.'''
    file_suffix = carbon_utils.make_suffix(args)

    def outfile_uri(prefix, scenario_type='', filetype='tif'):
        '''Create the URI for the appropriate output file.'''
        if not args.get('sequest_redd_uri'):
            # We're not doing REDD analysis, so don't append anything,
            # since there's only one scenario.
            scenario_type = ''
        elif scenario_type:
            scenario_type = '_' + scenario_type
        filename = '%s%s%s.%s' % (prefix, scenario_type, file_suffix, filetype)
        return os.path.join(output_directory, filename)

    outfile_uris = collections.OrderedDict()

    # Value sequestration for base scenario.
    outfile_uris['base_val'] = outfile_uri('value_seq', 'base')

    # Confidence-masked rasters for base scenario.
    if args.get('conf_uri'):
        outfile_uris['base_seq_mask'] = outfile_uri('seq_mask', 'base')
        outfile_uris['base_val_mask'] = outfile_uri('val_mask', 'base')

    # Outputs for REDD scenario.
    if args.get('sequest_redd_uri'):
        # Value sequestration.
        outfile_uris['redd_val'] = outfile_uri('value_seq', 'redd')

        # Confidence-masked rasters for REDD scenario.
        if args.get('conf_redd_uri'):
            outfile_uris['redd_seq_mask'] = outfile_uri('seq_mask', 'redd')
            outfile_uris['redd_val_mask'] = outfile_uri('val_mask', 'redd')

    # These sequestration rasters are actually input files (not output files),
    # but it's convenient to have them in this dictionary.
    outfile_uris['sequest_base'] = args['sequest_uri']
    if args.get('sequest_redd_uri'):
        outfile_uris['sequest_redd'] = args['sequest_redd_uri']

    return outfile_uris


def _create_masked_raster(orig_uri, mask_uri, result_uri):
    '''Creates a raster at result_uri with some areas masked out.

    orig_uri -- uri of the original raster
    mask_uri -- uri of the raster to use as a mask
    result_uri -- uri at which the new raster should be created

    Masked data in the result file is denoted as no data (not necessarily zero).

    Data is masked out at locations where mask_uri is 0 or no data.
    '''
    nodata_orig = pygeoprocessing.geoprocessing.get_nodata_from_uri(orig_uri)
    nodata_mask = pygeoprocessing.geoprocessing.get_nodata_from_uri(mask_uri)
    def mask_op(orig_val, mask_val):
        '''Return orig_val unless mask_val indicates uncertainty.'''
        if mask_val == 0 or mask_val == nodata_mask:
            return nodata_orig
        return orig_val

    pixel_size = pygeoprocessing.geoprocessing.get_cell_size_from_uri(orig_uri)
    pygeoprocessing.geoprocessing.vectorize_datasets(
        [orig_uri, mask_uri], mask_op, result_uri, gdal.GDT_Float32,
        nodata_orig, pixel_size, 'intersection', dataset_to_align_index=0)

def _compute_uncertainty_data(biophysical_uncertainty_data, valuation_const):
    """Computes mean and standard deviation for sequestration value."""
    LOGGER.info('Computing uncertainty data.')
    results = {}
    for fut_type in ['fut', 'redd']:
        try:
            # Get the tuple with mean and standard deviation for sequestration.
            sequest = biophysical_uncertainty_data['sequest_%s' % fut_type]
        except KeyError:
            continue

        results[fut_type] = {}

        # Note sequestered carbon (mean and std dev).
        results[fut_type]['sequest'] = sequest

        # Compute the value of sequestered carbon (mean and std dev).
        results[fut_type]['value'] = tuple(carbon * valuation_const
                                           for carbon in sequest)
    return results
