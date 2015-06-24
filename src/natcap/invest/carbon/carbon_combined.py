"""Integrated carbon model with biophysical and valuation components."""

import collections
import math
import logging
import os
from datetime import datetime

from natcap.invest.carbon import carbon_biophysical
from natcap.invest.carbon import carbon_valuation
from natcap.invest.carbon import carbon_utils
from natcap.invest.reporting import html

logging.basicConfig(format='%(asctime)s %(name)-18s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.carbon.combined')

def execute(args):
    execute_30(**args)

def execute_30(**args):
    """Run the carbon model.

    This can include the biophysical model, the valuation model, or both.

    Args:
        workspace_dir (string): a uri to the directory that will write output
            and other temporary files during calculation. (required)
        suffix (string): a string to append to any output file name (optional)

        do_biophysical (boolean): whether to run the biophysical model
        lulc_cur_uri (string): a uri to a GDAL raster dataset (required)
        lulc_cur_year (int): An integer representing the year of lulc_cur
            used in HWP calculation (required if args contains a
            'hwp_cur_shape_uri', or 'hwp_fut_shape_uri' key)
        lulc_fut_uri (string): a uri to a GDAL raster dataset (optional
            if calculating sequestration)
        lulc_redd_uri (string): a uri to a GDAL raster dataset that represents
            land cover data for the REDD policy scenario (optional).
        lulc_fut_year (int): An integer representing the year of  lulc_fut
            used in HWP calculation (required if args contains a
            'hwp_fut_shape_uri' key)
        carbon_pools_uri (string): a uri to a CSV or DBF dataset mapping carbon
            storage density to the lulc classifications specified in the
            lulc rasters. (required if 'do_uncertainty' is false)
        hwp_cur_shape_uri (String): Current shapefile uri for harvested wood
            calculation (optional, include if calculating current lulc hwp)
        hwp_fut_shape_uri (String): Future shapefile uri for harvested wood
            calculation (optional, include if calculating future lulc hwp)
        do_uncertainty (boolean): a boolean that indicates whether we should do
            uncertainty analysis. Defaults to False if not present.
        carbon_pools_uncertain_uri (string): as above, but has probability
            distribution data for each lulc type rather than point estimates.
            (required if 'do_uncertainty' is true)
        confidence_threshold (float): a number between 0 and 100 that indicates
            the minimum threshold for which we should highlight regions in the
            output raster. (required if 'do_uncertainty' is True)
        sequest_uri (string): uri to a GDAL raster dataset describing the
            amount of carbon sequestered.
        yr_cur (int): the year at which the sequestration measurement started
        yr_fut (int): the year at which the sequestration measurement ended
        do_valuation (boolean): whether to run the valuation model
        carbon_price_units (string): indicates whether the price is
            in terms of carbon or carbon dioxide. Can value either as
            'Carbon (C)' or 'Carbon Dioxide (CO2)'.
        V (string): value of a sequestered ton of carbon or carbon dioxide in
        dollars per metric ton
        r (int): the market discount rate in terms of a percentage
        c (float): the annual rate of change in the price of carbon


    Example Args Dictionary::

        {
            'workspace_dir': 'path/to/workspace_dir/',
            'suffix': '_results',
            'do_biophysical': True,
            'lulc_cur_uri': 'path/to/lulc_cur',
            'lulc_cur_year': 2014,
            'lulc_fut_uri': 'path/to/lulc_fut',
            'lulc_redd_uri': 'path/to/lulc_redd',
            'lulc_fut_year': 2025,
            'carbon_pools_uri': 'path/to/carbon_pools',
            'hwp_cur_shape_uri': 'path/to/hwp_cur_shape',
            'hwp_fut_shape_uri': 'path/to/hwp_fut_shape',
            'do_uncertainty': True,
            'carbon_pools_uncertain_uri': 'path/to/carbon_pools_uncertain',
            'confidence_threshold': 50.0,
            'sequest_uri': 'path/to/sequest_uri',
            'yr_cur': 2014,
            'yr_fut': 2025,
            'do_valuation': True,
            'carbon_price_units':, 'Carbon (C)',
            'V': 43.0,
            'r': 7,
            'c': 0,
        }

    Returns:
        outputs (dictionary): contains names of all output files

    """
    if not args['do_biophysical'] and not args['do_valuation']:
        raise Exception(
            'Neither biophysical nor valuation model selected. '
            'Nothing left to do. Exiting.')

    if args['do_biophysical']:
        LOGGER.info('Executing biophysical model.')
        biophysical_outputs = carbon_biophysical.execute(args)
    else:
        biophysical_outputs = None

        # We can't do uncertainty analysis if only the valuation model is run.
        args['do_uncertainty'] = False

    if args['do_valuation']:
        if not args['do_biophysical'] and not args.get('sequest_uri'):
            raise Exception(
                'In order to perform valuation, you must either run the '
                'biophysical model, or provide a sequestration raster '
                'mapping carbon sequestration for a landscape. Neither '
                'was provided in this case, so valuation cannot run.')
        LOGGER.info('Executing valuation model.')
        valuation_args = _package_valuation_args(args, biophysical_outputs)
        valuation_outputs = carbon_valuation.execute(valuation_args)
    else:
        valuation_outputs = None

    _create_HTML_report(args, biophysical_outputs, valuation_outputs)

def _package_valuation_args(args, biophysical_outputs):
    if not biophysical_outputs:
        return args

    if 'sequest_fut' not in biophysical_outputs:
        raise Exception(
            'Both biophysical and valuation models were requested, '
            'but sequestration was not calculated. In order to calculate '
            'valuation data, please run the biophysical model with '
            'sequestration analysis enabled. This requires a future LULC map '
            'in addition to the current LULC map.')

    args['sequest_uri'] = biophysical_outputs['sequest_fut']
    args['yr_cur'] = args['lulc_cur_year']
    args['yr_fut'] = args['lulc_fut_year']

    if args['yr_cur'] >= args['yr_fut']:
        raise Exception(
            'The current year must be earlier than the future year. '
            'The values for current/future year are: %d/%d' %
            (args['yr_cur'], args['yr_fut']))

    biophysical_to_valuation = {
        'uncertainty': 'uncertainty_data',
        'sequest_redd': 'sequest_redd_uri',
        'conf_fut': 'conf_uri',
        'conf_redd': 'conf_redd_uri'
        }

    for biophysical_key, valuation_key in biophysical_to_valuation.items():
        try:
            args[valuation_key] = biophysical_outputs[biophysical_key]
        except KeyError:
            continue

    return args

def _create_HTML_report(args, biophysical_outputs, valuation_outputs):
    html_uri = os.path.join(
        args['workspace_dir'], 'output',
        'summary%s.html' % carbon_utils.make_suffix(args))

    doc = html.HTMLDocument(html_uri, 'Carbon Results',
                            'InVEST Carbon Model Results')

    doc.write_paragraph(_make_report_intro(args))

    doc.insert_table_of_contents()

    if args['do_biophysical']:
        doc.write_header('Biophysical Results')
        doc.add(_make_biophysical_table(biophysical_outputs))
        if 'uncertainty' in biophysical_outputs:
            doc.write_header('Uncertainty Results', level=3)
            for paragraph in _make_biophysical_uncertainty_intro():
                doc.write_paragraph(paragraph)
            doc.add(_make_biophysical_uncertainty_table(
                    biophysical_outputs['uncertainty']))

    if args['do_valuation']:
        doc.write_header('Valuation Results')
        for paragraph in _make_valuation_intro(args):
            doc.write_paragraph(paragraph)
        for table in _make_valuation_tables(valuation_outputs):
            doc.add(table)
        if 'uncertainty_data' in valuation_outputs:
            doc.write_header('Uncertainty Results', level=3)
            for paragraph in _make_valuation_uncertainty_intro():
                doc.write_paragraph(paragraph)
            doc.add(_make_valuation_uncertainty_table(
                    valuation_outputs['uncertainty_data']))

    doc.write_header('Output Files')
    doc.write_paragraph(
        'This run of the carbon model produced the following output files.')
    doc.add(_make_outfile_table(
            args, biophysical_outputs, valuation_outputs, html_uri))

    doc.flush()

def _make_report_intro(args):
    models = []
    for model in 'biophysical', 'valuation':
        if args['do_%s' % model]:
            models.append(model)

    return ('This document summarizes the results from running the InVEST '
            'carbon model. This run of the model involved the %s %s.' %
            (' and '.join(models),
             'model' if len(models) == 1 else 'models'))

def _make_biophysical_uncertainty_intro():
    return [
        'This data was computed by doing a Monte Carlo '
        'simulation, which involved %d runs of the model.' %
        carbon_biophysical.NUM_MONTE_CARLO_RUNS,
        'For each run of the simulation, the amount of carbon '
        'per grid cell for each LULC type was independently sampled '
        'from the normal distribution given in the input carbon pools. '
        'Given this set of carbon pools, the model computed the amount of '
        'carbon in each scenario, and computed sequestration by subtracting '
        'the carbon storage in different scenarios. ',
        'Results across all Monte Carlo simulation runs were '
        'analyzed to produce the following mean and standard deviation data.',
        'All uncertainty analysis in this model assumes that true carbon pool '
        'values for different LULC types are independently distributed, '
        'with no systematic bias. If there is systematic bias in the carbon '
        'pool estimates, then actual standard deviations for results may be '
        'larger than reported in the following table.']

def _make_biophysical_uncertainty_table(uncertainty_results):
    table = html.Table(id='biophysical_uncertainty')
    table.add_two_level_header(
        outer_headers=['Total carbon (Mg of carbon)',
                       'Sequestered carbon (compared to current scenario)'
                       '<br>(Mg of carbon)'],
        inner_headers=['Mean', 'Standard deviation'],
        row_id_header='Scenario')

    for scenario in ['cur', 'fut', 'redd']:
        if scenario not in uncertainty_results:
            continue

        row = [_make_scenario_name(scenario, 'redd' in uncertainty_results)]
        row += uncertainty_results[scenario]

        if scenario == 'cur':
            row += ['n/a', 'n/a']
        else:
            row += uncertainty_results['sequest_%s' % scenario]

        table.add_row(row)

    return table

def _make_biophysical_table(biophysical_outputs):
    do_uncertainty = 'uncertainty' in biophysical_outputs

    table = html.Table(id='biophysical_table')
    headers = ['Scenario', 'Total carbon<br>(Mg of carbon)',
               'Sequestered carbon<br>(compared to current scenario)'
               '<br>(Mg of carbon)']

    table.add_row(headers, is_header=True)

    for scenario in ['cur', 'fut', 'redd']:
        total_carbon_key = 'tot_C_%s' % scenario
        if total_carbon_key not in biophysical_outputs:
            continue

        row = []
        row.append(
            _make_scenario_name(scenario, 'tot_C_redd' in biophysical_outputs))

        # Append total carbon.
        row.append(carbon_utils.sum_pixel_values_from_uri(
                biophysical_outputs[total_carbon_key]))

        # Append sequestration.
        sequest_key = 'sequest_%s' % scenario
        if sequest_key in biophysical_outputs:
            row.append(carbon_utils.sum_pixel_values_from_uri(
                    biophysical_outputs[sequest_key]))
        else:
            row.append('n/a')

        table.add_row(row)

    return table

def _make_valuation_tables(valuation_outputs):
    scenario_results = {}
    change_table = html.Table(id='change_table')
    change_table.add_row(["Scenario",
                          "Sequestered carbon<br>(Mg of carbon)",
                          "Net present value<br>(USD)"],
                         is_header=True)

    for scenario_type in ['base', 'redd']:
        try:
            sequest_uri = valuation_outputs['sequest_%s' % scenario_type]
        except KeyError:
            # We may not be doing REDD analysis.
            continue

        scenario_name = _make_scenario_name(
            scenario_type, 'sequest_redd' in valuation_outputs)

        total_seq = carbon_utils.sum_pixel_values_from_uri(sequest_uri)
        total_val = carbon_utils.sum_pixel_values_from_uri(
            valuation_outputs['%s_val' % scenario_type])
        scenario_results[scenario_type] = (total_seq, total_val)
        change_table.add_row([scenario_name, total_seq, total_val])

        try:
            seq_mask_uri = valuation_outputs['%s_seq_mask' % scenario_type]
            val_mask_uri = valuation_outputs['%s_val_mask' % scenario_type]
        except KeyError:
            # We may not have confidence-masking data.
            continue

        # Compute output for confidence-masked data.
        masked_seq = carbon_utils.sum_pixel_values_from_uri(seq_mask_uri)
        masked_val = carbon_utils.sum_pixel_values_from_uri(val_mask_uri)
        scenario_results['%s_mask' % scenario_type] = (masked_seq, masked_val)
        change_table.add_row(['%s (confident cells only)' % scenario_name,
                              masked_seq,
                              masked_val])

    yield change_table

    # If REDD scenario analysis is enabled, write the table
    # comparing the baseline and REDD scenarios.
    if 'base' in scenario_results and 'redd' in scenario_results:
        comparison_table = html.Table(id='comparison_table')
        comparison_table.add_row(
            ["Scenario Comparison",
             "Difference in carbon stocks<br>(Mg of carbon)",
             "Difference in net present value<br>(USD)"],
            is_header=True)

        # Add a row with the difference in carbon and in value.
        base_results = scenario_results['base']
        redd_results = scenario_results['redd']
        comparison_table.add_row(
            ['%s vs %s' % (_make_scenario_name('redd'),
                           _make_scenario_name('base')),
             redd_results[0] - base_results[0],
             redd_results[1] - base_results[1]
             ])

        if 'base_mask' in scenario_results and 'redd_mask' in scenario_results:
            # Add a row with the difference in carbon and in value for the
            # uncertainty-masked scenario.
            base_mask_results = scenario_results['base_mask']
            redd_mask_results = scenario_results['redd_mask']
            comparison_table.add_row(
                ['%s vs %s (confident cells only)'
                 % (_make_scenario_name('redd'),
                    _make_scenario_name('base')),
                 redd_mask_results[0] - base_mask_results[0],
                 redd_mask_results[1] - base_mask_results[1]
                 ])

        yield comparison_table


def _make_valuation_uncertainty_intro():
    return [
        'These results were computed by using the uncertainty data from the '
        'Monte Carlo simulation in the biophysical model.'
        ]


def _make_valuation_uncertainty_table(uncertainty_data):
    table = html.Table(id='valuation_uncertainty')

    table.add_two_level_header(
        outer_headers=['Sequestered carbon (Mg of carbon)',
                       'Net present value (USD)'],
        inner_headers=['Mean', 'Standard Deviation'],
        row_id_header='Scenario')

    for fut_type in ['fut', 'redd']:
        if fut_type not in uncertainty_data:
            continue

        scenario_data = uncertainty_data[fut_type]
        row = [_make_scenario_name(fut_type, 'redd' in uncertainty_data)]
        row += scenario_data['sequest']
        row += scenario_data['value']
        table.add_row(row)

    return table


def _make_valuation_intro(args):
    intro = [
        '<strong>Positive values</strong> in this table indicate that '
        'carbon storage increased. In this case, the positive Net Present '
        'Value represents the value of the sequestered carbon.',
        '<strong>Negative values</strong> indicate that carbon storage '
        'decreased. In this case, the negative Net Present Value represents '
        'the cost of carbon emission.'
        ]

    if args['do_uncertainty']:
        intro.append(
            'Entries in the table with the label "confident cells only" '
            'represent results for sequestration and value if we consider '
            'sequestration that occurs only in those cells where we are '
            'confident that carbon storage will either increase or decrease.')

    return intro


def _make_outfile_table(args, biophysical_outputs, valuation_outputs, html_uri):
    table = html.Table(id='outfile_table')
    table.add_row(['Filename', 'Description'], is_header=True)

    descriptions = collections.OrderedDict()

    if biophysical_outputs:
        descriptions.update(_make_biophysical_outfile_descriptions(
                biophysical_outputs, args))

    if valuation_outputs:
        descriptions.update(_make_valuation_outfile_descriptions(
                valuation_outputs))

    html_filename = os.path.basename(html_uri)
    descriptions[html_filename] = 'This summary file.' # dude, that's so meta

    for filename, description in sorted(descriptions.items()):
        table.add_row([filename, description])

    return table


def _make_biophysical_outfile_descriptions(outfile_uris, args):
    '''Return a dict with descriptions of biophysical outfiles.'''

    def name(scenario_type):
        return _make_scenario_name(scenario_type,
                                  do_redd=('tot_C_redd' in outfile_uris),
                                  capitalize=False)

    def total_carbon_description(scenario_type):
        return ('Maps the total carbon stored in the %s scenario, in '
                'Mg per grid cell.') % name(scenario_type)

    def sequest_description(scenario_type):
        return ('Maps the sequestered carbon in the %s scenario, relative to '
                'the %s scenario, in Mg per grid cell.') % (
            name(scenario_type), name('cur'))

    def conf_description(scenario_type):
        return ('Maps confident areas for carbon sequestration and emissions '
                'between the current scenario and the %s scenario. '
                'Grid cells where we are at least %.2f%% confident that '
                'carbon storage will increase have a value of 1. Grid cells '
                'where we are at least %.2f%% confident that carbon storage will '
                'decrease have a value of -1. Grid cells with a value of 0 '
                'indicate regions where we are not %.2f%% confident that carbon '
                'storage will either increase or decrease.') % (
            tuple([name(scenario_type)] + [args['confidence_threshold']] * 3))

    file_key_to_func = {
        'tot_C_%s': total_carbon_description,
        'sequest_%s': sequest_description,
        'conf_%s': conf_description
        }

    return _make_outfile_descriptions(outfile_uris, ['cur', 'fut', 'redd'],
                                     file_key_to_func)

def _make_valuation_outfile_descriptions(outfile_uris):
    '''Return a dict with descriptions of valuation outfiles.'''

    def name(scenario_type):
        return _make_scenario_name(scenario_type,
                                  do_redd=('sequest_redd' in outfile_uris),
                                  capitalize=False)

    def value_file_description(scenario_type):
        return ('Maps the economic value of carbon sequestered between the '
                'current and %s scenarios, with values in dollars per grid '
                'cell.') % name(scenario_type)

    def value_mask_file_description(scenario_type):
        return ('Maps the economic value of carbon sequestered between the '
                'current and %s scenarios, but only for cells where we are '
                'confident that carbon storage will either increase or '
                'decrease.') % name(scenario_type)

    def carbon_mask_file_description(scenario_type):
        return ('Maps the increase in carbon stored between the current and '
                '%s scenarios, in Mg per grid cell, but only for cells where '
                ' we are confident that carbon storage will either increase or '
                'decrease.') % name(scenario_type)

    file_key_to_func = {
        '%s_val': value_file_description,
        '%s_seq_mask': carbon_mask_file_description,
        '%s_val_mask': value_mask_file_description
        }

    return _make_outfile_descriptions(outfile_uris, ['base', 'redd'],
                                     file_key_to_func)


def _make_outfile_descriptions(outfile_uris, scenarios, file_key_to_func):
    descriptions = collections.OrderedDict()
    for scenario_type in scenarios:
        for file_key, description_func in file_key_to_func.items():
            try:
                uri = outfile_uris[file_key % scenario_type]
            except KeyError:
                continue

            filename = os.path.basename(uri)
            descriptions[filename] = description_func(scenario_type)

    return descriptions


def _make_scenario_name(scenario, do_redd=True, capitalize=True):
    names = {
        'cur': 'current',
        'fut': 'baseline' if do_redd else 'future',
        'redd': 'REDD policy'
        }
    names['base'] = names['fut']
    name = names[scenario]
    if capitalize:
        return name[0].upper() + name[1:]
    return name
