"""CBC Model IO Utilities."""

import csv
import os
import pprint as pp

import gdal
import pygeoprocessing as pygeo

from natcap.invest.coastal_blue_carbon.classes.raster import Raster
from natcap.invest.coastal_blue_carbon.global_variables import *


def get_inputs(args):
    """Create and validate derivative variables from args dictionary.

    Example Returns::

        vars_dict = {
            # ... args ...

            'outputs_dir': 'path/to/outputs_dir/',
            'lulc_lookup_dict': {},
            'lulc_transition_dict': {},
            'carbon_pool_initial_dict': {},
            'lulc_lookup_dict': {},
            'lulc_to_code_dict': {},
            'code_to_lulc_dict': {},
            'carbon_pool_transient_dict': {},
        }
    """
    vars_dict = args.copy()
    try:
        vars_dict['results_suffix']
    except:
        vars_dict['results_suffix'] = ''

    output_dir_name = 'outputs_core'
    if vars_dict['results_suffix'] != '':
        output_dir_name = output_dir_name + '_' + vars_dict['results_suffix']
    outputs_dir = os.path.join(vars_dict['workspace_dir'], output_dir_name)

    pygeo.geoprocessing.create_directories(
        [vars_dict['workspace_dir'], outputs_dir])

    vars_dict['outputs_dir'] = outputs_dir

    vars_dict['lulc_lookup_dict'] = pygeo.geoprocessing.get_lookup_from_csv(
        args['lulc_lookup_uri'], 'code')

    # Parse LULC Transition CSV (Carbon Direction and Relative Magnitude)
    lulc_transition_dict = pygeo.geoprocessing.get_lookup_from_csv(
        args['lulc_transition_uri'], 'lulc-class')
    lulc_transition_dict['undefined'] = {
        u'lulc-class': u'undefined', u'undefined': u'undefined'}
    item_list = []
    for item in lulc_transition_dict.items():
        # allows a second-column legend to be appended to input csv
        if item[0] is u'':
            continue
        del item[1]['lulc-class']
        item_list.append(item)
    vars_dict['lulc_transition_dict'] = dict(item_list)

    # LULC Lookup
    lulc_lookup_dict = vars_dict['lulc_lookup_dict']
    lulc_lookup_dict[NODATA_INT] = {
        u'code': NODATA_INT, u'lulc-class': u'undefined'}

    code_to_lulc_dict = {key: lulc_lookup_dict[key][
        'lulc-class'] for key in lulc_lookup_dict.keys()}
    vars_dict['lulc_to_code_dict'] = {
        v: k for k, v in code_to_lulc_dict.items()}
    vars_dict['code_to_lulc_dict'] = code_to_lulc_dict

    # Carbon Pool Initial
    vars_dict['carbon_pool_initial_dict'] = \
        pygeo.geoprocessing.get_lookup_from_csv(
            args['carbon_pool_initial_uri'], 'lulc-class')
    nan_dict = {
        u'biomass': NODATA_FLOAT,
        u'litter': NODATA_FLOAT,
        u'soil': NODATA_FLOAT,
        u'lulc-class': u'undefined'
    }
    vars_dict['carbon_pool_initial_dict']['undefined'] = nan_dict

    # Carbon Pool Transient
    vars_dict['carbon_pool_transient_dict'] = \
        _create_transient_dict(args['carbon_pool_transient_uri'])
    nan_dict = {
        u'half-life': NODATA_FLOAT,
        u'high-impact-disturb': NODATA_FLOAT,
        u'low-impact-disturb': NODATA_FLOAT,
        u'med-impact-disturb': NODATA_FLOAT,
        u'lulc-class': u'undefined',
        u'pool': u'biomass',
        u'yearly_accumulation': NODATA_FLOAT,
        u'undefined': NODATA_FLOAT
    }
    vars_dict['carbon_pool_transient_dict'][(u'undefined', u'biomass')] = nan_dict
    nan_dict['pool'] = 'soil'
    vars_dict['carbon_pool_transient_dict'][(u'undefined', u'soil')] = nan_dict

    # Str --> Int for Snapshot Years List
    vars_dict['lulc_snapshot_years_list'] = [
        int(i) for i in args['lulc_snapshot_years_list']]
    if vars_dict['analysis_year'] != '':
        vars_dict['analysis_year'] = int(vars_dict['analysis_year'])

    # Set LULC_Snapshots' NODATA to Program Standard
    lulc_snapshot_list = []
    for i in vars_dict['lulc_snapshot_list']:
        lulc_snapshot_list.append(Raster.from_file(
            i).set_datatype_and_nodata(gdal.GDT_Int32, NODATA_INT).uri)
    vars_dict['lulc_snapshot_list'] = lulc_snapshot_list

    # Fetch yearly_carbon_price_dict
    if args['do_economic_analysis']:
        vars_dict['yearly_carbon_price_dict'] = \
            _get_yearly_carbon_price_dict(vars_dict)
    else:
        vars_dict['yearly_carbon_price_dict'] = None

    return vars_dict


def _create_transient_dict(carbon_pool_transient_uri):
    """Create dictionary of transient variables for carbon pools.

    Args:
        carbon_pool_transient_uri (string): path to carbon pool transient
            variables csv file.

    Returns:
        carbon_pool_transient_dict (dict): dictionary of carbon pool transient
            variables.
    """
    def to_float(x):
        try:
            return float(x)
        except ValueError:
            return x

    lines = []
    with open(carbon_pool_transient_uri, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = [to_float(el) for el in row]
            lines.append(row)

    lulc_class_idx = lines[0].index('lulc-class')
    pool_idx = lines[0].index('pool')

    lines_transpose = zip(*lines[1:])
    combo = zip(lines_transpose[lulc_class_idx], lines_transpose[pool_idx])
    header = lines[0]

    carbon_pool_transient_dict = {}
    for pair in combo:
        carbon_pool_transient_dict[pair] = {}

    for line in lines[1:]:
        el_dict = dict(zip(header, line))
        lulc = el_dict['lulc-class']
        pool = el_dict['pool']
        carbon_pool_transient_dict[(lulc, pool)] = dict(zip(header, line))

    return carbon_pool_transient_dict


def _get_yearly_carbon_price_dict(vars_dict):
    """Return dictionary of discounted prices for each year.

    Args:
        discount_rate (float)
        lulc_snapshot_years_list (list)
        analysis_year (int)
        interest_rate (float)
        price (float)
        do_price_table (boolean)
        price_table_uri (string)

    Returns:
        yearly_carbon_price_dict (dictionary)
    """
    discount_rate = float(vars_dict['discount_rate'])
    start_year = int(vars_dict['lulc_snapshot_years_list'][0])
    end_year = int(vars_dict['lulc_snapshot_years_list'][-1])
    if vars_dict['analysis_year'] is not '':
        end_year = int(vars_dict['analysis_year'])

    yearly_carbon_price_dict = {}
    if vars_dict['do_price_table']:
        price_dict = pygeo.geoprocessing.get_lookup_from_table(
            vars_dict['price_table_uri'], 'year')

        # check all years in dict
        for year in range(start_year, end_year):
            if year not in price_dict:
                raise KeyError(
                    "Not all years are provided in carbon price table")

        for (year, d) in price_dict.items():
            t = year - start_year
            yearly_carbon_price_dict[int(year)] = d['price']/(
                (1 + discount_rate/100)**t)
    else:
        price_0 = float(vars_dict['price'])
        interest_rate = float(vars_dict['interest_rate'])

        for year in range(start_year, end_year):
            t = year - start_year
            price_t = price_0 * (1 + interest_rate/100)**t
            discounted_price = (price_t/((1 + discount_rate/100)**t))
            yearly_carbon_price_dict[year] = discounted_price

    return yearly_carbon_price_dict


def write_csv(filepath, l):
    """Write two-dimensional list to csv file."""
    f = open(filepath, 'wb')
    writer = csv.writer(f)
    for i in l:
        writer.writerow(i)
