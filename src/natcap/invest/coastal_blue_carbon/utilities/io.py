"""CBC Model IO Utilities."""
import csv
import os
import pygeoprocessing as pygeo
import pprint as pp

from natcap.invest.coastal_blue_carbon.utilities.raster import Raster

NODATA_FLOAT = -16777216
NODATA_INT = -9999


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
    if not os.path.exists(vars_dict['workspace_dir']):
        os.makedirs(vars_dict['workspace_dir'])
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    vars_dict['outputs_dir'] = outputs_dir

    vars_dict['lulc_lookup_dict'] = pygeo.geoprocessing.get_lookup_from_csv(
        args['lulc_lookup_uri'], 'code')

    # Parse LULC Transition CSV (Carbon Direction and Relative Magnitude)
    lulc_transition_dict = pygeo.geoprocessing.get_lookup_from_csv(
        args['lulc_transition_uri'], 'lulc-class')
    lulc_transition_dict['undefined'] = {
        u'lulc-class': u'undefined', u'undefined': u'undefined'}
    l = []
    for item in lulc_transition_dict.items():
        del item[1]['lulc-class']
        l.append(item)
    vars_dict['lulc_transition_dict'] = dict(l)

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
    vars_dict['carbon_pool_transient_dict'] = _create_transient_dict(args)
    nan_dict = {
        u'half-life': NODATA_FLOAT,
        u'high-impact-disturbance': NODATA_FLOAT,
        u'low-impact-disturbance': NODATA_FLOAT,
        u'med-impact-disturbance': NODATA_FLOAT,
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
    l = []
    for i in vars_dict['lulc_snapshot_list']:
        l.append(Raster.from_file(i).set_nodata(NODATA_INT).uri)
    vars_dict['lulc_snapshot_list'] = l

    return vars_dict


def _create_transient_dict(args):
    """Create transient dict."""
    def to_float(x):
        try:
            return float(x)
        except ValueError:
            return x

    carbon_pool_transient_uri = args['carbon_pool_transient_uri']
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
