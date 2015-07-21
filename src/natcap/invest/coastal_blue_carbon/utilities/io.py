"""CBC Model IO Utilities."""
import csv
import os
import pygeoprocessing as pygeo


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
            'disturbed_carbon_stock_object_list': [],
            'accumulated_carbon_stock_object_list': [],
            'net_sequestration_raster_list': [],
            'total_carbon_stock_raster_list': [],
            'biomass_carbon_stock_raster_list': [],
            'soil_carbon_stock_raster_list': [],
        }
    """
    vars_dict = args.copy()

    outputs_dir = os.path.join(vars_dict['workspace'], 'outputs')
    if not os.path.exists(vars_dict['workspace']):
        os.makedirs(vars_dict['workspace'])
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    vars_dict['outputs_dir'] = outputs_dir

    vars_dict['lulc_lookup_dict'] = pygeo.geoprocessing.get_lookup_from_csv(
        args['lulc_lookup_uri'], 'code')

    lulc_transition_dict = pygeo.geoprocessing.get_lookup_from_csv(
        args['lulc_transition_uri'], 'lulc-class')
    l = []
    for item in lulc_transition_dict.items():
        del item[1]['lulc-class']
        l.append(item)
    vars_dict['lulc_transition_dict'] = dict(l)

    vars_dict['carbon_pool_initial_dict'] = \
        pygeo.geoprocessing.get_lookup_from_csv(
            args['carbon_pool_initial_uri'], 'lulc-class')

    lulc_lookup_dict = vars_dict['lulc_lookup_dict']
    code_to_lulc_dict = {key: lulc_lookup_dict[key][
        'lulc-class'] for key in lulc_lookup_dict.keys()}
    vars_dict['lulc_to_code_dict'] = {
        v: k for k, v in code_to_lulc_dict.items()}
    vars_dict['code_to_lulc_dict'] = code_to_lulc_dict

    vars_dict['carbon_pool_transient_dict'] = _create_transient_dict(args)
    vars_dict['disturbed_carbon_stock_object_list'] = range(
        0, len(vars_dict['lulc_snapshot_list'])-1)
    vars_dict['accumulated_carbon_stock_object_list'] = range(
        0, len(vars_dict['lulc_snapshot_list'])-1)
    vars_dict['net_sequestration_raster_list'] = range(
        0, len(vars_dict['lulc_snapshot_list'])-1)
    vars_dict['total_carbon_stock_raster_list'] = range(
        0, len(vars_dict['lulc_snapshot_list']))
    vars_dict['biomass_carbon_stock_raster_list'] = range(
        0, len(vars_dict['lulc_snapshot_list']))
    vars_dict['soil_carbon_stock_raster_list'] = range(
        0, len(vars_dict['lulc_snapshot_list']))

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
