"""CBC Model IO Utilities."""

import csv
import os
import shutil
import pprint as pp

import gdal
from pygeoprocessing import geoprocessing as geoprocess

from natcap.invest.coastal_blue_carbon import NODATA_INT, NODATA_FLOAT, HA_PER_M2
from natcap.invest.coastal_blue_carbon.classes.raster import Raster
from natcap.invest.coastal_blue_carbon.classes.raster_stack import RasterStack

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_inputs(args):
    d = AttrDict({
        'workspace_dir': None,
        'outputs_dir': None,
        'border_year_list': None,
        'do_economic_analysis': False,
        'price_t': None,
        'discount_rate': None,
        'lulc_to_Sb': {'lulc': 'biomass'},         # ic
        'lulc_to_Ss': {'lulc': 'soil'},            # ic
        'lulc_to_L': {'lulc': 'litter'},           # ic
        'lulc_to_Yb': {'lulc': 'accum-bio'},       # tc
        'lulc_to_Ys': {'lulc': 'accum-soil'},      # tc
        'lulc_to_Hb': {'lulc': 'hl-bio'},          # tc
        'lulc_to_Hs': {'lulc': 'hl-soil'},         # tc
        'lulc_trans_to_Db': {('lulc1', 'lulc2'): 'dist-val'}, # tc <-- preprocess with lulc_to_Db_high, lulc_to_Db_med, lulc_to_Db_low, lulc_to_Ds_high, lulc_to_Ds_med, lulc_to_Ds_low
        'lulc_trans_to_Ds': {('lulc1', 'lulc2'): 'dist-val'}, # tc <-- same
        'C_s': [],                                      # given
        'Y_pr': AttrDict({'biomass': [], 'soil': []}),  # precompute
        'D_pr': AttrDict({'biomass': [], 'soil': []}),  # precompute
        'H_pr': AttrDict({'biomass': [], 'soil': []}),  # precompute
        'L_s': [],                                      # precompute
        'A_pr': AttrDict({'biomass': [], 'soil': []}),
        'E_pr': AttrDict({'biomass': [], 'soil': []}),
        'S_pb': AttrDict({'biomass': [], 'soil': []}),
        'T_b': [],
        'N_pr': AttrDict({'biomass': [], 'soil': []}),
        'N_r': [],
        'N': None,
        'V': None
    })

    # Directories
    try:
        args['results_suffix']
    except:
        args['results_suffix'] = ''
    output_dir_name = 'outputs_core'
    if args['results_suffix'] != '':
        output_dir_name = output_dir_name + '_' + args['results_suffix']
    outputs_dir = os.path.join(args['workspace_dir'], output_dir_name)
    geoprocess.create_directories([args['workspace_dir'], outputs_dir])
    d.workspace_dir = args['workspace_dir']
    d.outputs_dir = outputs_dir

    # Rasters
    d.border_year_list = [int(i) for i in args['lulc_snapshot_years_list']]
    for i in range(0, len(d.border_year_list)-1):
        if d.border_year_list[i] >= d.border_year_list[i+1]:
            raise ValueError(
                'LULC snapshot years must be provided in chronological order.')

    d.C_s = _get_snapshot_rasters(args['lulc_snapshot_list'])

    # Reclass Dictionaries
    lulc_lookup_dict = geoprocess.get_lookup_from_csv(
        args['lulc_lookup_uri'], 'lulc-class')
    lulc_to_code_dict = \
        dict([(k.lower(), v['code']) for k, v in lulc_lookup_dict.items()])
    initial_dict = geoprocess.get_lookup_from_csv(
            args['carbon_pool_initial_uri'], 'lulc-class')

    d.lulc_to_Sb = dict([(lulc_to_code_dict[key.lower()], sub['biomass']) \
        for key, sub in initial_dict.items()])
    d.lulc_to_Ss = dict([(lulc_to_code_dict[key.lower()], sub['soil']) \
        for key, sub in initial_dict.items()])
    d.lulc_to_L =  dict([(lulc_to_code_dict[key.lower()], sub['litter']) \
        for key, sub in initial_dict.items()])

    # Transition Dictionaries
    biomass_transient_dict, soil_transient_dict = \
        _create_transient_dict(args['carbon_pool_transient_uri'])

    d.lulc_to_Yb = dict([(lulc_to_code_dict[key.lower()], sub['yearly_accumulation']) \
        for key, sub in biomass_transient_dict.items()])
    d.lulc_to_Ys = dict([(lulc_to_code_dict[key.lower()], sub['yearly_accumulation']) \
        for key, sub in soil_transient_dict.items()])
    d.lulc_to_Hb = dict([(lulc_to_code_dict[key.lower()], sub['half-life']) \
        for key, sub in biomass_transient_dict.items()])
    d.lulc_to_Hs = dict([(lulc_to_code_dict[key.lower()], sub['half-life']) \
        for key, sub in soil_transient_dict.items()])

    # Parse LULC Transition CSV (Carbon Direction and Relative Magnitude)
    lulc_trans_to_Db, lulc_trans_to_Ds = _get_lulc_trans_to_D_dicts(
        args['lulc_transition_uri'],
        args['lulc_lookup_uri'],
        biomass_transient_dict,
        soil_transient_dict)
    d.lulc_trans_to_Db = lulc_trans_to_Db
    d.lulc_trans_to_Ds = lulc_trans_to_Ds

    # Economic Analysis
    if args['do_economic_analysis']:
        d.do_economic_analysis = True
        d.discount_rate = args['discount_rate']
        d.price_t = _get_yearly_carbon_price_dict(args)

    return d


def _get_lulc_trans_to_D_dicts(lulc_transition_uri, lulc_lookup_uri, biomass_transient_dict, soil_transient_dict):
    """Get the lulc_trans_to_D dictionaries.

    Paramters:
        lulc_transition_uri (str)

    Returns:
        lulc_trans_to_Db (dict)
        lulc_trans_to_Ds (dict)
        biomass_transient_dict (dict)
        soil_transient_dict (dict)

    Example Returns:
        lulc_trans_to_Db = {
            (lulc-1, lulc-2): dist-val,
            (lulc-1, lulc-3): dist-val,
            ...
        }
    """
    lulc_transition_dict = geoprocess.get_lookup_from_csv(
        lulc_transition_uri, 'lulc-class')
    lulc_lookup_dict = geoprocess.get_lookup_from_csv(
        lulc_lookup_uri, 'lulc-class')
    lulc_to_code_dict = \
        dict([(k, v['code']) for k, v in lulc_lookup_dict.items()])

    biomass_item_list = []
    soil_item_list = []
    for k, sub in lulc_transition_dict.items():
        for k2, v in sub.items():
            if k is not '' and k2 is not '' and v.endswith('disturb'):
                biomass_item_list.append(
                    ((lulc_to_code_dict[k], lulc_to_code_dict[k2]), biomass_transient_dict[k][v]))
                soil_item_list.append(
                    ((lulc_to_code_dict[k], lulc_to_code_dict[k2]), soil_transient_dict[k][v]))

    lulc_trans_to_Db = dict(biomass_item_list)
    lulc_trans_to_Db = dict(soil_item_list)
    return lulc_trans_to_Db, lulc_trans_to_Db


def _get_snapshot_rasters(lulc_snapshot_list):
    """Get valid snapshot rasters.

    - Assert same projection
    - Align
    - Set same nodata value

    Parameters:
        lulc_snapshot_list (list): filepaths to lulc rasters

    Returns:
        C_s (list): list of aligned rasters with standard nodata value
    """
    stack = RasterStack([Raster.from_file(i) for i in lulc_snapshot_list])
    for r in stack.raster_list:
        r.resample_method = 'nearest'
    if not stack.all_same_projection():
        raise ValueError('LULC snapshot rasters must be in same projection.')
    if not stack.all_aligned():
        stack = stack.align()
    stack = stack.set_standard_nodata(NODATA_INT)
    stack_filepath_list = stack.get_raster_uri_list()
    C_s = []
    for i in stack_filepath_list:
        temp = geoprocess.temporary_filename()
        shutil.copy(i, temp)
        C_s.append(temp)
    return C_s


def _create_transient_dict(carbon_pool_transient_uri):
    """Create dictionary of transient variables for carbon pools.

    Parameters:
        carbon_pool_transient_uri (string): path to carbon pool transient
            variables csv file.

    Returns:
        biomass_transient_dict (dict)
        soil_transient_dict (dict)
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
    header = lines[0]

    biomass_transient_dict = {}
    soil_transient_dict = {}

    for line in lines[1:]:
        el_dict = dict(zip(header, line))
        lulc = el_dict['lulc-class']
        pool = el_dict['pool']
        if pool == 'biomass':
            biomass_transient_dict[lulc] = dict(zip(header, line))
        elif pool == 'soil':
            soil_transient_dict[lulc] = dict(zip(header, line))

    return biomass_transient_dict, soil_transient_dict


def _get_yearly_carbon_price_dict(vars_dict):
    """Return dictionary of discounted prices for each year.

    Parameters:
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

    yearly_carbon_price_dict = {}
    if vars_dict['do_price_table']:
        price_dict = geoprocess.get_lookup_from_table(
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
