# -*- coding: utf-8 -*-
"""CBC Model IO Functions."""

import csv
import os
import shutil
import pprint as pp
import logging

import numpy as np
import gdal
from pygeoprocessing import geoprocessing as geoprocess

from natcap.invest.coastal_blue_carbon import NODATA_INT, NODATA_FLOAT, HA_PER_M2


logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')
LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.io')


class AttrDict(dict):
    """Create a subclass of dictionary where keys can be accessed as attributes.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_inputs(args):
    """Get Inputs.

    Parameters:
        args (dict): user args dictionary.

    Returns:
        d (dict): data dictionary.

    Example Returns:
        d = {
            'workspace_dir': <string>,
            'outputs_dir': <string>,
            'transition_years': <list>,
            'analysis_year': <int>,
            'snapshot_years': <list>
            'timesteps': <int>,
            'transitions': <int>,
            'do_economic_analysis': <bool>,
            'price_t': <dict>,
            'discount_rate': None,
            'lulc_to_Sb': <dict>,
            'lulc_to_Ss': <dict>,
            'lulc_to_L': <dict>,
            'lulc_to_Yb': <dict>,
            'lulc_to_Ys': <dict>,
            'lulc_to_Hb': <dict>,
            'lulc_to_Hs': <dict>,
            'lulc_trans_to_Db': <dict>,
            'lulc_trans_to_Ds': <dict>,
            'C_s': <list>,
            'Y_pr': <dict>,
            'D_pr': <dict>,
            'H_pr': <dict>,
            'L_s': <list>,
            'A_pr': <dict>,
            'E_pr': <dict>,
            'S_pb': <dict>,
            'T_b': <list>,
            'N_pr': <dict>,
            'N_r': <list>,
            'N': <string>,
            'V': <string>
        }

    """
    d = AttrDict({
        'workspace_dir': None,
        'outputs_dir': None,
        'border_year_list': None,
        'do_economic_analysis': False,
        'lulc_to_Sb': {'lulc': 'biomass'},         # ic
        'lulc_to_Ss': {'lulc': 'soil'},            # ic
        'lulc_to_L': {'lulc': 'litter'},           # ic
        'lulc_to_Yb': {'lulc': 'accum-bio'},       # tc
        'lulc_to_Ys': {'lulc': 'accum-soil'},      # tc
        'lulc_to_Hb': {'lulc': 'hl-bio'},          # tc
        'lulc_to_Hs': {'lulc': 'hl-soil'},         # tc
        'lulc_trans_to_Db': {('lulc1', 'lulc2'): 'dist-val'}, # tc <-- preprocess with lulc_to_Db_high, lulc_to_Db_med, lulc_to_Db_low, lulc_to_Ds_high, lulc_to_Ds_med, lulc_to_Ds_low
        'lulc_trans_to_Ds': {('lulc1', 'lulc2'): 'dist-val'}, # tc <-- same
        'C_s': [],
        'C_prior': None,
        'C_r_rasters': [],
        'transition_years': [],
        'analysis_year': None,
        'snapshot_years': [],
        'timesteps': None,
        'transitions': None,
        'discount_rate': None,
        'interest_rate': None,
        'price': None,
        'price_t': None,
        'T_s_rasters': [],
        'A_r_rasters': [],
        'E_r_rasters': [],
        'N_r_rasters': [],
        'N_total_raster': None,
        'NPV_raster': None,
    })

    # Directories
    try:
        args['results_suffix']
    except:
        args['results_suffix'] = ''
    output_dir_name = 'outputs_core'
    if args['results_suffix'] != '':
        output_dir_name = output_dir_name + '_' + args['results_suffix']
    d.workspace_dir = args['workspace_dir']
    d.outputs_dir = os.path.join(args['workspace_dir'], output_dir_name)
    geoprocess.create_directories([args['workspace_dir'], d.outputs_dir])

    # Rasters
    d.transition_years = [int(i) for i in args['lulc_transition_years_list']]
    for i in range(0, len(d.transition_years)-1):
        if d.transition_years[i] >= d.transition_years[i+1]:
            raise ValueError(
                'LULC snapshot years must be provided in chronological order.'
                ' and in the same order as the LULC snapshot rasters.')
    d.transitions = len(d.transition_years)

    d.snapshot_years = [int(i) for i in d.transition_years]
    if args['analysis_year'] not in ['', None]:
        if int(args['analysis_year']) <= d.snapshot_years[-1]:
            raise ValueError('Analysis year must be greater than last transition year.')
        d.snapshot_years += [int(args['analysis_year'])]
    d.timesteps = d.snapshot_years[-1] - d.snapshot_years[0]

    d.C_prior_raster = args['lulc_baseline_map_uri']  #d.C_s[0]
    d.C_r_rasters = args['lulc_transition_maps_list'] # d.C_s[1:]

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
        args['lulc_transition_matrix_uri'],
        args['lulc_lookup_uri'],
        biomass_transient_dict,
        soil_transient_dict)
    d.lulc_trans_to_Db = lulc_trans_to_Db
    d.lulc_trans_to_Ds = lulc_trans_to_Ds

    # Economic Analysis
    if args['do_economic_analysis']:
        d.do_economic_analysis = True
        d.discount_rate = float(args['discount_rate']) * 0.01
        if args['do_price_table']:
            d.price_t = _get_price_table(
                args['price_table_uri'],
                d.snapshot_years[0],
                d.snapshot_years[-1])
        else:
            d.interest_rate = float(args['interest_rate']) * 0.01
            d.price = args['price']
            d.price_t = (1 + d.interest_rate) ** np.arange(0, d.timesteps+1) * d.price

        d.price_t = d.price_t / (1 + d.discount_rate)**np.arange(0, d.timesteps+1)

    # Create Output Rasters
    d.template_raster = d.C_prior_raster

    # Total Carbon Stock
    for i in range(0, len(d.snapshot_years)):
        fn = 'carbon_stock_at_%s.tif' % d.snapshot_years[i]
        filepath = os.path.join(d.outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            d.template_raster, filepath, 'GTiff', NODATA_FLOAT, 6)
        d.T_s_rasters.append(filepath)

    # Transition Accumulation
    for i in range(0, len(d.snapshot_years)-1):
        fn = 'carbon_accumulation_between_%s_and_%s.tif' % (
            d.snapshot_years[i], d.snapshot_years[i+1])
        filepath = os.path.join(d.outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            d.template_raster, filepath, 'GTiff', NODATA_FLOAT, 6)
        d.A_r_rasters.append(filepath)

    # Transition Emissions
    for i in range(0, len(d.snapshot_years)-1):
        fn = 'carbon_emissions_between_%s_and_%s.tif' % (
            d.snapshot_years[i], d.snapshot_years[i+1])
        filepath = os.path.join(d.outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            d.template_raster, filepath, 'GTiff', NODATA_FLOAT, 6)
        d.E_r_rasters.append(filepath)

    # Transition Net Sequestration
    for i in range(0, len(d.snapshot_years)-1):
        fn = 'net_carbon_sequestration_between_%s_and_%s.tif' % (
            d.snapshot_years[i], d.snapshot_years[i+1])
        filepath = os.path.join(d.outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            d.template_raster, filepath, 'GTiff', NODATA_FLOAT, 6)
        d.N_r_rasters.append(filepath)

    # Total Net Sequestration
    fn = 'net_carbon_sequestration_between_%s_and_%s.tif' % (
        d.snapshot_years[0], d.snapshot_years[-1])
    filepath = os.path.join(d.outputs_dir, fn)
    geoprocess.new_raster_from_base_uri(
        d.template_raster, filepath, 'GTiff', NODATA_FLOAT, 6)
    d.N_total_raster = filepath

    # Net Sequestration from Base Year to Analysis Year
    if args['do_economic_analysis']:
        # Total Net Present Value
        fn = 'net_present_value.tif'
        filepath = os.path.join(d.outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            d.template_raster, filepath, 'GTiff', NODATA_FLOAT, 6)
        d.NPV_raster = filepath

    return d


def _get_lulc_trans_to_D_dicts(lulc_transition_uri, lulc_lookup_uri, biomass_transient_dict, soil_transient_dict):
    """Get the lulc_trans_to_D dictionaries.

    Paramters:
        lulc_transition_uri (str)
        lulc_lookup_uri (str)
        biomass_transient_dict (dict)
        soil_transient_dict (dict)

    Returns:
        lulc_trans_to_Db (dict)
        lulc_trans_to_Ds (dict)

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
        else:
            raise ValueError('Pools in transient value table must only be \
                \'biomass\' or \'soil\'.')

    return biomass_transient_dict, soil_transient_dict


def _get_price_table(price_table_uri, start_year, end_year):
    price_dict = geoprocess.get_lookup_from_table(price_table_uri, 'year')

    price_t = np.zeros(end_year - start_year + 1)

    for year in range(start_year, end_year+1):
        if year not in price_dict:
            raise KeyError("Carbon price table must contain prices for all"
                "relevant years in analysis.")

    for year in range(start_year, end_year+1):
        idx = year - start_year
        price_t[idx] = price_dict[year]['price']

    return price_t


def write_csv(filepath, l):
    """Write two-dimensional list to csv file."""
    f = open(filepath, 'wb')
    writer = csv.writer(f)
    for i in l:
        writer.writerow(i)
