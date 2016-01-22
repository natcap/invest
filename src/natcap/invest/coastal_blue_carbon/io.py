# -*- coding: utf-8 -*-
"""CBC Model IO Functions."""

import csv
import os
import shutil
import pprint as pp
import logging

import numpy as np
from osgeo import gdal
from pygeoprocessing import geoprocessing as geoprocess

NODATA_FLOAT = -16777216

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
            'transition_years': <list>,
            'analysis_year': <int>,
            'snapshot_years': <list>
            'timesteps': <int>,
            'transitions': <int>,
            'do_economic_analysis': <bool>,
            'price_t': <dict>,
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
        'border_year_list': None,
        'do_economic_analysis': False,
        'lulc_to_Sb': {'lulc': 'biomass'},         # ic
        'lulc_to_Ss': {'lulc': 'soil'},            # ic
        'lulc_to_L': {'lulc': 'litter'},           # ic
        'lulc_to_Yb': {'lulc': 'accum-bio'},       # tc
        'lulc_to_Ys': {'lulc': 'accum-soil'},      # tc
        'lulc_to_Hb': {'lulc': 'hl-bio'},          # tc
        'lulc_to_Hs': {'lulc': 'hl-soil'},         # tc
        'lulc_trans_to_Db': {('lulc1', 'lulc2'): 'dist-val'},
        'lulc_trans_to_Ds': {('lulc1', 'lulc2'): 'dist-val'},
        'C_s': [],
        'C_prior': None,
        'C_r_rasters': [],
        'transition_years': [],
        'analysis_year': None,
        'snapshot_years': [],
        'timesteps': None,
        'transitions': None,
        'interest_rate': None,
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

    if len(args['results_suffix']) > 1:
        args['results_suffix'] = '_' + args['results_suffix']

    d.results_suffix = args['results_suffix']
    d.workspace_dir = args['workspace_dir']
    outputs_dir = os.path.join(args['workspace_dir'], 'outputs_core')
    geoprocess.create_directories([args['workspace_dir'], outputs_dir])

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
            raise ValueError(
                'Analysis year must be greater than last transition year.')
        d.snapshot_years.append(int(args['analysis_year']))
    d.timesteps = d.snapshot_years[-1] - d.snapshot_years[0]

    d.C_prior_raster = args['lulc_baseline_map_uri']
    d.C_r_rasters = args['lulc_transition_maps_list']

    # Reclass Dictionaries
    lulc_lookup_dict = geoprocess.get_lookup_from_csv(
        args['lulc_lookup_uri'], 'lulc-class')
    lulc_to_code_dict = \
        dict((k.lower(), v['code']) for k, v in lulc_lookup_dict.items())
    initial_dict = geoprocess.get_lookup_from_csv(
            args['carbon_pool_initial_uri'], 'lulc-class')

    d.lulc_to_Sb = dict((lulc_to_code_dict[key.lower()], sub['biomass'])
                        for key, sub in initial_dict.items())
    d.lulc_to_Ss = dict((lulc_to_code_dict[key.lower()], sub['soil'])
                        for key, sub in initial_dict.items())
    d.lulc_to_L = dict((lulc_to_code_dict[key.lower()], sub['litter'])
                       for key, sub in initial_dict.items())

    # Transition Dictionaries
    biomass_transient_dict, soil_transient_dict = \
        _create_transient_dict(args['carbon_pool_transient_uri'])

    d.lulc_to_Yb = dict((lulc_to_code_dict[key.lower()],
                        sub['yearly_accumulation'])
                        for key, sub in biomass_transient_dict.items())
    d.lulc_to_Ys = dict((lulc_to_code_dict[key.lower()],
                        sub['yearly_accumulation'])
                        for key, sub in soil_transient_dict.items())
    d.lulc_to_Hb = dict((lulc_to_code_dict[key.lower()],
                        sub['half-life'])
                        for key, sub in biomass_transient_dict.items())
    d.lulc_to_Hs = dict((lulc_to_code_dict[key.lower()], sub['half-life'])
                        for key, sub in soil_transient_dict.items())

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
        discount_rate = float(args['discount_rate']) * 0.01
        if args['do_price_table']:
            d.price_t = _get_price_table(
                args['price_table_uri'],
                d.snapshot_years[0],
                d.snapshot_years[-1])
        else:
            d.interest_rate = float(args['interest_rate']) * 0.01
            price = args['price']
            d.price_t = (1 + d.interest_rate) ** np.arange(
                0, d.timesteps+1) * price

        d.price_t = d.price_t / (1 + discount_rate)**np.arange(
            0, d.timesteps+1)

    # Create Output Rasters
    template_raster = d.C_prior_raster

    # Total Carbon Stock
    for i in range(0, len(d.snapshot_years)):
        fn = 'carbon_stock_at_%s%s.tif' % (
            d.snapshot_years[i], d.results_suffix)
        filepath = os.path.join(outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            template_raster, filepath, 'GTiff', NODATA_FLOAT, gdal.GDT_Float32)
        d.T_s_rasters.append(filepath)

    for i in range(0, len(d.snapshot_years)-1):
        # Transition Accumulation
        fn = 'carbon_accumulation_between_%s_and_%s%s.tif' % (
            d.snapshot_years[i], d.snapshot_years[i+1], d.results_suffix)
        filepath = os.path.join(outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            template_raster, filepath, 'GTiff', NODATA_FLOAT, gdal.GDT_Float32)
        d.A_r_rasters.append(filepath)

        # Transition Emissions
        fn = 'carbon_emissions_between_%s_and_%s%s.tif' % (
            d.snapshot_years[i], d.snapshot_years[i+1], d.results_suffix)
        filepath = os.path.join(outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            template_raster, filepath, 'GTiff', NODATA_FLOAT, gdal.GDT_Float32)
        d.E_r_rasters.append(filepath)

        # Transition Net Sequestration
        fn = 'net_carbon_sequestration_between_%s_and_%s%s.tif' % (
            d.snapshot_years[i], d.snapshot_years[i+1], d.results_suffix)
        filepath = os.path.join(outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            template_raster, filepath, 'GTiff', NODATA_FLOAT, gdal.GDT_Float32)
        d.N_r_rasters.append(filepath)

    # Total Net Sequestration
    fn = 'net_carbon_sequestration_between_%s_and_%s%s.tif' % (
        d.snapshot_years[0], d.snapshot_years[-1], d.results_suffix)
    filepath = os.path.join(outputs_dir, fn)
    geoprocess.new_raster_from_base_uri(
        template_raster, filepath, 'GTiff', NODATA_FLOAT, gdal.GDT_Float32)
    d.N_total_raster = filepath

    # Net Sequestration from Base Year to Analysis Year
    if args['do_economic_analysis']:
        # Total Net Present Value
        fn = 'net_present_value%s.tif' % (d.results_suffix)
        filepath = os.path.join(outputs_dir, fn)
        geoprocess.new_raster_from_base_uri(
            template_raster, filepath, 'GTiff', NODATA_FLOAT, gdal.GDT_Float32)
        d.NPV_raster = filepath

    return d


def _get_lulc_trans_to_D_dicts(lulc_transition_uri, lulc_lookup_uri,
                               biomass_transient_dict, soil_transient_dict):
    """Get the lulc_trans_to_D dictionaries.

    Args:
        lulc_transition_uri (str): transition matrix table
        lulc_lookup_uri (str): lulc lookup table
        biomass_transient_dict (dict): transient biomass values
        soil_transient_dict (dict): transient soil values

    Returns:
        lulc_trans_to_Db (dict): biomass transition values
        lulc_trans_to_Ds (dict): soil transition values

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
        dict((k, v['code']) for k, v in lulc_lookup_dict.items())

    lulc_trans_to_Db = {}
    lulc_trans_to_Ds = {}
    for k, sub in lulc_transition_dict.items():
        if k is not '':
            continue
        for k2, v in sub.items():
            if k2 is not '' and v.endswith('disturb'):
                lulc_trans_to_Db[(
                    lulc_to_code_dict[k], lulc_to_code_dict[k2])] = \
                    biomass_transient_dict[k][v]
                lulc_trans_to_Ds[(
                    lulc_to_code_dict[k], lulc_to_code_dict[k2])] = \
                    soil_transient_dict[k][v]

    return lulc_trans_to_Db, lulc_trans_to_Db


def _create_transient_dict(carbon_pool_transient_uri):
    """Create dictionary of transient variables for carbon pools.

    Parameters:
        carbon_pool_transient_uri (string): path to carbon pool transient
            variables csv file.

    Returns:
        biomass_transient_dict (dict): transient biomass values
        soil_transient_dict (dict): transient soil values
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
            raise ValueError('Pools in transient value table must only be '
                             '"biomass" or "soil".')

    return biomass_transient_dict, soil_transient_dict


def _get_price_table(price_table_uri, start_year, end_year):
    """Get price table.

    Parameters:
        price_table_uri (str): filepath to price table csv file
        start_year (int): start year of analysis
        end_year (int): end year of analysis

    Returns:
        price_t (np.array): price for each year.
    """
    price_dict = geoprocess.get_lookup_from_table(price_table_uri, 'year')
    price_t = np.zeros(end_year - start_year + 1)

    try:
        return np.array([price_dict[year-start_year]['price']
                        for year in xrange(start_year, end_year+1)])
    except KeyError as missing_year:
        raise KeyError('Carbon price table does not contain a price value for '
                       '%s' % missing_year)
