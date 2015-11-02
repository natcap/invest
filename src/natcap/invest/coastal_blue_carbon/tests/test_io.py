"""Tests for the New CBC Model."""
import unittest
import pprint
import os
import shutil

import numpy as np
import gdal
from pygeoprocessing import geoprocessing as geoprocess
from pygeoprocessing import testing as geotest

from natcap.invest.coastal_blue_carbon import io
from natcap.invest.coastal_blue_carbon.classes.raster import Raster

pp = pprint.PrettyPrinter(indent=4)


def get_data():
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    price_t = []
    for year in range(2000, 2016):
        price_t.append((year, 1))
    price_t = dict(price_t)

    data = {
        'border_year_list': [2000, 2005, 2010, 2015],    # given
        'lulc_to_Sb': {'lulc': 'biomass'},         # ic
        'lulc_to_Ss': {'lulc': 'soil'},            # ic
        'lulc_to_L': {'lulc': 'litter'},           # ic
        'lulc_to_Yb': {'lulc': 'accum-bio'},       # tc
        'lulc_to_Ys': {'lulc': 'accum-soil'},      # tc
        'lulc_to_Hb': {'lulc': 'hl-bio'},          # tc
        'lulc_to_Hs': {'lulc': 'hl-soil'},         # tc
        'lulc_trans_to_Db': {('lulc1', 'lulc2'): 'dist-val'}, # tc <-- preprocess with lulc_to_Db_high, lulc_to_Db_med, lulc_to_Db_low, lulc_to_Ds_high, lulc_to_Ds_med, lulc_to_Ds_low
        'lulc_trans_to_Ds': {('lulc1', 'lulc2'): 'dist-val'}, # tc <-- same
        'price_t': price_t,
        'discount_rate': 0.05,
        'C_s': [],             # given
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
        'N': '',
        'V': ''
    }

    d = AttrDict(data)

    return d

def read_array(filepath):
    ds = gdal.Open(filepath)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()

def create_test_raster(a):
    a = [a]
    origin = geotest.sampledata.SRS_COLOMBIA.origin
    proj_wkt = geotest.sampledata.SRS_COLOMBIA.projection
    nodata = -9999
    pixel_size = geotest.sampledata.SRS_COLOMBIA.pixel_size(1)
    fn = geoprocess.temporary_filename()
    geotest.sampledata.create_raster_on_disk(
        a,
        origin,
        proj_wkt,
        nodata,
        pixel_size,
        filename=fn)
    return fn


class TestGetInputs(unittest.TestCase):

    """Test io.get_inputs."""

    def setUp(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        workspace = os.path.join(cwd, 'workspace')
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        self.workspace = workspace
        self.results_suffix = ''

        table = [
            ['lulc-class', 'code', 'is_coastal_blue_carbon_habitat'],
            ['seagrass', '1', 'true'],
            ['man-made', '2', 'false'],
            ['marsh', '3', 'true'],
            ['mangrove', '4', 'true']]
        self.lulc_lookup_uri = os.path.join(self.workspace, 'lookup.csv')
        io.write_csv(self.lulc_lookup_uri, table)

        table = [
            ['lulc-class', 'seagrass', 'man-made', 'marsh', 'mangrove'],
            ['seagrass', 'accum', 'med-impact-disturb', '', ''],
            ['man-made', 'accum', '', 'accum', ''],
            ['marsh', '', '', '', 'accum'],
            ['mangrove', '', '', '', '']]
        self.lulc_transition_uri = os.path.join(
            self.workspace, 'transition.csv')
        io.write_csv(self.lulc_transition_uri, table)

        year1_raster = create_test_raster(np.array([[1., 2.], [2., 1.]]))
        year2_raster = create_test_raster(np.array([[2., 1.], [1., 2.]]))
        year3_raster = create_test_raster(np.array([[3., 1.], [1., 3.]]))
        year4_raster = create_test_raster(np.array([[4., 1.], [1., 4.]]))

        self.lulc_snapshot_list = [
            year1_raster,
            year2_raster,
            year3_raster,
            year4_raster]

        self.lulc_snapshot_years_list = [2000, 2005, 2020, 2050]

        table = [
            ['lulc-class', 'biomass', 'soil', 'litter'],
            ['seagrass', '1.0', '1.0', '0.5'],
            ['man-made', '0.0', '0.0', '0'],
            ['marsh', '2.0', '2.0', '1.0'],
            ['mangrove', '3.0', '3.0', '1.5']]
        self.carbon_pool_initial_uri = os.path.join(
            self.workspace, 'initial.csv')
        io.write_csv(self.carbon_pool_initial_uri, table)

        table = [
            ['lulc-class', 'pool', 'half-life', 'yearly_accumulation', 'low-impact-disturb', 'med-impact-disturb', 'high-impact-disturb'],
            ['seagrass', 'biomass', '1', '10', '0.1', '0.3', '0.7'],
            ['seagrass', 'soil', '2', '10', '0.1', '0.3', '0.7'],
            ['man-made', 'biomass', '0', '0', '0', '0', '0'],
            ['man-made', 'soil', '0', '0', '0', '0', '0'],
            ['marsh', 'biomass', '1', '20', '0.2', '0.4', '0.8'],
            ['marsh', 'soil', '2', '20', '0.2', '0.4', '0.8'],
            ['mangrove', 'biomass', '1', '30', '0.3', '0.5', '0.7'],
            ['mangrove', 'soil', '2', '30', '0.3', '0.5', '0.7']]
        self.carbon_pool_transient_uri = os.path.join(
            self.workspace, 'transient.csv')
        io.write_csv(self.carbon_pool_transient_uri, table)

        table = [['year', 'price']]
        for year in range(2000, 2101):
            table.append([str(year), '10.0'])
        self.price_table_uri = os.path.join(self.workspace, 'price_table.csv')
        io.write_csv(self.price_table_uri, table)

        self.args = {
            'workspace_dir': self.workspace,
            'results_suffix': self.results_suffix,
            'lulc_lookup_uri': self.lulc_lookup_uri,
            'lulc_transition_uri': self.lulc_transition_uri,
            'lulc_snapshot_list': self.lulc_snapshot_list,
            'lulc_snapshot_years_list': self.lulc_snapshot_years_list,
            'carbon_pool_initial_uri': self.carbon_pool_initial_uri,
            'carbon_pool_transient_uri': self.carbon_pool_transient_uri,
            'do_economic_analysis': True,
            'do_price_table': True,
            'price': '10.0',
            'interest_rate': '5.0',
            'price_table_uri': self.price_table_uri,
            'discount_rate': '6.0'
        }

    def test_get_inputs(self):
        d = io.get_inputs(self.args)
        self.assertTrue('border_year_list' in d)
        self.assertTrue('lulc_to_Sb' in d)
        self.assertTrue('lulc_to_Ss' in d)
        self.assertTrue('lulc_to_L' in d)
        self.assertTrue('lulc_to_Yb' in d)
        self.assertTrue('lulc_to_Ys' in d)
        self.assertTrue('lulc_to_Hb' in d)
        self.assertTrue('lulc_to_Hs' in d)
        self.assertTrue('price_t' in d)
        self.assertTrue('lulc_trans_to_Ds' in d)
        self.assertTrue('C_s' in d)
        self.assertTrue('Y_pr' in d)
        self.assertTrue('D_pr' in d)
        self.assertTrue('H_pr' in d)
        self.assertTrue('L_s' in d)
        self.assertTrue('A_pr' in d)
        self.assertTrue('E_pr' in d)
        self.assertTrue('S_pb' in d)
        self.assertTrue('T_b' in d)
        self.assertTrue('N_pr' in d)
        self.assertTrue('N_r' in d)
        self.assertTrue('N' in d)
        self.assertTrue('V' in d)
        self.assertTrue('do_economic_analysis' in d)
        self.assertTrue('outputs_dir' in d)
        self.assertTrue('workspace_dir' in d)

    def test_get_lulc_trans_to_D_dicts(self):
        biomass_transient_dict, soil_transient_dict = \
            io._create_transient_dict(self.args['carbon_pool_transient_uri'])
        biomass_dict, soil_dict = io._get_lulc_trans_to_D_dicts(
            self.args['lulc_transition_uri'],
            self.args['lulc_lookup_uri'],
            biomass_transient_dict,
            soil_transient_dict)
        self.assertTrue(type(biomass_dict.items()[0]) is tuple)

    def test_get_snapshot_rasters(self):
        C_s = io._get_snapshot_rasters(self.lulc_snapshot_list)
        self.assertTrue(Raster.from_file(C_s[0]).get_nodata(1) == -9999)

    def tearDown(self):
        shutil.rmtree(self.workspace)


if __name__ == '__main__':
    unittest.main()
