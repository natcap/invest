"""Tests for the New CBC Model."""
import unittest
import os
import shutil
import pprint

import numpy as np
import gdal
from pygeoprocessing import geoprocessing as geoprocess
from pygeoprocessing import testing as geotest

from natcap.invest.coastal_blue_carbon import model

pp = pprint.PrettyPrinter(indent=4)


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

def get_data(dataset):
    """Get data."""
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
        'V': '',
        'do_economic_analysis': True,
        'outputs_dir': None,
        'workspace_dir': None
    }

    d = AttrDict(data)

    if dataset == 1:
        d.C_s.append(create_test_raster(np.array([[3.0]])))
        d.C_s.append(create_test_raster(np.array([[2.0]])))
        d.C_s.append(create_test_raster(np.array([[1.0]])))
        d.C_s.append(create_test_raster(np.array([[2.0]])))

    if dataset == 2:
        d.C_s.append(create_test_raster(np.random.randint(1, 4, size=shape)))
        d.C_s.append(create_test_raster(np.random.randint(1, 4, size=shape)))
        d.C_s.append(create_test_raster(np.random.randint(1, 4, size=shape)))
        d.C_s.append(create_test_raster(np.random.randint(1, 4, size=shape)))

    if dataset == 3:
        d.C_s.append(create_test_raster(np.array([[3.0]])))
        d.C_s.append(create_test_raster(np.array([[1.0]])))
        d.C_s.append(create_test_raster(np.array([[2.0]])))
        d.C_s.append(create_test_raster(np.array([[3.0]])))

    d.lulc_to_Sb = {1: 0, 2: 10.0, 3: 10.0}
    d.lulc_to_Ss = {1: 0, 2: 10.0, 3: 10.0}
    d.lulc_to_L = {1: 0, 2: 0.5, 3: 0.5}
    d.lulc_to_Yb = {1: 0, 2: 1.0, 3: 1.0}
    d.lulc_to_Ys = {1: 0, 2: 1.0, 3: 1.0}
    d.lulc_to_Hb = {1: 0, 2: 5.0, 3: 5.0}
    d.lulc_to_Hs = {1: 0, 2: 5.0, 3: 5.0}
    d.lulc_trans_to_Db = {(1, 1): 0, (2, 2): 0, (2, 2): 0, (1, 2): 0, (1, 3): 0, (2, 1): 0.5, (2, 3): 0, (3, 1): 0.5, (3, 2): 0}
    d.lulc_trans_to_Ds = {(1, 1): 0, (2, 2): 0, (2, 2): 0, (1, 2): 0, (1, 3): 0, (2, 1): 0.5, (2, 3): 0, (3, 1): 0.5, (3, 2): 0}

    return d


class TestFunctions(unittest.TestCase):

    """Function-Level Tests."""

    def setUp(self):
        pass

    def test_init_SLT(self):
        d = get_data(1)
        model.init_SLT(d)
        guess = read_array(d.T_b[0])
        np.testing.assert_array_equal(guess, np.array([[20.5]]))

    def test_reclass_C_to_YDH(self):
        d = get_data(1)
        model.init_SLT(d)
        r = 0
        model.reclass_C_to_YDH(d, r)
        check = np.array([[1.]])
        guess = read_array(d.Y_pr.biomass[0])
        np.testing.assert_array_almost_equal(guess, check)
        guess = read_array(d.Y_pr.soil[0])
        np.testing.assert_array_almost_equal(guess, check)

    def test_compute_save_AENSLT(self):
        d = get_data(1)
        model.init_SLT(d)
        model.compute_transition(d, 0)
        guess = read_array(d.A_pr.biomass[0])
        np.testing.assert_array_equal(guess, np.array([[5.]]))

    def test_compute_timestep_AENV(self):
        d = get_data(1)
        model.init_SLT(d)
        model.compute_transition(d, 0)
        model.compute_timestep_AENV(d, 0, 1)
        guess = read_array(d.V)
        np.testing.assert_array_equal(
            guess, np.array([[2.]]).astype(np.float32))

    def test_compute_NV_across_timesteps(self):
        d = get_data(1)
        model.init_SLT(d)
        model.compute_transition(d, 0)
        model.compute_NV_across_timesteps(d, 0)
        guess = read_array(d.V)
        np.testing.assert_array_equal(
            guess, np.array([[10.]]).astype(np.float32))

    def test_save_TAENV(self):
        d = get_data(1)
        pass

    def tearDown(self):
        pass


class TestModel(unittest.TestCase):

    """Model-Level Tests."""

    def setUp(self):
        """About."""
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.workspace_dir = os.path.join(cwd, 'workspace')
        if not os.path.exists(self.workspace_dir):
            os.mkdir(self.workspace_dir)
        self.outputs_dir = os.path.join(self.workspace_dir, 'outputs_core')
        if not os.path.exists(self.outputs_dir):
            os.mkdir(self.outputs_dir)

    def test_run(self):
        """About."""
        d = get_data(1)
        d.workspace_dir = self.workspace_dir
        d.outputs_dir = self.outputs_dir

        d = model.run(d)

        check = [
            np.array([[20.5]]),
            np.array([[30.5]]),
            np.array([[22.5]]),
            np.array([[29.25]])
        ]
        z = zip(d.T_b, check)
        [np.testing.assert_array_equal(
            read_array(guess), check) for guess, check in z]

    def tearDown(self):
        """About."""
        shutil.rmtree(self.workspace_dir)


if __name__ == '__main__':
    unittest.main()
