"""Tests for Utilities."""
import unittest
import pprint

import numpy as np
import gdal
from pygeoprocessing import geoprocessing as geoprocess
from pygeoprocessing import testing as geotest

from natcap.invest.coastal_blue_carbon import utils

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


class TestUtils(unittest.TestCase):

    """Reclass Tests."""

    def setUp(self):
        """Setup."""
        pass

    def test_reclass(self):
        """Test Reclass Operation."""
        reclass_dict = {1: 2, 4: 5}

        a = np.ma.masked_array(np.ones((2, 2)))
        a[0, 0] = 4
        r = create_test_raster(a)

        guess_filepath = utils.reclass(r, reclass_dict, mask_other_vals=True)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array([[5, 2], [2, 2]])
        np.testing.assert_array_equal(guess, check)

    def test_reclass_transition(self):
        """Test Reclass Transition Operation."""
        trans_dict = {(1, 4): 2.0}
        a_prev = np.ma.masked_array(np.ones((1, 1)))
        r_prev = create_test_raster(a_prev)
        a_next = np.ma.masked_array(np.ones((1, 1)))
        a_next[0, 0] = 4
        r_next = create_test_raster(a_next)

        guess_filepath = utils.reclass_transition(
            r_prev, r_next, trans_dict, out_dtype=np.float32)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array(np.ones((1, 1)))
        check[0, 0] = 2.0

        np.testing.assert_array_equal(guess, check)

    def test_add(self):
        """Test Add Operation."""
        a = np.ma.masked_array(np.ones((1, 1)))
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1)))
        r_2 = create_test_raster(b)

        guess_filepath = utils.add(r_1, r_2)
        guess = read_array(guess_filepath)
        check = b * 2

        np.testing.assert_array_equal(guess, check)

    def test_add_nodata(self):
        """Test Add Operation."""
        a = np.ma.masked_array(np.ones((1, 1)))
        a[:] = -9999
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1)))
        r_2 = create_test_raster(b)

        guess_filepath = utils.add(r_1, r_2)
        guess = read_array(guess_filepath)
        check = -9999

        np.testing.assert_array_equal(guess, check)

    def test_add_inplace(self):
        """Test Add Inplace Operation."""
        a = np.ma.masked_array(np.ones((1, 1)))
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1)))
        r_2 = create_test_raster(b)

        guess_filepath = utils.add_inplace(r_1, r_2)
        guess = read_array(guess_filepath)
        check = b * 2

        np.testing.assert_array_equal(guess, check)

    def test_sub(self):
        """Test Subtract Operation."""
        a = np.ma.masked_array(np.ones((1, 1)))
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1)))
        r_2 = create_test_raster(b)

        guess_filepath = utils.sub(r_1, r_2)
        guess = read_array(guess_filepath)
        check = 0

        np.testing.assert_array_equal(guess, check)

    def test_mul(self):
        """Test Multiply Operation."""
        a = np.ma.masked_array(np.ones((1, 1))*2)
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1))*2)
        r_2 = create_test_raster(b)

        guess_filepath = utils.mul(r_1, r_2)
        guess = read_array(guess_filepath)
        check = 4

        np.testing.assert_array_equal(guess, check)

    def test_mul_scalar(self):
        """Test Multiply Scalar Operation."""
        a = np.ma.masked_array(np.ones((1, 1)))
        r_1 = create_test_raster(a)

        guess_filepath = utils.mul_scalar(r_1, 2)
        guess = read_array(guess_filepath)
        check = 2

        np.testing.assert_array_equal(guess, check)

    def test_div(self):
        """Test Divide Operation."""
        a = np.ma.masked_array(np.ones((1, 1))*2)
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1))*2)
        r_2 = create_test_raster(b)

        guess_filepath = utils.div(r_1, r_2)
        guess = read_array(guess_filepath)
        check = 1

        np.testing.assert_array_equal(guess, check)

    def test_pow(self):
        """Test Power Operation."""
        a = np.ma.masked_array(np.ones((1, 1))*2)
        r_1 = create_test_raster(a)
        b = np.ma.masked_array(np.ones((1, 1))*2)
        r_2 = create_test_raster(b)

        guess_filepath = utils.pow(r_1, r_2)
        guess = read_array(guess_filepath)
        check = 4

        np.testing.assert_array_equal(guess, check)

    def test_zeros(self):
        """Test Create Zero Raster Operation."""
        a = np.ma.masked_array(np.ones((1, 1)))
        r_1 = create_test_raster(a)

        guess_filepath = utils.zeros(r_1)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array(np.zeros((1, 1)))

        np.testing.assert_array_equal(guess, check)

    def test_copy(self):
        """Test Copy Function."""
        a = np.ma.masked_array(np.ones((1, 1)))
        r_1 = create_test_raster(a)

        guess_filepath = utils.copy(r_1)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array(np.ones((1, 1)))

        np.testing.assert_array_equal(guess, check)

    def test_nodata_to_zeros(self):
        """Test Nodata to Zeros Function."""
        a = np.ma.masked_array(np.ones((1, 1)))
        a[:] = -9999
        r_1 = create_test_raster(a)

        guess_filepath = utils.nodata_to_zeros(r_1)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array(np.zeros((1, 1)))

        np.testing.assert_array_equal(guess, check)

    def test_compute_E_pt(self):
        """Test Compute Emissions Function."""
        D_pr = np.ma.masked_array([[10.0]])
        D_pr = create_test_raster(D_pr)
        H_pr = np.ma.masked_array([[1.0]])
        H_pr = create_test_raster(H_pr)
        offset_t = 0
        end_t = 1

        guess_filepath = utils.compute_E_pt(D_pr, H_pr, offset_t, end_t)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array([[5.]])

        np.testing.assert_array_equal(guess, check)

    def test_compute_V_t(self):
        """Test Compute Net Present Value Function."""
        N_t = np.ma.masked_array([[1.0]])
        N_t = create_test_raster(N_t)
        price_t = 0.66666667

        guess_filepath = utils.compute_V_t(N_t, price_t)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array([[0.66666667]])

        np.testing.assert_array_almost_equal(guess, check)

    def test_compute_T_b(self):
        """Test Compute Total Stock Function."""
        T_b = np.ma.masked_array([[10.5]])
        T_b = create_test_raster(T_b)
        N_r = np.ma.masked_array([[1.0]])
        N_r = create_test_raster(N_r)
        L_s_0 = np.ma.masked_array([[0.5]])
        L_s_0 = create_test_raster(L_s_0)
        L_s_1 = np.ma.masked_array([[1.0]])
        L_s_1 = create_test_raster(L_s_1)

        guess_filepath = utils.compute_T_b(T_b, N_r, L_s_0, L_s_1)
        guess = read_array(guess_filepath)
        check = np.ma.masked_array([[12.0]])

        np.testing.assert_array_almost_equal(guess, check)

    def tearDown(self):
        """About."""
        pass


if __name__ == '__main__':
    unittest.main()
