"""InVEST urban stormwater retention model tests."""
import functools
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy
from osgeo import gdal, osr
import pandas
import pygeoprocessing
from pygeoprocessing.geoprocessing_core import (
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS as opts_tuple)


TEST_DATA = os.path.join(os.path.dirname(
    __file__), '..', 'data', 'invest-test-data', 'stormwater')


def to_raster(array, path, nodata=-1, pixel_size=(20, -20), origin=(0, 0),
              epsg=3857, raster_driver_creation_tuple=opts_tuple):
    """Wrap around pygeoprocessing.numpy_array_to_raster to set defaults.

    Sets some reasonable defaults for ``numpy_array_to_raster`` and takes care
    of setting up a WKT spatial reference so that it can be done in one line.

    Args:
        array (numpy.ndarray): array to be written to ``path`` as a raster
        path (str): raster path to write ``array` to
        nodata (float): nodata value to pass to ``numpy_array_to_raster``
        pixel_size (tuple(float, float)): pixel size value to pass to
            ``numpy_array_to_raster``
        origin (tuple(float, float)): origin value to pass to
            ``numpy_array_to_raster``
        epsg (int): EPSG code used to instantiate a spatial reference that is
            passed to ``numpy_array_to_raster`` in WKT format
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second.

    Returns:
        None
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    projection_wkt = srs.ExportToWkt()
    pygeoprocessing.numpy_array_to_raster(
        array,
        nodata,
        pixel_size,
        origin,
        projection_wkt,
        path)


def mock_iterblocks(*args, **kwargs):
    """Mock function for pygeoprocessing.iterblocks that yields custom blocks.

    Args:
        xoffs (list[int]): list of x-offsets for each block in order
        xsizes (list[int]): list of widths for each block in order
        yoffs (list[int]): list of y-offsets for each block in order
        ysizes (list[int]): list of heights for each block in order

        For python 3.7 compatibility, these have to be extracted from the
        kwargs dictionary (can't have keyword-only arguments).

    Yields:
        dictionary with keys 'xoff', 'yoff', 'win_xsize', 'win_ysize'
        that have the same meaning as in pygeoprocessing.iterblocks.
    """
    for yoff, ysize in zip(kwargs['yoffs'], kwargs['ysizes']):
        for xoff, xsize in zip(kwargs['xoffs'], kwargs['xsizes']):
            yield {
                'xoff': xoff,
                'yoff': yoff,
                'win_xsize': xsize,
                'win_ysize': ysize}


class StormwaterTests(unittest.TestCase):
    """Tests for InVEST stormwater model."""

    def setUp(self):
        """Create a temp directory for the workspace."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def basic_setup(workspace_dir, pe=False):
        """
        Set up for the full model run tests.

        Args:
            workspace_dir (str): path to a directory in which to create files
            pe (bool): if True, include PE data in the biophysical table

        Returns:
            List of the data and files that were created, in this order:

            0 (numpy.ndarray): array written to the biophysical table path
            1 (str): path to the biophysical table csv
            2 (numpy.ndarray): array of LULC values written to the LULC path
            3 (str): path to the LULC raster
            4 (numpy.ndarray): array of soil group values written to the soil
                group raster path
            5 (str): path to the soil group raster
            6 (numpy.ndarray): array of precipitation values written to the
                precipitation raster path
            7 (str): path to the precipitation raster
            8 (float): stormwater retention cost value per cubic meter
            9 (float): pixel area for all the rasters created
        """
        # In practice RC_X + PE_X <= 1, but they are independent in the model,
        # so ignoring that constraint for convenience.
        biophysical_dict = {
            'lucode': [0, 1, 11, 12],
            'EMC_pollutant1': [2.55, 0, 1, 5],
            'RC_A': [0, 0.15, 0.1, 1],
            'RC_B': [0, 0.25, 0.2, 1],
            'RC_C': [0, 0.35, 0.3, 1],
            'RC_D': [0, 0.45, 0.4, 1],
            'is_connected': [0, 0, 0, 1]
        }
        if pe:
            biophysical_dict.update({
                'PE_A': [0, 0.55, 0.5, 1],
                'PE_B': [0, 0.65, 0.6, 1],
                'PE_C': [0, 0.75, 0.7, 1],
                'PE_D': [0, 0.85, 0.8, 1]
            })

        biophysical_table = pandas.DataFrame(
            biophysical_dict).set_index(['lucode'])
        retention_cost = 2.53

        lulc_array = numpy.array([
            [0,  0,  0,  0],
            [1,  1,  1,  1],
            [11, 11, 11, 11],
            [12, 12, 12, 12]], dtype=numpy.uint8)
        soil_group_array = numpy.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]], dtype=numpy.uint8)
        precipitation_array = numpy.array([
            [0,    0,    0,    0],
            [0,    0,    0,    0],
            [12.5, 12.5, 12.5, 12.5],
            [12.5, 12.5, 12.5, 12.5]], dtype=numpy.float32)
        lulc_path = os.path.join(workspace_dir, 'lulc.tif')
        soil_group_path = os.path.join(workspace_dir, 'soil_group.tif')
        precipitation_path = os.path.join(workspace_dir, 'precipitation.tif')
        biophysical_table_path = os.path.join(workspace_dir, 'biophysical.csv')

        pixel_size = (20, -20)
        pixel_area = abs(pixel_size[0] * pixel_size[1])
        # save each dataset to a file
        for (array, path) in [
                (lulc_array, lulc_path),
                (soil_group_array, soil_group_path),
                (precipitation_array, precipitation_path)]:
            to_raster(array, path, pixel_size=pixel_size)
        biophysical_table.to_csv(biophysical_table_path)

        return [
            biophysical_table,
            biophysical_table_path,
            lulc_array,
            lulc_path,
            soil_group_array,
            soil_group_path,
            precipitation_array,
            precipitation_path,
            retention_cost,
            pixel_area
        ]

    def test_basic(self):
        """Stormwater: basic model run."""
        from natcap.invest import stormwater

        (biophysical_table,
         biophysical_table_path,
         lulc_array,
         lulc_path,
         soil_group_array,
         soil_group_path,
         precipitation_array,
         precipitation_path,
         retention_cost,
         pixel_area) = self.basic_setup(self.workspace_dir)

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': 'suffix',
            'lulc_path': lulc_path,
            'soil_group_path': soil_group_path,
            'precipitation_path': precipitation_path,
            'biophysical_table': biophysical_table_path,
            'adjust_retention_ratios': False,
            'retention_radius': None,
            'road_centerlines_path': None,
            'aggregate_areas_path': None,
            'replacement_cost': retention_cost
        }

        soil_group_codes = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

        stormwater.execute(args)

        retention_volume_path = os.path.join(
            self.workspace_dir, 'retention_volume_suffix.tif')
        percolation_volume_path = os.path.join(
            self.workspace_dir, 'percolation_volume_suffix.tif')
        pollutant_path = os.path.join(
            self.workspace_dir, 'avoided_pollutant_load_pollutant1_suffix.tif')
        value_path = os.path.join(
            self.workspace_dir, 'retention_value_suffix.tif')
        # there should be no percolation output because there's no
        # percolation data in the biophysical table
        self.assertFalse(os.path.exists(percolation_volume_path))

        retention_raster = gdal.OpenEx(retention_volume_path, gdal.OF_RASTER)
        retention_volume = retention_raster.GetRasterBand(1).ReadAsArray()

        avoided_pollutant_raster = gdal.OpenEx(pollutant_path, gdal.OF_RASTER)
        avoided_pollutant_load = avoided_pollutant_raster.GetRasterBand(
            1).ReadAsArray()

        retention_value_raster = gdal.OpenEx(value_path, gdal.OF_RASTER)
        retention_value = retention_value_raster.GetRasterBand(1).ReadAsArray()

        for row in range(retention_volume.shape[0]):
            for col in range(retention_volume.shape[1]):

                soil_group = soil_group_array[row, col]
                lulc = lulc_array[row, col]
                precipitation = precipitation_array[row, col]

                rc_value = biophysical_table[
                    f'RC_{soil_group_codes[soil_group]}'][lulc]

                # precipitation (mm/yr) * 0.001 (m/mm) * pixel area (m^2) =
                # m^3/yr
                actual_retention_volume = retention_volume[row, col]
                expected_retention_volume = (1 - rc_value) * \
                    precipitation * 0.001 * pixel_area
                numpy.testing.assert_allclose(
                    actual_retention_volume,
                    expected_retention_volume, rtol=1e-6)

                # retention (m^3/yr) * cost ($/m^3) = value ($/yr)
                actual_value = retention_value[row, col]
                expected_value = expected_retention_volume * retention_cost
                numpy.testing.assert_allclose(
                    actual_value, expected_value, rtol=1e-6)

        for row in range(avoided_pollutant_load.shape[0]):
            for col in range(avoided_pollutant_load.shape[1]):

                lulc = lulc_array[row, col]
                retention = retention_volume[row, col]
                emc = biophysical_table['EMC_pollutant1'][lulc]

                # retention (m^3/yr) * emc (mg/L) * 1000 (L/m^3) * 0.000001
                # (kg/mg) = kg/yr
                avoided_load = avoided_pollutant_load[row, col]
                expected_avoided_load = retention * emc * 0.001
                numpy.testing.assert_allclose(
                    avoided_load, expected_avoided_load, rtol=1e-6)

    def test_pe(self):
        """Stormwater: full model run with PE data in biophysical table."""
        from natcap.invest import stormwater

        (biophysical_table,
         biophysical_table_path,
         lulc_array,
         lulc_path,
         soil_group_array,
         soil_group_path,
         precipitation_array,
         precipitation_path,
         retention_cost,
         pixel_area) = self.basic_setup(self.workspace_dir, pe=True)

        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_path': lulc_path,
            'soil_group_path': soil_group_path,
            'precipitation_path': precipitation_path,
            'biophysical_table': biophysical_table_path,
            'adjust_retention_ratios': False,
            'retention_radius': None,
            'road_centerlines_path': None,
            'aggregate_areas_path': None,
            'replacement_cost': retention_cost
        }

        soil_group_codes = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

        stormwater.execute(args)

        retention_volume_path = os.path.join(
            self.workspace_dir,
            stormwater.FINAL_OUTPUTS['retention_volume_path'])
        percolation_volume_path = os.path.join(
            self.workspace_dir,
            stormwater.FINAL_OUTPUTS['percolation_volume_path'])
        value_path = os.path.join(
            self.workspace_dir,
            stormwater.FINAL_OUTPUTS['retention_value_path'])

        retention_raster = gdal.OpenEx(retention_volume_path, gdal.OF_RASTER)
        retention_volume = retention_raster.GetRasterBand(1).ReadAsArray()

        percolation_raster = gdal.OpenEx(
            percolation_volume_path, gdal.OF_RASTER)
        percolation_volume = percolation_raster.GetRasterBand(
            1).ReadAsArray()

        retention_value_raster = gdal.OpenEx(value_path, gdal.OF_RASTER)
        retention_value = retention_value_raster.GetRasterBand(1).ReadAsArray()

        for row in range(retention_volume.shape[0]):
            for col in range(retention_volume.shape[1]):

                soil_group = soil_group_array[row, col]
                lulc = lulc_array[row, col]
                precipitation = precipitation_array[row, col]

                rc_value = biophysical_table[
                    f'RC_{soil_group_codes[soil_group]}'][lulc]

                # precipitation (mm/yr) * 0.001 (m/mm) * pixel area (m^2) =
                # m^3/yr
                actual_volume = retention_volume[row, col]
                expected_volume = (1 - rc_value) * \
                    precipitation * 0.001 * pixel_area
                numpy.testing.assert_allclose(actual_volume, expected_volume,
                                              rtol=1e-6)

                # retention (m^3/yr) * cost ($/m^3) = value ($/yr)
                actual_value = retention_value[row, col]
                expected_value = expected_volume * retention_cost
                numpy.testing.assert_allclose(actual_value, expected_value,
                                              rtol=1e-6)

        for row in range(percolation_volume.shape[0]):
            for col in range(percolation_volume.shape[1]):

                soil_group = soil_group_array[row][col]
                lulc = lulc_array[row][col]
                precipitation = precipitation_array[row][col]

                pe_value = biophysical_table[
                    f'PE_{soil_group_codes[soil_group]}'][lulc]

                # precipitation (mm/yr) * 0.001 (m/mm) * pixel area (m^2) = m^3
                expected_volume = (pe_value) * \
                    precipitation * 0.001 * pixel_area
                numpy.testing.assert_allclose(percolation_volume[row][col],
                                              expected_volume, rtol=1e-6)

    def test_adjust(self):
        """Stormwater: full model run with adjust retention ratios."""
        from natcap.invest import stormwater

        (biophysical_table,
         biophysical_table_path,
         lulc_array,
         lulc_path,
         soil_group_array,
         soil_group_path,
         precipitation_array,
         precipitation_path,
         retention_cost,
         pixel_area) = self.basic_setup(self.workspace_dir)

        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_path': lulc_path,
            'soil_group_path': soil_group_path,
            'precipitation_path': precipitation_path,
            'biophysical_table': biophysical_table_path,
            'adjust_retention_ratios': True,
            'retention_radius': 30,
            'road_centerlines_path': os.path.join(
                TEST_DATA, 'centerlines.gpkg'),
            'aggregate_areas_path': None,
            'replacement_cost': retention_cost
        }
        stormwater.execute(args)

        adjusted_ratio_raster = gdal.OpenEx(
            os.path.join(
                self.workspace_dir,
                stormwater.FINAL_OUTPUTS['adjusted_retention_ratio_path']),
            gdal.OF_RASTER)
        retention_volume_raster = gdal.OpenEx(
            os.path.join(
                self.workspace_dir,
                stormwater.FINAL_OUTPUTS['retention_volume_path']),
            gdal.OF_RASTER)
        runoff_volume_raster = gdal.OpenEx(
            os.path.join(
                self.workspace_dir,
                stormwater.FINAL_OUTPUTS['runoff_volume_path']),
            gdal.OF_RASTER)
        actual_runoff_volume = runoff_volume_raster.GetRasterBand(
            1).ReadAsArray()
        actual_adjusted_ratios = adjusted_ratio_raster.GetRasterBand(
            1).ReadAsArray()
        actual_retention_volume = retention_volume_raster.GetRasterBand(
            1).ReadAsArray()

        expected_adjusted_ratios = numpy.array([
            [1,      1,      1,        1],
            [0.9825, 0.9625, 0.924167, 0.8875],
            [0.9,    0.8,    0.7,      0.6],
            [0,      0,      0,        0]], dtype=numpy.float32)
        numpy.testing.assert_allclose(actual_adjusted_ratios,
                                      expected_adjusted_ratios, rtol=1e-6)
        expected_retention_volume = (expected_adjusted_ratios *
                                     precipitation_array * pixel_area * 0.001)
        numpy.testing.assert_allclose(actual_retention_volume,
                                      expected_retention_volume, rtol=1e-6)
        expected_runoff_volume = ((1 - expected_adjusted_ratios) *
                                  precipitation_array * pixel_area * 0.001)
        numpy.testing.assert_allclose(actual_runoff_volume,
                                      expected_runoff_volume, rtol=1e-6)

    def test_aggregate(self):
        """Stormwater: full model run with aggregate results."""
        from natcap.invest import stormwater
        (biophysical_table,
         biophysical_table_path,
         lulc_array,
         lulc_path,
         soil_group_array,
         soil_group_path,
         precipitation_array,
         precipitation_path,
         retention_cost,
         pixel_area) = self.basic_setup(self.workspace_dir, pe=True)

        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_path': lulc_path,
            'soil_group_path': soil_group_path,
            'precipitation_path': precipitation_path,
            'biophysical_table': biophysical_table_path,
            'adjust_retention_ratios': False,
            'retention_radius': None,
            'road_centerlines_path': None,
            'aggregate_areas_path': os.path.join(TEST_DATA, 'aoi.gpkg'),
            'replacement_cost': retention_cost
        }
        stormwater.execute(args)

        expected_feature_fields = {
            1: {
                'mean_retention_ratio': 0.825,
                'mean_runoff_ratio': 0.175,
                'mean_percolation_ratio': 0.575,
                'total_retention_volume': 8.5,
                'total_runoff_volume': 1.5,
                'total_percolation_volume': 5.5,
                'pollutant1_total_avoided_load': .0085,
                'pollutant1_total_load': .0015,
                'total_retention_value': 21.505
            },
            2: {
                'mean_retention_ratio': 0.5375,
                'mean_runoff_ratio': 0.4625,
                'mean_percolation_ratio': 0.7625,
                'total_retention_volume': 7.5,
                'total_runoff_volume': 7.5,
                'total_percolation_volume': 11.5,
                'pollutant1_total_avoided_load': .0075,
                'pollutant1_total_load': .0275,
                'total_retention_value': 18.975
            },
            3: {
                'mean_retention_ratio': 0,
                'total_retention_volume': 0,
                'mean_runoff_ratio': 0,
                'total_runoff_volume': 0,
                'mean_percolation_ratio': 0,
                'total_percolation_volume': 0,
                'pollutant1_total_avoided_load': 0,
                'pollutant1_total_load': 0,
                'total_retention_value': 0
            }
        }

        aggregate_data_path = os.path.join(
            args['workspace_dir'],
            stormwater.FINAL_OUTPUTS['reprojected_aggregate_areas_path'])
        aggregate_vector = gdal.OpenEx(aggregate_data_path, gdal.OF_VECTOR)
        aggregate_layer = aggregate_vector.GetLayer()
        for feature in aggregate_layer:
            feature_id = feature.GetFID()
            for key, val in expected_feature_fields[feature_id].items():
                field_value = feature.GetField(key)
                numpy.testing.assert_allclose(field_value, val, rtol=1e-6)

    def test_lookup_ratios(self):
        """Stormwater: test lookup_ratios function."""
        from natcap.invest import stormwater

        sorted_lucodes = [10, 11, 12, 13]
        lulc_array = numpy.array([
            [13, 12],
            [11, 10]], dtype=numpy.uint8)
        soil_group_array = numpy.array([
            [4, 4],
            [2, 2]], dtype=numpy.uint8)
        lulc_path = os.path.join(self.workspace_dir, 'lulc.tif')
        soil_group_path = os.path.join(self.workspace_dir, 'soil_groups.tif')
        output_path = os.path.join(self.workspace_dir, 'out.tif')
        to_raster(lulc_array, lulc_path, nodata=255)
        to_raster(soil_group_array, soil_group_path, nodata=255)
        # rows correspond to sorted lucodes, columns to soil groups A-D
        ratio_array = numpy.array([
            [0.11, 0.12, 0.13, 0.14],
            [0.21, 0.22, 0.23, 0.24],
            [0.31, 0.32, 0.33, 0.34],
            [0.41, 0.42, 0.43, 0.44]], dtype=numpy.float32)
        expected_output = numpy.array([
            [0.44, 0.34],
            [0.22, 0.12]], dtype=numpy.float32)
        stormwater.lookup_ratios(
            lulc_path,
            soil_group_path,
            ratio_array,
            sorted_lucodes,
            output_path)
        actual_output = pygeoprocessing.raster_to_numpy_array(output_path)
        numpy.testing.assert_allclose(expected_output, actual_output)

    def test_volume_op(self):
        """Stormwater: test volume_op function."""
        from natcap.invest import stormwater

        precip_nodata = -2.5
        ratio_array = numpy.array([
            [0,   0.0001, stormwater.FLOAT_NODATA],
            [0.5, 0.9,    1]], dtype=numpy.float32)
        precip_array = numpy.array([
            [10.5, 0, 1],
            [0.5,  0, precip_nodata]], dtype=numpy.float32)
        pixel_area = 400

        out = stormwater.volume_op(
            ratio_array,
            precip_array,
            precip_nodata,
            pixel_area)
        # precip (mm/yr) * area (m^2) * 0.001 (m/mm) * ratio = volume (m^3/yr)
        for y in range(ratio_array.shape[0]):
            for x in range(ratio_array.shape[1]):
                if (ratio_array[y, x] == stormwater.FLOAT_NODATA or
                        precip_array[y, x] == precip_nodata):
                    numpy.testing.assert_allclose(
                        out[y, x],
                        stormwater.FLOAT_NODATA)
                else:
                    numpy.testing.assert_allclose(
                        out[y, x],
                        precip_array[y, x]*ratio_array[y, x]*pixel_area/1000)

    def test_pollutant_load_op(self):
        """Stormwater: test pollutant_load_op function."""
        from natcap.invest import stormwater

        # test with nodata values greater and less than the LULC codes
        # there was a bug that only happened with a larger nodata value
        for lulc_nodata in [-1, 127]:
            with self.subTest(lulc_nodata=lulc_nodata):
                lulc_array = numpy.array([
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, lulc_nodata]], dtype=numpy.int8)
                retention_volume_array = numpy.array([
                    [0, 1.5, stormwater.FLOAT_NODATA],
                    [0, 1.5, 100],
                    [0, 1.5, 100]], dtype=numpy.float32)
                sorted_lucodes = numpy.array([0, 1, 2], dtype=numpy.uint8)
                emc_array = numpy.array([0, 0.5, 3], dtype=numpy.float32)

                out = stormwater.pollutant_load_op(
                    lulc_array,
                    lulc_nodata,
                    retention_volume_array,
                    sorted_lucodes,
                    emc_array)
                for y in range(lulc_array.shape[0]):
                    for x in range(lulc_array.shape[1]):
                        if (lulc_array[y, x] == lulc_nodata or
                                retention_volume_array[y, x] == stormwater.FLOAT_NODATA):
                            numpy.testing.assert_allclose(
                                out[y, x], stormwater.FLOAT_NODATA)
                        else:
                            emc_value = emc_array[lulc_array[y, x]]
                            expected = emc_value * \
                                retention_volume_array[y, x] / 1000
                            numpy.testing.assert_allclose(out[y, x], expected)

    def test_retention_value_op(self):
        """Stormwater: test retention_value_op function."""
        from natcap.invest import stormwater

        retention_volume_array = numpy.array([
            [0, 1.5, stormwater.FLOAT_NODATA],
            [0, 1.5, 100]], dtype=numpy.float32)
        replacement_cost = 1.5
        expected = numpy.array([
            [0, 2.25, stormwater.FLOAT_NODATA],
            [0, 2.25, 150]], dtype=numpy.float32)
        actual = stormwater.retention_value_op(
            retention_volume_array,
            replacement_cost)
        numpy.testing.assert_allclose(actual, expected)

    def test_adjust_op(self):
        """Stormwater: test adjust_op function."""
        from natcap.invest import stormwater

        ratio_array = numpy.array([
            [0,   0.0001, stormwater.FLOAT_NODATA],
            [0.5, 0.9,    1]], dtype=numpy.float32)
        # these are obviously not averages from the above array but
        # it doesn't matter for this test
        avg_ratio_array = numpy.array([
            [0.5, 0.5, 0.5],
            [0.5, stormwater.FLOAT_NODATA, 0.5]], dtype=numpy.float32)
        near_connected_lulc_array = numpy.array([
            [0, 0, 1],
            [stormwater.UINT8_NODATA, 0, 1]], dtype=numpy.uint8)
        near_road_centerline_array = numpy.array([
            [1, 1, 1],
            [0, 0, 0]], dtype=numpy.uint8)

        out = stormwater.adjust_op(
            ratio_array,
            avg_ratio_array,
            near_connected_lulc_array,
            near_road_centerline_array)
        for y in range(ratio_array.shape[0]):
            for x in range(ratio_array.shape[1]):
                if (ratio_array[y, x] == stormwater.FLOAT_NODATA or
                    avg_ratio_array[y, x] == stormwater.FLOAT_NODATA or
                    near_connected_lulc_array[y, x] == stormwater.UINT8_NODATA or
                        near_road_centerline_array[y, x] == stormwater.UINT8_NODATA):
                    numpy.testing.assert_allclose(
                        out[y, x], stormwater.FLOAT_NODATA)
                else:
                    # equation 2-4: Radj_ij = R_ij + (1 - R_ij) * C_ij
                    adjust_factor = (
                        0 if (
                            near_connected_lulc_array[y, x] or
                            near_road_centerline_array[y, x]
                        ) else avg_ratio_array[y, x])
                    adjusted = (ratio_array[y, x] +
                                (1 - ratio_array[y, x]) * adjust_factor)
                    numpy.testing.assert_allclose(out[y, x], adjusted,
                                                  rtol=1e-6)

    def test_is_near(self):
        """Stormwater: test is_near function."""
        from natcap.invest import stormwater
        is_connected_array = numpy.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=numpy.uint8)
        radius = 1  # 1 pixel
        # search kernel:
        # [0, 1, 0],
        # [1, 1, 1],
        # [0, 1, 0]
        # convolution sum array:
        # [1, 1, 2, 1, 0, 0],
        # [1, 1, 2, 1, 0, 1],
        # [1, 0, 1, 0, 1, 1]
        # expected is_near array: sum > 0
        expected = numpy.array([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1]
        ], dtype=numpy.uint8)

        connected_path = os.path.join(self.workspace_dir, 'connected.tif')
        distance_path = os.path.join(self.workspace_dir, 'distance.tif')
        out_path = os.path.join(self.workspace_dir, 'near_connected.tif')
        to_raster(is_connected_array, connected_path, pixel_size=(10, -10))

        mocked = functools.partial(mock_iterblocks, yoffs=[0], ysizes=[3],
                                   xoffs=[0, 3], xsizes=[3, 3])
        with mock.patch('natcap.invest.stormwater.pygeoprocessing.iterblocks',
                        mocked):
            stormwater.is_near(connected_path, radius, distance_path, out_path)
            actual = pygeoprocessing.raster_to_numpy_array(out_path)
            numpy.testing.assert_equal(expected, actual)

    def test_make_search_kernel(self):
        """Stormwater: test make_search_kernel function."""
        from natcap.invest import stormwater

        array = numpy.zeros((10, 10))
        path = os.path.join(self.workspace_dir, 'make_search_kernel.tif')
        to_raster(array, path, pixel_size=(10, -10))

        expected_5 = numpy.array([[1]], dtype=numpy.uint8)
        actual_5 = stormwater.make_search_kernel(path, 5)
        numpy.testing.assert_equal(expected_5, actual_5)

        expected_9 = numpy.array([[1]], dtype=numpy.uint8)
        actual_9 = stormwater.make_search_kernel(path, 9)
        numpy.testing.assert_equal(expected_9, actual_9)

        expected_10 = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=numpy.uint8)
        actual_10 = stormwater.make_search_kernel(path, 10)
        numpy.testing.assert_equal(expected_10, actual_10)

        expected_15 = numpy.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]], dtype=numpy.uint8)
        actual_15 = stormwater.make_search_kernel(path, 15)
        numpy.testing.assert_equal(expected_15, actual_15)

    def test_raster_average(self):
        """Stormwater: test raster_average function."""
        from natcap.invest import stormwater

        array = numpy.empty((150, 150))
        nodata = -1
        array[:, 0:128] = 10
        array[:, 128:149] = 20
        array[:, 149] = nodata

        data_path = os.path.join(self.workspace_dir, 'data.tif')
        kernel_path = os.path.join(self.workspace_dir, 'kernel.tif')
        average_path = os.path.join(self.workspace_dir, 'average.tif')
        to_raster(array, data_path, pixel_size=(10, -10))
        stormwater.raster_average(data_path, 11, kernel_path, average_path)

        expected_kernel = numpy.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], dtype=numpy.uint8)
        actual_kernel = pygeoprocessing.raster_to_numpy_array(kernel_path)
        numpy.testing.assert_equal(actual_kernel, expected_kernel)

        actual_average = pygeoprocessing.raster_to_numpy_array(average_path)
        expected_average = numpy.empty((150, 150))
        expected_average[:, 0:127] = 10
        expected_average[:, 127] = 12
        expected_average[0, 127] = 12.5
        expected_average[-1, 127] = 12.5
        expected_average[:, 128] = 18
        expected_average[0, 128] = 17.5
        expected_average[-1, 128] = 17.5
        expected_average[:, 129:149] = 20
        expected_average[:, 149] = -1
        numpy.testing.assert_allclose(actual_average, expected_average)

    def test_validate(self):
        """Stormwater: test arg validation."""
        from natcap.invest import stormwater, validation

        # test args missing necessary values for adjust ratios
        args = {
            'workspace_dir': self.workspace_dir,
            'lulc_path': 'x',
            'soil_group_path': 'x',
            'precipitation_path': 'x',
            'biophysical_table': 'x',
            'adjust_retention_ratios': True,
            'retention_radius': None,
            'road_centerlines_path': None,
            'aggregate_areas_path': None,
            'replacement_cost': None
        }
        messages = stormwater.validate(args)
        for arg_list, message in messages:
            if arg_list[0] in ['retention_radius', 'road_centerlines_path']:
                self.assertEqual(message, validation.MESSAGES['MISSING_VALUE'])

    def test_lulc_signed_byte(self):
        """Stormwater: regression test for handling signed byte LULC input."""
        from natcap.invest import stormwater

        (_,
         biophysical_table_path, _, _, _,
         soil_group_path, _,
         precipitation_path,
         retention_cost,
         pixel_size) = self.basic_setup(self.workspace_dir)

        # make custom lulc raster with signed byte type
        lulc_array = numpy.array([
            [0,  0,  0,  0],
            [1,  1,  1,  1],
            [11, 11, 11, 11],
            [12, 12, 12, 12]], dtype=numpy.int8)
        lulc_path = os.path.join(self.workspace_dir, 'lulc.tif')
        signed_byte_creation_opts = opts_tuple[1] + ('PIXELTYPE=SIGNEDBYTE',)
        to_raster(
            lulc_array,
            lulc_path,
            raster_driver_creation_tuple=(
                opts_tuple[0], signed_byte_creation_opts
            )
        )

        args = {
            'workspace_dir': self.workspace_dir,
            'results_suffix': '',
            'lulc_path': lulc_path,
            'soil_group_path': soil_group_path,
            'precipitation_path': precipitation_path,
            'biophysical_table': biophysical_table_path,
            'adjust_retention_ratios': True,
            'retention_radius': 20,
            'road_centerlines_path': os.path.join(
                TEST_DATA, 'centerlines.gpkg'),
            'aggregate_areas_path': None,
            'replacement_cost': retention_cost
        }

        stormwater.execute(args)

        # assert that not all distances to roads are zero
        # this problem resulted from not handling signed byte rasters
        # when calling `new_raster_from_base`
        road_distance_path = os.path.join(
            self.workspace_dir, 'intermediate', 'road_distance.tif')
        distance_is_zero = pygeoprocessing.raster_to_numpy_array(
            road_distance_path) == 0
        self.assertFalse(numpy.all(distance_is_zero))
