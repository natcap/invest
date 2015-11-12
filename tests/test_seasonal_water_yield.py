"""InVEST Seasonal water yield model tests that use the InVEST sample data"""

import unittest
import tempfile
import shutil
import os

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data', 'seasonal_water_yield')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'seasonal_water_yield')


def _test_same_files(base_list_path, path):
    """Assert that the files listed in `base_list_file` are also in the
    directory pointed to by `path`.

    Parameters:
        base_list_file (string): a path to a file that has one relative file
            path per line.
        path (string): a path to a directory whose contents will be checked
            against the files listed in `base_list_file`

    Returns:
        None

    Raises:
        AssertionError when there are files listed in `base_list_file` that
            don't exist in the directory indicated by `path`"""

    missing_files = []
    with open(base_list_path, 'r') as file_list:
        for file_path in file_list:
            if not os.path.isfile(os.path.join(path, file_path)):
                missing_files.append(os.path.join(path, file_path))
    if len(missing_files) > 0:
        raise AssertionError(
            "The following files were expected but not found: " +
            '\n'.join(missing_files))


class ExampleTest(unittest.TestCase):
    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_base_regression(self):
        """
        SWY base regression test
        """
        from natcap.invest.seasonal_water_yield import seasonal_water_yield

        args = {
            u'alpha_m': u'1/12',
            u'aoi_path': os.path.join(SAMPLE_DATA, 'watershed.shp'),
            u'beta_i': u'1.0',
            u'biophysical_table_path': os.path.join(SAMPLE_DATA, 'biophysical_table.csv'),
            u'climate_zone_raster_path': os.path.join(SAMPLE_DATA, 'climate_zones.tif'),
            u'climate_zone_table_path': os.path.join(SAMPLE_DATA, 'climate_zone_events.csv'),
            u'dem_raster_path': os.path.join(SAMPLE_DATA, 'dem.tif'),
            u'et0_dir': os.path.join(SAMPLE_DATA, 'eto_dir'),
            u'gamma': u'1.0',
            u'lulc_raster_path': os.path.join(SAMPLE_DATA, 'lulc.tif'),
            u'precip_dir': os.path.join(SAMPLE_DATA, 'precip_dir'),
            u'rain_events_table_path': os.path.join(SAMPLE_DATA, 'rain_events_table.csv'),
            u'results_suffix': '',
            u'soil_group_path': os.path.join(SAMPLE_DATA, 'soil_group.tif'),
            u'threshold_flow_accumulation': u'1000',
            u'user_defined_climate_zones': False,
            u'user_defined_local_recharge': False,
            u'workspace_dir': tempfile.mkdtemp(),
        }

        seasonal_water_yield.execute(args)

        # Test that the workspace has the same files as we expect
        _test_same_files(
            os.path.join(REGRESSION_DATA, 'file_list_base.txt'),
            args['workspace_dir'])

        # The output baseflow is equal to the regression baseflow raster
        pygeoprocessing.testing.assert_rasters_equal(
            os.path.join(args['workspace_dir'], 'B.tif'),
            os.path.join(REGRESSION_DATA, 'B.tif'))

        shutil.rmtree(args['workspace_dir'])
