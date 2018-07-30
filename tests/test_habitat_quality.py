"""Module for Regression Testing the InVEST Habitat Quality model."""
import unittest
import tempfile
import shutil
import os
# import logging
# logging.basicConfig(level=logging.DEBUG)
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import pygeoprocessing.testing
import natcap.invest.pygeoprocessing_0_3_3.testing
from natcap.invest.pygeoprocessing_0_3_3.testing import scm

# temporary imports
import pdb
temp_dir = r"C:\Users\chiay\Documents\invest_fork\habitat_quality_test"

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'habitat_quality')


def make_simple_poly(origin):
    """Make a 10x10 ogr rectangular geometry clockwisely from origin."""
    # Create a rectangular ring
    lon, lat = origin[0], origin[1]
    width = 10
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(lon, lat)
    ring.AddPoint(lon + width, lat)
    ring.AddPoint(lon + width, lat - width)
    ring.AddPoint(lon, lat - width)
    ring.AddPoint(lon, lat)

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def make_simple_raster(array, raster_path):
    """Make a raster from an array on a designated path."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # UTM Zone 10N
    project_wkt = srs.ExportToWkt()

    pygeoprocessing.testing.create_raster_on_disk(
        [array],
        (1180000, 690000),  # Origin same as access_samp.shp
        project_wkt,
        -1,
        (1000, -1000),
        filename=raster_path)


def make_access_shp(access_shp_path):
    # Set up parameters. Fid and access values are based on the sample data
    fid_list = [0.0, 1.0, 2.0]
    access_list = [0.2, 0.8, 1.0]
    coord_list = [(1180000.0, 690000.0 - i * 10) for i in range(3)]  # 30x10 px
    poly_list = [make_simple_poly(coord) for coord in coord_list]

    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.CreateDataSource(access_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # Spatial reference UTM Zone 10N
    # pdb.set_trace()
    layer = data_source.CreateLayer('access_samp', srs, ogr.wkbPolygon)

    # Add FID and ACCESS fields and make the format same to sample data
    fid_field = ogr.FieldDefn('FID', ogr.OFTInteger64)
    fid_field.SetWidth(11)
    fid_field.SetPrecision(0)
    layer.CreateField(fid_field)

    access_field = ogr.FieldDefn('ACCESS', ogr.OFTReal)
    access_field.SetWidth(8)
    access_field.SetPrecision(1)
    layer.CreateField(access_field)

    # Create the feature
    for fid_val, access_val, poly in zip(fid_list, access_list, poly_list):
        # pdb.set_trace()
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('FID', fid_val)
        feature.SetField('ACCESS', access_val)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        feature = None

    data_source = None


def make_lulc_raster(raster_path, lulc_val):
    """Create a 30x10 raster on designated path with designated LULC code.

    Parameters:
        raster_path (str): the raster path for making the LULC raster.
        lulc_val (int): the LULC value to be filled in the raster.

    Returns:
        None.

    """
    lulc_array = numpy.empty((30, 10))
    lulc_array.fill(lulc_val)
    make_simple_raster(lulc_array, raster_path)


def make_threats_raster(raster_folder):
    """Create a 40x10 raster on designated path with 1 as threat and 0 as none.

    Parameters:
        raster_path (str): the raster path for making the threat raster.

    Returns:
        None.

    """
    array = numpy.concatenate((numpy.full((20, 10), 0), numpy.full((10, 10),
                                                                   1)))
    for suffix in ['_c', '_f']:
        raster_path = os.path.join(raster_folder, 'threat_1' + suffix + '.tif')
        make_simple_raster(array, raster_path)


def make_sensitivity_samp_csv(csv_path):
    """Create a simplified sensitivity csv file with five land cover types.

    Parameters:
        csv_path (str): the path of sensitivity csv.

    Returns:
        None.

    """
    with open(csv_path, 'wb') as open_table:
        open_table.write('LULC,NAME,HABITAT,L_threat_1\n')
        open_table.write('0,"lulc 0",1,0.4\n')
        open_table.write('1,"lulc 1",1,0.3\n')
        open_table.write('2,"lulc 2",0,0.0\n')


def make_threats_csv(csv_path):
    """Create a simplified threat csv with two threat types.

    Parameters:
        csv_path (str): the path of threat csv.

    Returns:
        None.

    """
    with open(csv_path, 'wb') as open_table:
        open_table.write('MAX_DIST,WEIGHT,THREAT,DECAY\n')
        open_table.write('5,1,threat_1,linear\n')


class HabitatQualityTests(unittest.TestCase):
    """Tests for the Habitat Quality model."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_habitat_quality_regression(self):
        """Habitat Quality: base regression test."""
        from natcap.invest import habitat_quality

        args = {
            'access_uri':
            os.path.join(SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
            'half_saturation_constant':
            '0.5',
            'landuse_bas_uri':
            os.path.join(SAMPLE_DATA, 'HabitatQuality', 'lc_samp_bse_b.tif'),
            'landuse_cur_uri':
            os.path.join(SAMPLE_DATA, 'HabitatQuality', 'lc_samp_cur_b.tif'),
            'landuse_fut_uri':
            os.path.join(SAMPLE_DATA, 'HabitatQuality', 'lc_samp_fut_b.tif'),
            'sensitivity_uri':
            os.path.join(SAMPLE_DATA, 'HabitatQuality',
                         'sensitivity_samp.csv'),
            'suffix':
            'regression',
            'threat_raster_folder':
            os.path.join(SAMPLE_DATA, 'HabitatQuality'),
            'threats_uri':
            os.path.join(SAMPLE_DATA, 'HabitatQuality', 'threats_samp.csv'),
            u'workspace_dir':
            self.workspace_dir,
        }

        # habitat_quality.execute(args)
        # HabitatQualityTests._test_same_files(
        #     os.path.join(REGRESSION_DATA, 'file_list_regression.txt'),
        #     args['workspace_dir'])

        # for output_filename in [
        #         'rarity_f_regression.tif', 'deg_sum_out_c_regression.tif',
        #         'deg_sum_out_f_regression.tif',
        #         'quality_out_c_regression.tif',
        #         'quality_out_f_regression.tif', 'rarity_c_regression.tif']:
        #     natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
        #         os.path.join(REGRESSION_DATA, output_filename),
        #         os.path.join(self.workspace_dir, 'output', output_filename),
        #         1e-6)

    def test_habitat_quality_regression_fast(self):
        """Habitat Quality: base regression test with simplified data."""
        from natcap.invest import habitat_quality

        args = {
            # 'access_uri':
            # os.path.join(SAMPLE_DATA, 'access_samp.shp'),
            'half_saturation_constant':
            '0.5',
            # 'landuse_bas_uri':
            # os.path.join(SAMPLE_DATA, 'HabitatQuality', 'lc_samp_bse_b.tif'),
            # 'landuse_cur_uri':
            # os.path.join(SAMPLE_DATA, 'HabitatQuality', 'lc_samp_cur_b.tif'),
            # 'landuse_fut_uri':
            # os.path.join(SAMPLE_DATA, 'HabitatQuality', 'lc_samp_fut_b.tif'),
            # 'sensitivity_uri':
            # os.path.join(SAMPLE_DATA, 'HabitatQuality',
            #              'sensitivity_samp.csv'),
            'suffix':
            'regression',
            # 'threat_raster_folder':
            # os.path.join(SAMPLE_DATA, 'HabitatQuality'),
            # 'threats_uri':
            # os.path.join(SAMPLE_DATA, 'HabitatQuality', 'threats_samp.csv'),
            u'workspace_dir':
            self.workspace_dir,
        }

        args['workspace_dir'] = temp_dir

        args['access_uri'] = os.path.join(args['workspace_dir'],
                                          'access_samp.shp')
        make_access_shp(args['access_uri'])

        lulc_names = ['_bas_', '_cur_', '_fut_']
        for lulc_val, lulc_name in enumerate(lulc_names):
            args['landuse' + lulc_name + 'uri'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + lulc_name + 'b.tif')
            make_lulc_raster(args['landuse' + lulc_name + 'uri'], lulc_val)

        args['sensitivity_uri'] = os.path.join(args['workspace_dir'],
                                               'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_uri'])

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        args['threats_uri'] = os.path.join(args['workspace_dir'],
                                           'threats_samp.csv')
        make_threats_csv(args['threats_uri'])
        habitat_quality.execute(args)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # @scm.skip_if_data_missing(REGRESSION_DATA)
    # def test_habitat_quality_missing_sensitivity_threat(self):
    #     """Habitat Quality: ValueError w/ missing threat in sensitivity."""
    #     from natcap.invest import habitat_quality

    #     args = {
    #         'access_uri': os.path.join(
    #             SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
    #         'half_saturation_constant': '0.5',
    #         'landuse_cur_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'sensitivity_uri': os.path.join(
    #             REGRESSION_DATA, 'small_sensitivity_samp.csv'),
    #         'threat_raster_folder': os.path.join(REGRESSION_DATA),
    #         'threats_uri': os.path.join(
    #             REGRESSION_DATA, 'small_threats_samp_missing_threat.csv'),
    #         u'workspace_dir': self.workspace_dir,
    #     }

    #     with self.assertRaises(ValueError):
    #         habitat_quality.execute(args)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # @scm.skip_if_data_missing(REGRESSION_DATA)
    # def test_habitat_quality_missing_threat(self):
    #     """Habitat Quality: expected ValueError on missing threat raster."""
    #     from natcap.invest import habitat_quality

    #     args = {
    #         'access_uri': os.path.join(
    #             SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
    #         'half_saturation_constant': '0.5',
    #         'landuse_cur_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'sensitivity_uri': os.path.join(
    #             REGRESSION_DATA, 'small_sensitivity_samp_missing_threat.csv'),
    #         'threat_raster_folder': os.path.join(REGRESSION_DATA),
    #         'threats_uri': os.path.join(
    #             REGRESSION_DATA, 'small_threats_samp_missing_threat.csv'),
    #         u'workspace_dir': self.workspace_dir,
    #     }

    #     with self.assertRaises(ValueError):
    #         habitat_quality.execute(args)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # @scm.skip_if_data_missing(REGRESSION_DATA)
    # def test_habitat_quality_invalid_decay_type(self):
    #     """Habitat Quality: expected ValueError on invalid decay type."""
    #     from natcap.invest import habitat_quality

    #     args = {
    #         'access_uri': os.path.join(
    #             SAMPLE_DATA, 'HabitatQuality', 'access_samp.shp'),
    #         'half_saturation_constant': '0.5',
    #         'landuse_cur_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'sensitivity_uri': os.path.join(
    #             REGRESSION_DATA, 'small_sensitivity_samp.csv'),
    #         'threat_raster_folder': os.path.join(REGRESSION_DATA),
    #         'threats_uri': os.path.join(
    #             REGRESSION_DATA, 'small_threats_samp_invalid_decay.csv'),
    #         u'workspace_dir': self.workspace_dir,
    #     }

    #     with self.assertRaises(ValueError):
    #         habitat_quality.execute(args)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # @scm.skip_if_data_missing(REGRESSION_DATA)
    # def test_habitat_quality_bad_rasters(self):
    #     """Habitat Quality: on threats that aren't real rasters."""
    #     from natcap.invest import habitat_quality

    #     args = {
    #         'half_saturation_constant': '0.5',
    #         'landuse_cur_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'sensitivity_uri': os.path.join(
    #             REGRESSION_DATA, 'small_sensitivity_samp.csv'),
    #         'threat_raster_folder': os.path.join(
    #             REGRESSION_DATA, 'bad_rasters'),
    #         'threats_uri': os.path.join(
    #             REGRESSION_DATA, 'small_threats_samp.csv'),
    #         u'workspace_dir': self.workspace_dir,
    #     }

    #     with self.assertRaises(ValueError):
    #         habitat_quality.execute(args)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # @scm.skip_if_data_missing(REGRESSION_DATA)
    # def test_habitat_quality_nodata_small(self):
    #     """Habitat Quality: on rasters that have nodata values."""
    #     from natcap.invest import habitat_quality

    #     args = {
    #         'half_saturation_constant': '0.5',
    #         'landuse_cur_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'sensitivity_uri': os.path.join(
    #             REGRESSION_DATA, 'small_sensitivity_samp.csv'),
    #         'threat_raster_folder': os.path.join(REGRESSION_DATA),
    #         'threats_uri': os.path.join(
    #             REGRESSION_DATA, 'small_threats_samp.csv'),
    #         u'workspace_dir': self.workspace_dir,
    #     }

    #     habitat_quality.execute(args)
    #     HabitatQualityTests._test_same_files(
    #         os.path.join(REGRESSION_DATA, 'file_list_small_nodata.txt'),
    #         args['workspace_dir'])

    #     # reasonable to just check quality out in this case
    #     natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
    #         os.path.join(REGRESSION_DATA, 'small_quality_out_c.tif'),
    #         os.path.join(self.workspace_dir, 'output', 'quality_out_c.tif'),
    #         1e-6)

    # @scm.skip_if_data_missing(SAMPLE_DATA)
    # @scm.skip_if_data_missing(REGRESSION_DATA)
    # def test_habitat_quality_nodata_small_fut(self):
    #     """Habitat Quality: small test with future raster only."""
    #     from natcap.invest import habitat_quality

    #     args = {
    #         'half_saturation_constant': '0.5',
    #         'landuse_cur_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'landuse_bas_uri': os.path.join(
    #             REGRESSION_DATA, 'small_lulc_base.tif'),
    #         'sensitivity_uri': os.path.join(
    #             REGRESSION_DATA, 'small_sensitivity_samp.csv'),
    #         'threat_raster_folder': os.path.join(REGRESSION_DATA),
    #         'threats_uri': os.path.join(
    #             REGRESSION_DATA, 'small_threats_samp.csv'),
    #         u'workspace_dir': self.workspace_dir,
    #     }

    #     habitat_quality.execute(args)
    #     HabitatQualityTests._test_same_files(
    #         os.path.join(REGRESSION_DATA, 'file_list_small_nodata_fut.txt'),
    #         args['workspace_dir'])

    #     # reasonable to just check quality out in this case
    #     natcap.invest.pygeoprocessing_0_3_3.testing.assert_rasters_equal(
    #         os.path.join(REGRESSION_DATA, 'small_quality_out_c.tif'),
    #         os.path.join(self.workspace_dir, 'output', 'quality_out_c.tif'),
    #         1e-6)

    # @staticmethod
    # def _test_same_files(base_list_path, directory_path):
    #     """Assert files in `base_list_path` are in `directory_path`.

    #     Parameters:
    #         base_list_path (string): a path to a file that has one relative
    #             file path per line.
    #         directory_path (string): a path to a directory whose contents will
    #             be checked against the files listed in `base_list_file`

    #     Returns:
    #         None

    #     Raises:
    #         AssertionError when there are files listed in `base_list_file`
    #             that don't exist in the directory indicated by `path`

    #     """
    #     missing_files = []
    #     with open(base_list_path, 'r') as file_list:
    #         for file_path in file_list:
    #             full_path = os.path.join(directory_path, file_path.rstrip())
    #             if full_path == '':
    #                 continue
    #             if not os.path.isfile(full_path):
    #                 missing_files.append(full_path)
    #     if len(missing_files) > 0:
    #         raise AssertionError(
    #             "The following files were expected but not found: " +
    #             '\n'.join(missing_files))
