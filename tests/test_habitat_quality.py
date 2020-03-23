"""Module for Regression Testing the InVEST Habitat Quality model."""
import unittest
import tempfile
import shutil
import os

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy
import pygeoprocessing.testing


def make_simple_poly(origin):
    """Make a 50x100 ogr rectangular geometry clockwisely from origin.

    Parameters:
        origin (tuple): the longitude and latitude of the origin of the
                        rectangle.

    Returns:
        None.

    """
    # Create a rectangular ring
    lon, lat = origin[0], origin[1]
    width = 100
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(lon, lat)
    ring.AddPoint(lon + width, lat)
    ring.AddPoint(lon + width, lat - width / 2.0)
    ring.AddPoint(lon, lat - width / 2.0)
    ring.AddPoint(lon, lat)

    # Create polygon geometry
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def make_raster_from_array(base_array, base_raster_path):
    """Make a raster from an array on a designated path.

    Parameters:
        array (numpy.ndarray): the 2D array for making the raster.
        raster_path (str): the path for the raster to be created.

    Returns:
        None.

    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # UTM Zone 10N
    project_wkt = srs.ExportToWkt()

    pygeoprocessing.testing.create_raster_on_disk(
        [base_array],
        (1180000, 690000),
        project_wkt,
        -1,
        (1, -1),  # Each pixel is 1x1 m
        filename=base_raster_path)


def make_access_shp(access_shp_path):
    """Create a 100x100 accessibility polygon shapefile with two access values.

    Parameters:
        access_shp_path (str): the path for the shapefile.

    Returns:
        None.

    """
    # Set up parameters. Fid and access values are based on the sample data
    fid_list = [0.0, 1.0]
    access_list = [0.2, 1.0]
    coord_list = [(1180000.0, 690000.0 - i * 50) for i in range(2)]
    poly_list = [make_simple_poly(coord) for coord in coord_list]

    # Create a new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.CreateDataSource(access_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # Spatial reference UTM Zone 10N
    layer = data_source.CreateLayer('access_samp', srs, ogr.wkbPolygon)

    # Add FID and ACCESS fields and make their format same to sample data
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
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('FID', fid_val)
        feature.SetField('ACCESS', access_val)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        feature = None
    layer.SyncToDisk()
    data_source.SyncToDisk()
    data_source = None


def make_lulc_raster(raster_path, lulc_val, side_length=100):
    """Create a 100x100 raster on raster path with designated LULC code.

    Parameters:
        raster_path (str): the path for the LULC raster.
        lulc_val (int): the LULC value to be filled in the raster.

    Returns:
        None.

    """
    lulc_array = numpy.ones((side_length, side_length), dtype=numpy.int8)
    lulc_array[50:, :] = lulc_val
    make_raster_from_array(lulc_array, raster_path)


def make_threats_raster(folder_path, make_empty_raster=False, side_length=100,
                        threat_values=None):
    """Create a side_lengthXside_length raster on designated path with 1 as
    threat and 0 as none.

    Parameters:
        folder_path (str): the folder path for saving the threat rasters.
        make_empty_raster=False (bool): Whether to write a raster file
            that has no values at all.
        side_length=100 (int): The length of the sides of the threat raster.
        threat_values=None (None or list): If None, threat values of 1 will be
            used for the two threat rasters created.  Otherwise, a 2-element
            list should include numeric threat values for the two threat
            rasters.

    Returns:
        None.

    """
    threat_names = ['threat_1', 'threat_2']
    if not threat_values:
        threat_values = [1, 1]  # Nonzero should mean threat presence.

    threat_array = numpy.zeros((side_length, side_length), dtype=numpy.int8)

    for suffix in ['_c', '_f']:
        for (i, threat), value in zip(enumerate(threat_names), threat_values):
            raster_path = os.path.join(folder_path, threat + suffix + '.tif')
            threat_array[100//(i+1):, :] = value  # making variations among threats
            if make_empty_raster:
                open(raster_path, 'a').close()  # writes an empty raster.
            else:
                make_raster_from_array(threat_array, raster_path)


def make_sensitivity_samp_csv(csv_path,
                              include_threat=True, missing_lines=False):
    """Create a simplified sensitivity csv file with five land cover types.

    Parameters:
        csv_path (str): the path of sensitivity csv.
        include_threat (bool): whether the "threat" column is included in csv.

    Returns:
        None.

    """
    if include_threat:
        with open(csv_path, 'w') as open_table:
            open_table.write('LULC,NAME,HABITAT,L_threat_1,L_threat_2\n')
            open_table.write('1,"lulc 1",1,1,1\n')
            if not missing_lines:
                open_table.write('2,"lulc 2",0.5,0.5,1\n')
                open_table.write('3,"lulc 3",0,0.3,1\n')
    else:
        with open(csv_path, 'w') as open_table:
            open_table.write('LULC,NAME,HABITAT\n')
            open_table.write('1,"lulc 1",1\n')
            if not missing_lines:
                open_table.write('2,"lulc 2",0.5\n')
                open_table.write('3,"lulc 3",0\n')


def make_threats_csv(csv_path,
                     include_missing_threat=False,
                     include_invalid_decay=False):
    """Create a simplified threat csv with two threat types.

    Parameters:
        csv_path (str): the path of threat csv.
        include_missing_threat (bool): whether an extra threat is included in
                                       the csv.
        include_invalid_decay (bool): whether an invalid decay function name
                                      is set for "threat 1".

    Returns:
        None.

    """
    with open(csv_path, 'w') as open_table:
        open_table.write('MAX_DIST,WEIGHT,THREAT,DECAY\n')
        if include_invalid_decay:
            open_table.write('0.9,0.7,threat_1,invalid\n')
        else:
            open_table.write('0.9,0.7,threat_1,linear\n')
            open_table.write('0.5,1.0,threat_2,exponential\n')
        if include_missing_threat:
            open_table.write('0.5,0.8,missing_threat,linear\n')


def assert_array_sum(base_raster_path, desired_sum, include_nodata=True):
    """Assert that the sum of a raster is equal to the specified value.

    Parameters:
        base_raster_path (str): the filepath of the raster to be asserted.
        desired_sum (float): the value to be compared with the raster sum.

    Returns:
        None.

    """
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    base_band = base_raster.GetRasterBand(1)
    base_array = base_band.ReadAsArray()
    nodata = base_band.GetNoDataValue()
    if not include_nodata:
        base_array = base_array[~numpy.isclose(base_array, nodata)]

    raster_sum = numpy.sum(base_array)
    numpy.testing.assert_almost_equal(raster_sum, desired_sum, decimal=3)


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

    def test_habitat_quality_regression(self):
        """Habitat Quality: base regression test with simplified data."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
        }

        args['access_vector_path'] = os.path.join(args['workspace_dir'],
                                                  'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_lulc_raster(args['lulc' + scenario + 'path'], lulc_val)

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'],
                            threat_values=[0.5, 6.6])

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 1792.8088,
                'deg_sum_f_regression.tif': 2308.9636,
                'quality_c_regression.tif': 6928.5293,
                'quality_f_regression.tif': 4916.338,
                'rarity_c_regression.tif': 2500.0000000,
                'rarity_f_regression.tif': 2500.0000000}.items():
            raster_path = os.path.join(args['workspace_dir'], 'output',
                                       output_filename)
            # Check that the raster's computed values are what we expect.
            # In this case, the LULC and threat rasters should have been
            # expanded to be beyond the bounds of the original threat values,
            # so we should exclude those new nodata pixel values.
            assert_array_sum(raster_path, assert_value, include_nodata=False)

    def test_habitat_quality_lulc_bbox(self):
        """Habitat Quality: regression test for bbox sizes."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
        }

        args['access_vector_path'] = os.path.join(args['workspace_dir'],
                                                  'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_lulc_raster(args['lulc' + scenario + 'path'], lulc_val)

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'], side_length=50,
                            threat_values=[1, 1])

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        habitat_quality.execute(args)

        base_lulc_bbox = pygeoprocessing.get_raster_info(
            args['lulc_bas_path'])['bounding_box']

        # Assert values were obtained by summing each output raster.
        for output_filename in ['deg_sum_c_regression.tif',
                                'deg_sum_f_regression.tif',
                                'quality_c_regression.tif',
                                'quality_f_regression.tif',
                                'rarity_c_regression.tif',
                                'rarity_f_regression.tif']:
            raster_path = os.path.join(args['workspace_dir'], 'output',
                                       output_filename)

            # Check that the output raster has the same bounding box as the
            # LULC rasters.
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            raster_bbox = raster_info['bounding_box']
            numpy.testing.assert_array_almost_equal(
                raster_bbox, base_lulc_bbox)

    def test_habitat_quality_numeric_threats(self):
        """Habitat Quality: regression test on numeric threat names."""
        from natcap.invest import habitat_quality
        threat_array = numpy.zeros((100, 100), dtype=numpy.int8)

        threatnames = ['1111', '2222']
        threats_folder = os.path.join(self.workspace_dir, 'threats')
        os.makedirs(threats_folder)
        for suffix in ['_c', '_f']:
            for i, threat in enumerate(threatnames):
                raster_path = os.path.join(threats_folder,
                                           threat + suffix + '.tif')
                threat_array[100//(i+1):, :] = 1  # making variations among threats
                make_raster_from_array(threat_array, raster_path)

        threat_csv_path = os.path.join(self.workspace_dir, 'threats.csv')
        with open(threat_csv_path, 'w') as open_table:
            open_table.write('MAX_DIST,WEIGHT,THREAT,DECAY\n')
            open_table.write('0.9,0.7,%s,linear\n' % threatnames[0])
            open_table.write('0.5,1.0,%s,exponential\n' % threatnames[1])

        args = {
            'half_saturation_constant': '0.5',
            'results_suffix': 'regression',
            'threats_table_path': threat_csv_path,
            'workspace_dir': os.path.join(self.workspace_dir, 'workspace'),
            'threat_raster_folder': threats_folder,
            'sensitivity_table_path': os.path.join(self.workspace_dir,
                                                   'sensitivity_samp.csv'),
            'access_vector_path': os.path.join(self.workspace_dir,
                                               'access_samp.shp')
        }
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            args['lulc' + scenario + 'path'] = os.path.join(
                self.workspace_dir, 'lc_samp' + scenario + 'b.tif')
            make_lulc_raster(args['lulc' + scenario + 'path'], lulc_val)

        with open(args['sensitivity_table_path'], 'w') as open_table:
            open_table.write('LULC,NAME,HABITAT,L_%s,L_%s\n' % tuple(threatnames))
            open_table.write('1,"lulc 1",1,1,1\n')
            open_table.write('2,"lulc 2",0.5,0.5,1\n')
            open_table.write('3,"lulc 3",0,0.3,1\n')

        habitat_quality.execute(args)

        # Assert values were obtained by summing each output raster.
        for output_filename, assert_value in {
                'deg_sum_c_regression.tif': 1792.8088,
                'deg_sum_f_regression.tif': 2308.9636,
                'quality_c_regression.tif': 6928.5293,
                'quality_f_regression.tif': 4916.338,
                'rarity_c_regression.tif': 2500.0000000,
                'rarity_f_regression.tif': 2500.0000000
        }.items():
            assert_array_sum(
                os.path.join(args['workspace_dir'], 'output', output_filename),
                assert_value)

    def test_habitat_quality_missing_sensitivity_threat(self):
        """Habitat Quality: ValueError w/ missing threat in sensitivity."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['access_vector_path'] = os.path.join(args['workspace_dir'],
                                                  'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        # Include a missing threat to the sensitivity csv table
        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(
            args['sensitivity_table_path'], include_threat=False)

        args['lulc_cur_path'] = os.path.join(args['workspace_dir'],
                                             'lc_samp_cur_b.tif')
        make_lulc_raster(args['lulc_cur_path'], 2)

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_missing_threat(self):
        """Habitat Quality: expected ValueError on missing threat raster."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['access_vector_path'] = os.path.join(args['workspace_dir'],
                                                  'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(args['workspace_dir'],
                                             'lc_samp_cur_b.tif')
        make_lulc_raster(args['lulc_cur_path'], 2)

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        # Include a missing threat to the threats csv table
        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(
            args['threats_table_path'], include_missing_threat=True)

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_invalid_decay_type(self):
        """Habitat Quality: expected ValueError on invalid decay type."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['access_vector_path'] = os.path.join(args['workspace_dir'],
                                                  'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(args['workspace_dir'],
                                             'lc_samp_cur_b.tif')
        make_lulc_raster(args['lulc_cur_path'], 2)

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        # Include an invalid decay function name to the threats csv table.
        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(
            args['threats_table_path'], include_invalid_decay=True)

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_bad_rasters(self):
        """Habitat Quality: on threats that aren't real rasters."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(args['workspace_dir'],
                                             'lc_samp_cur_b.tif')
        make_lulc_raster(args['lulc_cur_path'], 2)

        # Make an empty threat raster in the workspace folder.
        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'],
                            make_empty_raster=True)

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        with self.assertRaises(ValueError):
            habitat_quality.execute(args)

    def test_habitat_quality_nodata(self):
        """Habitat Quality: on missing base and future LULC rasters."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        args['lulc_cur_path'] = os.path.join(args['workspace_dir'],
                                             'lc_samp_cur_b.tif')
        make_lulc_raster(args['lulc_cur_path'], 2)

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        habitat_quality.execute(args)

        # Reasonable to just check quality out in this case
        assert_array_sum(
            os.path.join(args['workspace_dir'], 'output', 'quality_c.tif'),
            5951.2827)

    def test_habitat_quality_nodata_fut(self):
        """Habitat Quality: on missing future LULC raster."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'])

        scenarios = ['_bas_', '_cur_']  # Missing '_fut_'
        for lulc_val, scenario in enumerate(scenarios, start=1):
            args['lulc' + scenario + 'path'] = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            make_lulc_raster(args['lulc' + scenario + 'path'], lulc_val)

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        habitat_quality.execute(args)

        # Reasonable to just check quality out in this case
        assert_array_sum(
            os.path.join(args['workspace_dir'], 'output', 'quality_c.tif'),
            5951.2827)

    def test_habitat_quality_missing_lucodes_in_table(self):
        """Habitat Quality: on missing lucodes in the sensitivity table."""
        from natcap.invest import habitat_quality

        args = {
            'half_saturation_constant': '0.5',
            'workspace_dir': self.workspace_dir,
        }

        args['access_vector_path'] = os.path.join(args['workspace_dir'],
                                                  'access_samp.shp')
        make_access_shp(args['access_vector_path'])

        scenarios = ['_bas_', '_cur_', '_fut_']
        for lulc_val, scenario in enumerate(scenarios, start=1):
            path = os.path.join(
                args['workspace_dir'], 'lc_samp' + scenario + 'b.tif')
            args['lulc' + scenario + 'path'] = path
            make_lulc_raster(path, lulc_val)

            # Add a nodata value to this raster to make sure we don't include
            # the nodata value in the error message.
            raster = gdal.OpenEx(path, gdal.OF_RASTER | gdal.GA_Update)
            band = raster.GetRasterBand(1)
            band_nodata = 255
            band.SetNoDataValue(band_nodata)  # band nodata before this is -1
            current_array = band.ReadAsArray()
            current_array[49][49] = band_nodata
            band.WriteArray(current_array)
            band = None
            raster = None

        args['sensitivity_table_path'] = os.path.join(args['workspace_dir'],
                                                      'sensitivity_samp.csv')
        make_sensitivity_samp_csv(args['sensitivity_table_path'],
                                  missing_lines=True)

        args['threat_raster_folder'] = args['workspace_dir']
        make_threats_raster(args['threat_raster_folder'])

        args['threats_table_path'] = os.path.join(args['workspace_dir'],
                                                  'threats_samp.csv')
        make_threats_csv(args['threats_table_path'])

        with self.assertRaises(ValueError) as cm:
            habitat_quality.execute(args)

        actual_message = str(cm.exception)
        self.assertTrue(
            'The following land cover codes were found in ' in
            actual_message, actual_message)
        # 2, 3 are the missing landcover codes.
        # Raster nodata is 255 and should NOT appear in this list.
        self.assertTrue(': 2, 3.' in actual_message, actual_message)

    def test_habitat_quality_validate(self):
        """Habitat Quality: validate raise exception as expected."""
        from natcap.invest import habitat_quality

        args = {
            'results_suffix': 'regression',
            'workspace_dir': self.workspace_dir,
            'threat_raster_folder': self.workspace_dir,
        }

        validation_results = habitat_quality.validate(args)
        self.assertEqual(len(validation_results), 1)

        keys_without_value = set([
            'lulc_cur_path', 'threats_table_path', 'sensitivity_table_path',
            'half_saturation_constant'])
        self.assertEqual(set(validation_results[0][0]), keys_without_value)
