import unittest
import gdal
import os
import tempfile


from natcap.invest import routedem


SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-sample-data',
    'DelineateIt')
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data',
    'delineateit')


class RouteDEMCoreTests(unittest.TestCase):

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_identify_pour_points(self):
        input_path = '/Users/emily/invest/test_flow_direction.tif'
        output_path = os.path.join(self.workspace_dir, 'point_vector.gpkg')

        routedem.routedem_core.identify_pour_points(input_path, output_path)

        vector = gdal.OpenEx(output_path, gdal.OF_VECTOR)
        layer = vector.GetLayer(1)

        points = set()
        for feature in layer:
            geom = feature.GetGeometryRef()
            point = geom.GetPoint()
            points.add(point)

        expected_points = {(1917, 1), (1917, 0)}
        self.assertEqual(points, expected_points)


