import os
import shutil
import tempfile
import unittest

import shapely
import pygeoprocessing
from bs4 import BeautifulSoup
from osgeo import ogr, osr

from natcap.invest.reports import sdr_ndr_utils

BSOUP_HTML_PARSER = 'html.parser'


MAIN_TABLE_COLS = ['ws_id', 'ws_name',
                   'calculated_value_1', 'calculated_value_2']


def _generate_mock_watershed_data(num_features, target_vector_path):
    ws_names = ['Willamette', 'Columbia', 'Snake', 'Salmon', 'Boise',
                'Owyhee', 'Deschutes', 'Sacramento', 'American', 'Tuolomne',
                'San Joaquin', 'Los Angeles', 'Santa Ana', 'Colorado', 'Green',
                'Wind', 'Yellowstone', 'Missouri', 'Mississippi', 'Minnesota',
                'Rock', 'Chicago', 'Detroit', 'Ohio', 'Cuyahoga',
                'Allegheny', 'Youghiogheny', 'Genesee', 'Susquehanna', 'Hudson',
                'Connecticut', 'Potomac', 'Patapsco', 'Patuxent', 'Shenandoah',
                'Tennessee', 'Red', 'Arkansas', 'Platte']
    # All polygons can be identical: their specifics don't matter to the tests.
    polygon = shapely.Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0)))
    field_types = [ogr.OFTInteger, ogr.OFTString, ogr.OFTReal, ogr.OFTReal]
    field_dict = {
        name: dtype for name, dtype in zip(MAIN_TABLE_COLS, field_types)}
    attribute_list = [
        {'ws_id': i + 1,
         'ws_name': ws_names[i % len(ws_names)],
         'calculated_value_1': i + 101.0,
         'calculated_value_2': i + 201.0}
        for i in range(num_features)
    ]
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(4326)
    pygeoprocessing.shapely_geometry_to_vector(
        shapely_geometry_list=[polygon] * num_features,
        target_vector_path=target_vector_path,
        projection_wkt=projection.ExportToWkt(),
        vector_format='GPKG',
        fields=field_dict,
        attribute_list=attribute_list,
        ogr_geom_type=ogr.wkbPolygon)
    return attribute_list


class SDRNDRUtilsTests(unittest.TestCase):
    """Unit tests for SDR/NDR utils."""

    def setUp(self):
        """Initialize SDRNDRUtilsTests tests."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_generate_results_table_single_feature(self):
        """Return a single-row HTML table; do not return totals."""

        num_features = 1

        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        attribute_list = _generate_mock_watershed_data(num_features, filepath)
        cols_to_sum = []

        (main_table, totals_table) = (
            sdr_ndr_utils.generate_results_table_from_vector(
                filepath, cols_to_sum))
        self.assertIsNotNone(main_table)

        soup = BeautifulSoup(main_table, BSOUP_HTML_PARSER)

        # Make sure table body has exactly 1 row per feature.
        table_body_rows = soup.find('tbody').find_all('tr')
        self.assertEqual(len(table_body_rows), num_features)

        # Make sure table body has expected number of columns.
        ws_1_row = table_body_rows[0]
        ws_1_cells = ws_1_row.find_all('td')
        self.assertEqual(len(ws_1_cells), 4)

        # Check values.
        ws_1_data = attribute_list[0]
        actual_values = [cell.string for cell in ws_1_cells]
        expected_values = [str(v) for v in ws_1_data.values()]
        self.assertEqual(expected_values, actual_values)
        
        # Make sure table has class 'datatable' but not 'paginate'.
        datatable_table = soup.find_all(class_='datatable')
        self.assertEqual(len(datatable_table), 1)
        paginated_table = soup.find_all(class_='paginate')
        self.assertEqual(len(paginated_table), 0)

        # Make sure totals_table is None.
        self.assertIsNone(totals_table)

    def test_generate_results_table_and_totals(self):
        """Return main table and totals table when vector has > 1 feature."""

        num_features = 2

        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        attribute_list = _generate_mock_watershed_data(
            num_features, filepath)
        cols_to_sum = ['calculated_value_1', 'calculated_value_2']

        (main_table, totals_table) = (
            sdr_ndr_utils.generate_results_table_from_vector(
                filepath, cols_to_sum))
        self.assertIsNotNone(main_table)

        main_soup = BeautifulSoup(main_table, BSOUP_HTML_PARSER)

        # Make sure main table body has exactly `num_features` rows.
        table_body_rows = main_soup.find('tbody').find_all('tr')
        self.assertEqual(len(table_body_rows), num_features)

        # Check main table column headings for accuracy.
        table_header_row = main_soup.find('thead').find('tr')
        header_cells = table_header_row.find_all('th')
        col_names = MAIN_TABLE_COLS
        actual_header_values = [cell.string for cell in header_cells]
        self.assertEqual(col_names, actual_header_values)

        # Check main table values.
        for i, feature in enumerate(attribute_list):
            cells = table_body_rows[i].find_all('td')
            actual_values = [cell.string for cell in cells]
            expected_values = [str(v) for v in feature.values()]
            self.assertEqual(expected_values, actual_values)

        # Make sure main table has class 'datatable' but not 'paginate'.
        datatable_table = main_soup.find_all(class_='datatable')
        self.assertEqual(len(datatable_table), 1)
        paginated_table = main_soup.find_all(class_='paginate')
        self.assertEqual(len(paginated_table), 0)

        totals_soup = BeautifulSoup(totals_table, BSOUP_HTML_PARSER)

        # Make sure totals table body has exactly 1 row.
        totals_body_rows = totals_soup.find('tbody').find_all('tr')
        self.assertEqual(len(totals_body_rows), 1)

        # Make sure totals table body has expected number of columns.
        totals_row = totals_body_rows[0]
        totals_cells = totals_row.find_all('td')
        self.assertEqual(len(totals_cells), 2)

        # Check totals table column headings for accuracy.
        totals_header_row = totals_soup.find('thead').find('tr')
        totals_header_cells = totals_header_row.find_all('th')
        actual_header_values = [cell.string for cell in totals_header_cells]
        # The first column is a row index, it did not get summed.
        self.assertEqual([None] + cols_to_sum, actual_header_values)

        # Check totals table values.
        # calculated_value_1 is ws_id + 100; calculated_value_2 is ws_id + 200.
        expected_totals = [101.0 + 102.0, 201.0 + 202.0]
        actual_values = [cell.string for cell in totals_cells]
        expected_values = [str(v) for v in expected_totals]
        self.assertEqual(expected_values, actual_values)

    def test_generate_results_table_with_pagination_directive(self):
        """Return table with paginate flag when there are a lot of features."""

        num_features = 11
        self.assertGreater(num_features,
                           sdr_ndr_utils.TABLE_PAGINATION_THRESHOLD)

        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        _ = _generate_mock_watershed_data(num_features, filepath)
        cols_to_sum = ['calculated_value_1', 'calculated_value_2']

        (main_table, totals_table) = (
            sdr_ndr_utils.generate_results_table_from_vector(
                filepath, cols_to_sum))
        self.assertIsNotNone(main_table)

        main_soup = BeautifulSoup(main_table, BSOUP_HTML_PARSER)

        # Make sure main table has classes 'datatable' AND 'paginate'.
        datatable_table = main_soup.find(class_='datatable')
        self.assertIsNotNone(datatable_table)
        paginated_table = main_soup.find(class_='paginate')
        self.assertIsNotNone(paginated_table)

        # Make sure totals table is not None.
        self.assertIsNotNone(totals_table)
