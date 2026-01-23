import os
import shutil
import sys
import tempfile
import unittest

import lxml.html
import shapely
from osgeo import ogr, osr
import pygeoprocessing

from natcap.invest.reports import sdr_ndr_utils


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
    field_types = [ogr.OFTInteger, ogr.OFTString, ogr.OFTInteger, ogr.OFTInteger]
    field_dict = {
        name: dtype for name, dtype in zip(MAIN_TABLE_COLS, field_types)}
    attribute_list = [
        {'ws_id': i + 1,
         'ws_name': ws_names[i % len(ws_names)],
         'calculated_value_1': i + 101,
         'calculated_value_2': i + 201}
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


@unittest.skipIf(sys.platform.startswith("win"), "segfaults on Windows")
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

        root = lxml.html.document_fromstring(main_table)

        # Make sure table body has exactly 1 row.
        table_body_rows = root.xpath('.//table/tbody/tr')
        self.assertEqual(len(table_body_rows), num_features)

        # Make sure table body has expected number of columns.
        ws_1_row = table_body_rows[0]
        ws_1_cells = ws_1_row.xpath('./td')
        self.assertEqual(len(ws_1_cells), 4)

        # Check values.
        ws_1_data = attribute_list[0]
        # xpath positions are 1-indexed.
        for (i, val) in enumerate(ws_1_data.values(), start=1):
            ws_1_cell = ws_1_row.xpath(f'./td[{i}]')
            self.assertEqual(str(val), ws_1_cell[0].text)

        # Make sure table has class 'datatable' but not 'paginate'.
        datatable_table = root.find_class('datatable')
        self.assertEqual(len(datatable_table), 1)
        paginated_table = root.find_class('paginate')
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

        main_table_root = lxml.html.document_fromstring(main_table)

        # Make sure main table body has exactly `num_features` rows.
        table_body_rows = main_table_root.xpath('.//table/tbody/tr')
        self.assertEqual(len(table_body_rows), num_features)

        # Make sure main table body has expected number of columns.
        ws_1_row = table_body_rows[0]
        ws_1_cells = ws_1_row.xpath('./td')
        self.assertEqual(len(ws_1_cells), 4)

        # Check main table column headings for accuracy.
        table_header_rows = main_table_root.xpath('.//table/thead/tr')
        table_header_row = table_header_rows[0]
        col_names = MAIN_TABLE_COLS
        # xpath positions are 1-indexed.
        for (i, col_name) in enumerate(col_names, start=1):
            col_header = table_header_row.xpath(f'./th[{i}]')
            self.assertEqual(col_name, col_header[0].text)

        # Check main table values.
        for (i, ws_data) in enumerate(attribute_list):
            html_table_row = table_body_rows[i]
            # xpath positions are 1-indexed.
            for (j, val) in enumerate(ws_data.values(), start=1):
                table_cell = html_table_row.xpath(f'./td[{j}]')
                self.assertEqual(str(val), table_cell[0].text)

        # Make sure main table has class 'datatable' but not 'paginate'.
        datatable_table = main_table_root.find_class('datatable')
        self.assertEqual(len(datatable_table), 1)
        paginated_table = main_table_root.find_class('paginate')
        self.assertEqual(len(paginated_table), 0)

        # Make sure totals table is not None.
        self.assertIsNotNone(totals_table)

        totals_table_root = lxml.html.document_fromstring(totals_table)

        # Make sure totals table body has exactly 1 row.
        table_body_rows = totals_table_root.xpath('.//table/tbody/tr')
        self.assertEqual(len(table_body_rows), 1)

        # Make sure totals table body has expected number of columns.
        totals_row = table_body_rows[0]
        totals_cells = totals_row.xpath('./td')
        self.assertEqual(len(totals_cells), 2)

        # Check totals table column headings for accuracy.
        table_header_rows = totals_table_root.xpath('.//table/thead/tr')
        table_header_row = table_header_rows[0]
        # xpath positions are 1-indexed. Skip first (empty) th and start at 2.
        for (i, col_name) in enumerate(cols_to_sum, start=2):
            col_header = table_header_row.xpath(f'./th[{i}]')
            self.assertEqual(col_name, col_header[0].text)

        # Check totals table values.
        # calculated_value_1 is ws_id + 100; calculated_value_2 is ws_id + 200.
        totals = [101 + 102, 201 + 202]
        # xpath positions are 1-indexed.
        for (i, val) in enumerate(totals, start=1):
            totals_cell = totals_row.xpath(f'./td[{i}]')
            self.assertEqual(str(val), totals_cell[0].text)

    def test_generate_results_table_with_pagination_directive(self):
        """Return table with paginate flag when there are a lot of features."""

        num_features = 11
        self.assertGreater(num_features,
                           sdr_ndr_utils.TABLE_PAGINATION_THRESHOLD)

        filepath = os.path.join(self.workspace_dir, 'vector.gpkg')
        attribute_list = _generate_mock_watershed_data(num_features, filepath)
        cols_to_sum = ['calculated_value_1', 'calculated_value_2']

        (main_table, totals_table) = (
            sdr_ndr_utils.generate_results_table_from_vector(
                filepath, cols_to_sum))
        self.assertIsNotNone(main_table)

        main_table_root = lxml.html.document_fromstring(main_table)

        # Make sure main table body has exactly `num_features` rows.
        table_body_rows = main_table_root.xpath('.//table/tbody/tr')
        self.assertEqual(len(table_body_rows), num_features)

        # Check main table values.
        for (i, ws_data) in enumerate(attribute_list):
            html_table_row = table_body_rows[i]
            # xpath positions are 1-indexed.
            for (j, val) in enumerate(ws_data.values(), start=1):
                table_cell = html_table_row.xpath(f'./td[{j}]')
                self.assertEqual(str(val), table_cell[0].text)

        # Make sure main table has classes 'datatable' AND 'paginate'.
        datatable_table = main_table_root.find_class('datatable')
        self.assertEqual(len(datatable_table), 1)
        paginated_table = main_table_root.find_class('paginate')
        self.assertEqual(len(paginated_table), 1)

        # Make sure totals table is not None.
        self.assertIsNotNone(totals_table)

        # Check totals table values.
        totals_table_root = lxml.html.document_fromstring(totals_table)
        table_body_rows = totals_table_root.xpath('.//table/tbody/tr')
        totals_row = table_body_rows[0]
        # calculated_value_1 is ws_id + 100; calculated_value_2 is ws_id + 200.
        totals = [
            101 + 102 + 103 + 104 + 105 + 106 + 107 + 108 + 109 + 110 + 111,
            201 + 202 + 203 + 204 + 205 + 206 + 207 + 208 + 209 + 210 + 211
        ]
        # xpath positions are 1-indexed.
        for (i, val) in enumerate(totals, start=1):
            totals_cell = totals_row.xpath(f'./td[{i}]')
            self.assertEqual(str(val), totals_cell[0].text)
