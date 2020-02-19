"""Module for Testing the InVEST Reporting module."""
import unittest
import tempfile
import shutil
import os
import codecs
import itertools

import pygeoprocessing.testing
from pygeoprocessing.testing import scm

REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'reporting')


class ReportingRegressionTests(unittest.TestCase):
    """Regression Tests for the Reporting Module."""

    def setUp(self):
        """Overriding setUp function to create temp workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def generate_base_args():
        """Generate an args list that is consistent across regression tests."""
        args = {
            'title': 'Test InVEST Reporting',
            'sortable': True,
            'totals': True,
            'elements': [
                {'type': 'table', 'section': 'body', 'sortable': True,
                 'checkbox': True, 'checkbox_pos': 1,
                 'columns': [
                    {'name': 'ws_id', 'total': False,
                     'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'num_pixels', 'total': True,
                     'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'wyield_vol', 'total': True,
                     'attr': {'class': 'my_class'}, 'td_class': 'my_class'}],
                 'total': True},
                {'type': 'head', 'section': 'head',
                 'attributes': {'class': 'my_class'}},
                {'type': 'text', 'section': 'body',
                 'text': '<p>Hello World This is InVEST Reporting.</p>'}]
        }
        return args

    @unittest.skip("skipping instead of updating regression data")
    def test_generate_report_csv_style(self):
        """Reporting: testing full report w/ csv table data and css file."""
        from natcap.invest import reporting

        workspace_dir = self.workspace_dir
        args = ReportingRegressionTests.generate_base_args()

        csv_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_csv.csv')
        style_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_style.css')

        args['out_uri'] = os.path.join(workspace_dir, 'report_csv_style.html')
        args['elements'][0]['attributes'] = {'class': 'my_class'}
        args['elements'][0]['data_type'] = 'csv'
        args['elements'][0]['data'] = csv_path
        args['elements'][0]['key'] = 'ws_id'
        args['elements'][1]['format'] = 'style'
        args['elements'][1]['data_src'] = style_path
        args['elements'][1]['input_type'] = 'File'

        reporting.generate_report(args)

        pygeoprocessing.testing.assert_text_equal(
            args['out_uri'],
            os.path.join(
                REGRESSION_DATA, 'html_reports', 'report_csv_style.html'))

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_dict_script(self):
        """Reporting: test full report w/ dict table data and script file."""
        from natcap.invest import reporting

        workspace_dir = self.workspace_dir
        args = ReportingRegressionTests.generate_base_args()

        script_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_script.js')

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        args['out_uri'] = os.path.join(
            workspace_dir, 'report_dict_script.html')
        args['elements'][0]['attributes'] = {'id': 'my_id'}
        args['elements'][0]['data_type'] = 'dictionary'
        args['elements'][0]['data'] = dict_list
        args['elements'][1]['format'] = 'script'
        args['elements'][1]['data_src'] = script_path
        args['elements'][1]['input_type'] = 'File'

        reporting.generate_report(args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'html_reports', 'report_dict_script.html')
        for source_line, regression_line in itertools.zip_longest(
                open(args['out_uri']).readlines(),
                open(regression_path).readlines(), fillvalue=None):
            if source_line is None or regression_line is None:
                raise AssertionError('Number of lines are unequal.')

            # Strip trailing newlines.
            self.assertEqual(source_line.rstrip(), regression_line.rstrip())

    @unittest.skip("skipping due to different number truncation in py36 and py27")
    def test_generate_report_shape_json(self):
        """Reporting: testing full report w/ shape table data and json file."""
        from natcap.invest import reporting

        workspace_dir = self.workspace_dir
        workspace_dir = 'C:/Users/dmf/projects/invest_dev/py36_compatibility/reporting'
        args = ReportingRegressionTests.generate_base_args()

        shape_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_shape.shp')
        json_data = "{'key': 0, 'data': {'door' : 1, 'room': 'kitchen'}}"

        args['out_uri'] = os.path.join(
            workspace_dir, 'report_shape_json.html')
        args['elements'][0]['data_type'] = 'shapefile'
        args['elements'][0]['data'] = shape_path
        args['elements'][0]['key'] = 'ws_id'
        args['elements'][1]['format'] = 'json'
        args['elements'][1]['data_src'] = json_data
        args['elements'][1]['input_type'] = 'Text'

        reporting.generate_report(args)
        print(args['out_uri'])
        pygeoprocessing.testing.assert_text_equal(
            args['out_uri'],
            os.path.join(
                REGRESSION_DATA, 'html_reports', 'report_shape_json.html'))

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_tags_error(self):
        """Reporting: testing module raises excpetion on included tags."""
        from natcap.invest import reporting

        workspace_dir = self.workspace_dir
        args = ReportingRegressionTests.generate_base_args()

        shape_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_shape.shp')
        json_data = (
            "<script> {'key': 0, 'data': {'door' : 1,"
            " 'room': 'kitchen'}} </script>")

        args['out_uri'] = os.path.join(workspace_dir, 'report_tags_error.html')
        args['elements'][0]['data_type'] = 'shapefile'
        args['elements'][0]['data'] = shape_path
        args['elements'][0]['key'] = 'ws_id'
        args['elements'][1]['format'] = 'json'
        args['elements'][1]['data_src'] = json_data
        args['elements'][1]['input_type'] = 'Text'

        self.assertRaises(Exception, reporting.generate_report, args)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_head_error(self):
        """Reporting: test report raises exception on unknown head type."""
        from natcap.invest import reporting

        workspace_dir = self.workspace_dir
        args = ReportingRegressionTests.generate_base_args()

        shape_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_shape.shp')
        json_data = "{'key': 0, 'data': {'door' : 1, 'room': 'kitchen'}}"

        args['out_uri'] = os.path.join(workspace_dir, 'report_head_error.html')
        args['elements'][0]['data_type'] = 'shapefile'
        args['elements'][0]['data'] = shape_path
        args['elements'][0]['key'] = 'ws_id'
        args['elements'][1]['format'] = 'svg'
        args['elements'][1]['data_src'] = json_data
        args['elements'][1]['input_type'] = 'Text'

        self.assertRaises(Exception, reporting.generate_report, args)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_remove_output(self):
        """Reporting: test full report removes html output if exists."""
        from natcap.invest import reporting

        workspace_dir = self.workspace_dir
        args = ReportingRegressionTests.generate_base_args()

        script_path = os.path.join(
            REGRESSION_DATA, 'sample_input', 'sample_script.js')

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        args['out_uri'] = os.path.join(
            workspace_dir, 'report_test_remove.html')
        args['elements'][0]['attributes'] = {'id': 'my_id'}
        args['elements'][0]['data_type'] = 'dictionary'
        args['elements'][0]['data'] = dict_list
        args['elements'][1]['format'] = 'script'
        args['elements'][1]['data_src'] = script_path
        args['elements'][1]['input_type'] = 'File'

        reporting.generate_report(args)
        # Run again to make sure output file that was created is removed
        reporting.generate_report(args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'html_reports', 'report_dict_script.html')
        for source_line, regression_line in itertools.zip_longest(
                open(args['out_uri']).readlines(),
                open(regression_path).readlines(), fillvalue=None):
            if source_line is None or regression_line is None:
                raise AssertionError('Number of lines are unequal.')

            # Strip trailing newlines.
            self.assertEqual(source_line.rstrip(), regression_line.rstrip())

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_table_generator_attributes(self):
        """Reporting: testing table generator with table attributes."""
        from natcap.invest.reporting import table_generator

        cols = [
            {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'},
             'td_class': 'my_class'},
            {'name': 'num_pixels', 'total': True,
             'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
            {'name': 'wyield_vol', 'total': True,
             'attr': {'class': 'my_class'}, 'td_class': 'my_class'}]

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        table_args = {
            'cols': cols,
            'rows': dict_list,
            'checkbox': True,
            'checkbox_pos': 1,
            'total': True,
            'attributes': {'class': 'sorttable', 'id': 'parcel_table'}
        }

        result_str = table_generator.generate_table(table_args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'table_strings', 'table_string_attrs.txt')
        regression_file = codecs.open(regression_path, 'rU', 'utf-8')
        regression_str = regression_file.read()

        self.assertEqual(result_str, regression_str)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_table_generator_no_attributes(self):
        """Reporting: testing table generator without table attributes."""
        from natcap.invest.reporting import table_generator

        cols = [
            {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'},
             'td_class': 'my_class'},
            {'name': 'num_pixels', 'total': True,
             'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
            {'name': 'wyield_vol', 'total': True,
             'attr': {'class': 'my_class'}, 'td_class': 'my_class'}]

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        table_args = {
            'cols': cols,
            'rows': dict_list,
            'checkbox': True,
            'checkbox_pos': 1,
            'total': True
        }

        result_str = table_generator.generate_table(table_args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'table_strings', 'table_string_no_attrs.txt')
        regression_file = codecs.open(regression_path, 'rU', 'utf-8')
        regression_str = regression_file.read()

        self.assertEqual(result_str, regression_str)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_table_generator_no_checkbox(self):
        """Reporting: testing table generator without checkbox column."""
        from natcap.invest.reporting import table_generator

        cols = [
            {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'},
             'td_class': 'my_class'},
            {'name': 'num_pixels', 'total': True,
             'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
            {'name': 'wyield_vol', 'total': True,
             'attr': {'class': 'my_class'}, 'td_class': 'my_class'}]

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        table_args = {
            'cols': cols,
            'rows': dict_list,
            'total': True
        }

        result_str = table_generator.generate_table(table_args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'table_strings', 'table_string_no_checkbox.txt')
        regression_file = codecs.open(regression_path, 'rU', 'utf-8')
        regression_str = regression_file.read()

        self.assertEqual(result_str, regression_str)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_table_generator_no_td_classes(self):
        """Reporting: testing table generator without table data classes."""
        from natcap.invest.reporting import table_generator

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        cols = [
            {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'}},
            {'name': 'num_pixels', 'total': True,
             'attr': {'class': 'my_class'}},
            {'name': 'wyield_vol', 'total': True,
             'attr': {'class': 'my_class'}}]

        table_args = {
            'cols': cols,
            'rows': dict_list,
            'checkbox': True,
            'checkbox_pos': 1,
            'total': True,
            'attributes': {'class': 'sorttable', 'id': 'parcel_table'}
        }

        result_str = table_generator.generate_table(table_args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'table_strings', 'table_string_no_td_classes.txt')
        regression_file = codecs.open(regression_path, 'rU', 'utf-8')
        regression_str = regression_file.read()

        self.assertEqual(result_str, regression_str)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_table_generator_no_col_attrs(self):
        """Reporting: testing table generator without column attributes."""
        from natcap.invest.reporting import table_generator

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        cols = [
            {'name': 'ws_id', 'total': False},
            {'name': 'num_pixels', 'total': True},
            {'name': 'wyield_vol', 'total': True}]

        table_args = {
            'cols': cols,
            'rows': dict_list,
            'checkbox': True,
            'checkbox_pos': 1,
            'total': True,
            'attributes': {'class': 'sorttable', 'id': 'parcel_table'}
        }

        result_str = table_generator.generate_table(table_args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'table_strings', 'table_string_no_col_attrs.txt')
        regression_file = codecs.open(regression_path, 'rU', 'utf-8')
        regression_str = regression_file.read()

        self.assertEqual(result_str, regression_str)

    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_table_generator_no_totals(self):
        """Reporting: testing table generator without totals."""
        from natcap.invest.reporting import table_generator

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]

        cols = [
            {'name': 'ws_id', 'total': False},
            {'name': 'num_pixels', 'total': False},
            {'name': 'wyield_vol', 'total': False}]

        table_args = {
            'cols': cols,
            'rows': dict_list,
            'checkbox': True,
            'checkbox_pos': 1,
            'total': False,
            'attributes': {'class': 'sorttable', 'id': 'parcel_table'}
        }

        result_str = table_generator.generate_table(table_args)

        regression_path = os.path.join(
            REGRESSION_DATA, 'table_strings', 'table_string_no_totals.txt')
        regression_file = codecs.open(regression_path, 'rU', 'utf-8')
        regression_str = regression_file.read()

        self.assertEqual(result_str, regression_str)
