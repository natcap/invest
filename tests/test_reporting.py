"""Module for Testing the InVEST Reporting module."""
import unittest
import tempfile
import shutil
import os
import csv

from osgeo import ogr
import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from nose.tools import nottest

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'natcap', 'invest')
REGRESSION_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-test-data', 'reporting')


class ReportingUnitTests(unittest.TestCase):
    """Unit tests for Reporting."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)
    @nottest
    def test_add_head_element_script(self):
        """ """
        param_args = {
            'format': 'script',
            'data_src': os.path.join(
                SAMPLE_DATA, 'reporting', 'reporting_data',
                'total_functions.js'),
            'input_type': 'File'
        }

        result = reporting.add_head_element(param_args)


    @nottest
    def test_add_head_element_json(self):
        """ """
        param_args = {
            'format': 'json',
            'data_src': os.path.join(
                SAMPLE_DATA, 'reporting', 'reporting_data', 'total_functions.js'),
            'input_type': 'Text'
        }

        reporting.add_head_element(param_args)

    @nottest
    def test_add_head_element_style(self):
        """ """
        param_args = {
            'format': 'style',
            'data_src': os.path.join(
                SAMPLE_DATA, 'reporting', 'reporting_data', 'total_functions.js'),
            'input_type': 'Text',
            'attributes': {'class': 'my_class'}
        }

        reporting.add_head_element(param_args)
    @nottest
    def test_add_head_element_exception(self):
        """ """
        from natcap.invest import reporting
        param_args = {
            'format': 'foobar',
            'data_src': os.path.join(
                SAMPLE_DATA, 'reporting', 'reporting_data', 'total_functions.js'),
            'input_type': 'File'
        }

        reporting.add_head_element(param_args)


class ReportingRegressionTests(unittest.TestCase):
    """Regression Tests for Reporting Module."""

    def setUp(self):
        """Overriding setUp function to create temporary workspace directory."""
        # this lets us delete the workspace after its done no matter the
        # the rest result
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Overriding tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report(self):
        """Reporting: testing full report."""
        from natcap.invest import reporting

        #workspace_dir = self.workspace_dir
        workspace_dir = REGRESSION_DATA

        csv_path = os.path.join(REGRESSION_DATA, 'sample_csv.csv')
        style_path = os.path.join(REGRESSION_DATA, 'sample_style.css')

        args = {
            'title': 'Test InVEST Reporting',
            'sortable': True,
            'totals': True,
            'out_uri': os.path.join(workspace_dir, 'test_reporting.html'),
            'elements': [
                {'type': 'table', 'section': 'body',
                 'attributes': {'class': 'my_class'}, 'sortable': True,
                 'checkbox': True, 'checkbox_pos': 2, 'data_type': 'csv',
                 'data': csv_path, 'key': 'ws_id',
                 'columns': [
                    {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'num_pixels', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'wyield_vol', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'}],
                 'total': True},
                {'type': 'head', 'section': 'head', 'format': 'style',
                 'data_src': style_path, 'input_type': 'File',
                 'attributes': {'class': 'my_class'}},
                {'type': 'text', 'section': 'body',
                 'text': '<p>Hello World This is InVEST Reporting.</p>'}]
        }

        reporting.generate_report(args)

        pygeoprocessing.testing.assert_text_equal(
            args['out_uri'],
            os.path.join(REGRESSION_DATA, 'test_reporting.html'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_script(self):
        """Reporting: testing full report."""
        from natcap.invest import reporting

        #workspace_dir = self.workspace_dir
        workspace_dir = REGRESSION_DATA

        script_path = os.path.join(REGRESSION_DATA, 'sample_script.js')

        dict_list = [
            {'ws_id': 0, 'num_pixels': 47017.0, 'wyield_vol': 50390640.85},
            {'ws_id': 1, 'num_pixels': 93339.0, 'wyield_vol': 103843576.83},
            {'ws_id': 2, 'num_pixels': 20977.0, 'wyield_vol': 21336791.14}]


        args = {
            'title': 'Test InVEST Reporting',
            'sortable': True,
            'totals': True,
            'out_uri': os.path.join(workspace_dir, 'test_reporting_script.html'),
            'elements': [
                {'type': 'table', 'section': 'body',
                 'attributes': {'id': 'my_id'}, 'sortable': True,
                 'checkbox': True, 'checkbox_pos': 1, 'data_type': 'dictionary',
                 'data': dict_list,
                 'columns': [
                    {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'num_pixels', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'wyield_vol', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'}],
                 'total': True},
                {'type': 'head', 'section': 'head', 'format': 'script',
                 'data_src': script_path, 'input_type': 'File',
                 'attributes': {'class': 'my_class'}},
                {'type': 'text', 'section': 'body',
                 'text': '<p>Hello World This is InVEST Reporting.</p>'}]
        }

        reporting.generate_report(args)

        pygeoprocessing.testing.assert_text_equal(
            args['out_uri'],
            os.path.join(REGRESSION_DATA, 'test_reporting_script.html'))

    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_json(self):
        """Reporting: testing full report."""
        from natcap.invest import reporting

        #workspace_dir = self.workspace_dir
        workspace_dir = REGRESSION_DATA

        shape_path = os.path.join(REGRESSION_DATA, 'sample_shape.shp')
        #json_path = os.path.join(REGRESSION_DATA, 'sample_json.json')
        json_data = "{'key': 0, 'data': {'door' : 1, 'room': 'kitchen'}}"

        args = {
            'title': 'Test InVEST Reporting',
            'sortable': True,
            'totals': True,
            'out_uri': os.path.join(workspace_dir, 'test_reporting_json.html'),
            'elements': [
                {'type': 'table', 'section': 'body', 'sortable': True,
                 'checkbox': True, 'checkbox_pos': 1, 'data_type': 'shapefile',
                 'data': shape_path, 'key': 'ws_id',
                 'columns': [
                    {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'num_pixels', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'wyield_vol', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'}],
                 'total': True},
                {'type': 'head', 'section': 'head', 'format': 'json',
                 'data_src': json_data, 'input_type': 'Text',
                 'attributes': {'class': 'my_class'}},
                {'type': 'text', 'section': 'body',
                 'text': '<p>Hello World This is InVEST Reporting.</p>'}]
        }

        reporting.generate_report(args)

        pygeoprocessing.testing.assert_text_equal(
            args['out_uri'],
            os.path.join(REGRESSION_DATA, 'test_reporting_json.html'))


    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_tagserror(self):
        """Reporting: testing full report."""
        from natcap.invest import reporting

        #workspace_dir = self.workspace_dir
        workspace_dir = REGRESSION_DATA

        shape_path = os.path.join(REGRESSION_DATA, 'sample_shape.shp')
        json_data = "<script> {'key': 0, 'data': {'door' : 1, 'room': 'kitchen'}} </script>"

        args = {
            'title': 'Test InVEST Reporting',
            'sortable': True,
            'totals': True,
            'out_uri': os.path.join(workspace_dir, 'test_reporting_json.html'),
            'elements': [
                {'type': 'table', 'section': 'body', 'sortable': True,
                 'checkbox': True, 'checkbox_pos': 1, 'data_type': 'shapefile',
                 'data': shape_path, 'key': 'ws_id',
                 'columns': [
                    {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'num_pixels', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'wyield_vol', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'}],
                 'total': True},
                {'type': 'head', 'section': 'head', 'format': 'json',
                 'data_src': json_data, 'input_type': 'Text',
                 'attributes': {'class': 'my_class'}},
                {'type': 'text', 'section': 'body',
                 'text': '<p>Hello World This is InVEST Reporting.</p>'}]
        }

        self.assertRaises(Exception, reporting.generate_report, args)


    @scm.skip_if_data_missing(SAMPLE_DATA)
    @scm.skip_if_data_missing(REGRESSION_DATA)
    def test_generate_report_headerror(self):
        """Reporting: testing full report."""
        from natcap.invest import reporting

        #workspace_dir = self.workspace_dir
        workspace_dir = REGRESSION_DATA

        shape_path = os.path.join(REGRESSION_DATA, 'sample_shape.shp')
        json_data = "{'key': 0, 'data': {'door' : 1, 'room': 'kitchen'}}"

        args = {
            'title': 'Test InVEST Reporting',
            'sortable': True,
            'totals': True,
            'out_uri': os.path.join(workspace_dir, 'test_reporting_json.html'),
            'elements': [
                {'type': 'table', 'section': 'body', 'sortable': True,
                 'checkbox': True, 'checkbox_pos': 1, 'data_type': 'shapefile',
                 'data': shape_path, 'key': 'ws_id',
                 'columns': [
                    {'name': 'ws_id', 'total': False, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'num_pixels', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'},
                    {'name': 'wyield_vol', 'total': True, 'attr': {'class': 'my_class'}, 'td_class': 'my_class'}],
                 'total': True},
                {'type': 'head', 'section': 'head', 'format': 'svg',
                 'data_src': json_data, 'input_type': 'Text',
                 'attributes': {'class': 'my_class'}},
                {'type': 'text', 'section': 'body',
                 'text': '<p>Hello World This is InVEST Reporting.</p>'}]
        }

        self.assertRaises(Exception, reporting.generate_report, args)

