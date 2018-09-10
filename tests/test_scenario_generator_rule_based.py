# -*- coding: utf-8 -*-
"""Tests for Scenario Generator model."""

import logging
import os
import shutil
import unittest
import csv
from decimal import Decimal
import hashlib

import numpy as np
from osgeo import gdal
import shapely
import pygeoprocessing.testing


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')

LOGGER = logging.getLogger('test_scenario_generator')

land_cover_array = [
    [1., 2., 3., 4.],
    [1., 2., 3., 4.],
    [1., 2., 3., 4.],
    [1., 2., 3., 4.]]

transition_likelihood_table = [
    ['Id', 'Name', 'Short', '1', '2', '3', '4', 'Percent Change',
        'Area Change', 'Priority', 'Proximity', 'Patch ha'],
    ['1', 'Grassland', 'Grassland', '0', '4', '0', '1', '0', '0', '0', '0',
        '0'],
    ['2', 'Agriculture', 'Agriculture', '0', '0', '0', '0', '0', '8000', '8',
        '5000', '0'],
    ['3', 'Forest', 'Forest', '0', '8', '0', '1', '0', '0', '0', '0', '0'],
    ['4', 'Bareland', 'Bareland', '0', '0', '0', '0', '25', '0', '5', '10000',
        '0']]

land_suitability_factors_table = [
    ['Id', 'Cover ID', 'Factorname', 'Layer', 'Wt', 'Suitfield', 'Dist',
        'Cover'],
    ['2', '9', 'roads', 'roads.shp', '5',  '', '10000', 'Smallscl']]

priority_table = [
    ['Id', 'Name',        '1',   '2', '3', '4', 'Priority'],
    ['1',  'Grassland',   '1',   '',  '',  '',  ''],
    ['2',  'Agriculture', '0.5', '1', '',  '',  ''],
    ['3',  'Forest',      '0.1', '5', '1', '',  ''],
    ['4',  'Bareland',    '0.1', '5', '1', '1', '']]

pairwise_comparison_table = [
    ['Record', 'Item',      'DistRoads', 'DistTown', 'Slope', 'PRIORITY'],
    ['1',      'DistRoads', '1',         '',         '',      ''],
    ['2',      'DistTowns', '0.33',      '1',        '',      ''],
    ['3',      'Slope',     '0.1',       '5',        '1',     '']]

transition_matrix = [
    ['', 'Grassland', 'Agriculture', 'Forest', 'Baseland', 'Change'],
    ['Grassland',   '0',         '4',           '0',      '0'         '30%'],
    ['Agriculture', '0',         '0',           '0',      '0',        '0'],
    ['Forest',      '10',        '2',           '0',      '0',        '-10%'],
    ['Baseland',    '0',         '0',           '0',      '0'         '0']]


def read_raster(raster_path):
    """"Read raster as array.

    Args:
        raster_path (str): file path to raster.

    Returns:
        a (np.array): raster's first band as an array.
    """
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    ds = None
    return array


def create_raster(raster_path, array):
    """Create test raster.

    Args:
        raster_path (str): path to output raster.
        array (np.array): input array.

    Returns:
        raster_path (str): path to output raster.
    """
    srs = pygeoprocessing.testing.sampledata.SRS_WILLAMETTE
    pygeoprocessing.testing.create_raster_on_disk(
        [array],
        srs.origin,
        srs.projection,
        -1,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=raster_path)
    return raster_path


def create_shapefile(shapefile_path, geometries, fields=None):
    """Create test shapefile.

    Args:
        shapefile_path (str): path to shapefile.
        geometries (list): list of shapely geometry objects
    """
    srs = pygeoprocessing.testing.sampledata.SRS_WILLAMETTE
    return pygeoprocessing.testing.create_vector_on_disk(
        geometries,
        srs.projection,
        fields=fields,
        attributes=None,
        vector_format='ESRI Shapefile',
        filename=shapefile_path)


def create_csv_table(table_path, rows_list):
    """Create csv file from list of lists.

    Args:
        table_path (str): file path to table.
        rows_list (list): nested list of elements to write to table.

    Returns:
        table_path (str): filepath to table.
    """
    with open(table_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows_list)
    return table_path


def get_args():
    """Create test-case arguments for Scenario Generator model.

    Returns:
        args (dict): main model arguments.
    """
    workspace_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'workspace')
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    if not os.path.exists(workspace_dir):
        os.mkdir(workspace_dir)

    array = np.array(land_cover_array)
    land_cover_raster_uri = os.path.join(workspace_dir, 'lulc.tif')
    create_raster(land_cover_raster_uri, array)

    transition_likelihood_uri = os.path.join(
        workspace_dir, 'scenario_transition_likelihood.csv')
    create_csv_table(transition_likelihood_uri, transition_likelihood_table)

    scenario_transition_uri = os.path.join(
        workspace_dir, 'scenario_transition.csv')
    create_csv_table(scenario_transition_uri, transition_matrix)

    suitability_dir = os.path.join(workspace_dir, 'suitability')
    if not os.path.exists(suitability_dir):
        os.mkdir(suitability_dir)

    priorities_csv_uri = os.path.join(workspace_dir, 'priorities.csv')
    create_csv_table(priorities_csv_uri, priority_table)

    suitability_factors_csv_uri = os.path.join(
        workspace_dir, 'suitability.csv')
    create_csv_table(
        suitability_factors_csv_uri, land_suitability_factors_table)

    srs = pygeoprocessing.testing.sampledata.SRS_WILLAMETTE
    x, y = srs.origin

    constraints_shapefile_uri = os.path.join(workspace_dir, 'constraints.shp')
    constraints_geometry = [shapely.geometry.Point(x+20., y-20.).buffer(1.0)]
    create_shapefile(constraints_shapefile_uri, constraints_geometry)

    override_shapefile_uri = os.path.join(workspace_dir, 'override.shp')
    override_geometry = [shapely.geometry.Point(x+240., y-240.).buffer(1.0)]
    create_shapefile(override_shapefile_uri, override_geometry)

    roads_shapefile_uri = os.path.join(suitability_dir, 'roads.shp')
    roads_geometry = [shapely.geometry.LineString(
        [(x+100., y-0.), (x+100., y-240.)])]
    create_shapefile(roads_shapefile_uri, roads_geometry)

    args = {
        'workspace_dir': workspace_dir,              # workspace directory
        'suffix': '',                                #
        'landcover': land_cover_raster_uri,          # land cover raster
        'transition': transition_likelihood_uri,     # transition matrix
        'calculate_priorities': True,                # use relative priorities
        'priorities_csv_uri': priorities_csv_uri,    # relative priorities
        'calculate_proximity': True,                 #
        'calculate_transition': True,                #
        'calculate_factors': True,                   #
        'suitability_folder': suitability_dir,       # suitability shapefiles
                                                     # folder
        'suitability': suitability_factors_csv_uri,  # suitability factors
        'weight': 0.5,                               # factor weight
        'factor_inclusion': 0,                       # all_touched=True for
                                                     # vectorize_datasets
        'factors_field_container': True,             #
        'calculate_constraints': True,               #
        'constraints': constraints_shapefile_uri,    #
        'constraints_field': 'protlevel',            #
        'override_layer': True,                      #
        'override': override_shapefile_uri,          #
        'override_field': 'newclass',                #
        'override_inclusion': 0                      #
    }

    return args


class ModelTests(unittest.TestCase):
    """Test execute function in scenario generator model."""

    def setUp(self):
        """Setup."""
        self.args = get_args()

    def test_execute(self):
        """Scenario Generator: Test Execute."""
        import natcap.invest.scenario_generator as sg
        sg.scenario_generator.execute(self.args)
        array = read_raster(os.path.join(
            self.args['workspace_dir'], 'intermediate', 'scenario.tif'))
        self.assertTrue(4 in array[0])
        self.assertTrue(2 in array)

    def tearDown(self):
        """Tear Down."""
        shutil.rmtree(self.args['workspace_dir'])


class UnitTests(unittest.TestCase):
    """Test functions in scenario generator model."""

    def setUp(self):
        """Setup."""
        self.args = get_args()

    def test_calculate_weights(self):
        """Scenario Generator: test calculate weights."""
        from natcap.invest import scenario_generator as sg
        array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        weights_list = sg.scenario_generator.calculate_weights(array)
        self.assertEqual(weights_list[0], Decimal('0.3333'))

    def test_calculate_priority(self):
        """Scenario Generator: test calculate priority."""
        from natcap.invest import scenario_generator as sg
        priority_table_uri = self.args['priorities_csv_uri']
        priority_dict = sg.scenario_generator.calculate_priority(
            priority_table_uri)
        self.assertEqual(priority_dict[1], Decimal('0.6430'))

    def test_calculate_distance_raster_uri(self):
        """Scenario Generator: test calculate distance raster."""
        from natcap.invest import scenario_generator as sg
        dataset_in_uri = os.path.join(
            self.args['workspace_dir'], 'dataset_in.tif')
        array = np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        create_raster(dataset_in_uri, array)
        dataset_out_uri = os.path.join(
            self.args['workspace_dir'], 'dataset_out.tif')
        sg.scenario_generator.calculate_distance_raster_uri(
            dataset_in_uri, dataset_out_uri)
        guess = read_raster(dataset_out_uri)
        np.testing.assert_almost_equal(guess[0, 1], np.array([100.]))

    def test_get_geometry_type_from_uri(self):
        """Scenario Generator: test get geometry type."""
        from natcap.invest import scenario_generator as sg
        datasource_uri = self.args['constraints']
        shape_type = sg.scenario_generator.get_geometry_type_from_uri(
            datasource_uri)
        self.assertEqual(shape_type, 5)

    def test_get_transition_set_count_from_uri(self):
        """Scenario Generator: test get transition set count."""
        from natcap.invest import scenario_generator as sg
        dataset_uri_list = [self.args['landcover'], self.args['landcover']]
        unique_raster_values_count, transitions = \
            sg.scenario_generator.get_transition_pairs_count_from_uri(
                dataset_uri_list)
        self.assertEqual(
            unique_raster_values_count.values()[0],
            {1: 4, 2: 4, 3: 4, 4: 4})

    def test_generate_chart_html(self):
        """Scenario Generator: test generate chart html."""
        from natcap.invest import scenario_generator as sg
        hash_md5 = hashlib.md5()

        cover_dict = {9.: (1., 2.)}
        cover_names_dict = {'Cover': 'Cover'}
        chart_html = sg.scenario_generator.generate_chart_html(
            cover_dict, cover_names_dict, self.args['workspace_dir'])
        chart_html = ''.join(format(ord(x), 'b') for x in chart_html)

        hash_md5.update(chart_html)
        self.assertEqual(
            hash_md5.hexdigest(), '4b77891d5f88dd02621350fa16b95891')

    def test_filter_fragments(self):
        """Scenario Generator: test filter fragments."""
        from natcap.invest import scenario_generator as sg
        input_uri = self.args['landcover']
        size = 200.
        output_uri = os.path.join(
            self.args['workspace_dir'], 'fragments_output.tif')
        sg.scenario_generator.filter_fragments(
            input_uri, size, output_uri)
        self.assertEqual(read_raster(output_uri)[0, 1], 0.)

    def tearDown(self):
        """Tear Down."""
        shutil.rmtree(self.args['workspace_dir'])


if __name__ == '__main__':
    unittest.main()
