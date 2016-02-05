import importlib
import itertools
import logging
import os
import pkgutil
import shutil
import tempfile
import unittest

import shapely
import pygeoprocessing.testing
from pygeoprocessing.testing import scm

SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')

LOGGER = logging.getLogger('test_scenario_generator')

land_cover_array = [
    [1., 2., 3., 4., 5., 6.],
    [1., 2., 3., 4., 5., 6.],
    [1., 2., 3., 4., 5., 6.],
    [1., 2., 3., 4., 5., 6.],
    [1., 2., 3., 4., 5., 6.],
    [1., 2., 3., 4., 5., 6.]]

transition_likelihood_table = [
    ['Id', 'Name', 'Short', '1', '2', '3', '4', 'Percent Change',
        'Area Change', 'Priority', 'Proximity', 'Patch (ha)'],
    ['1', 'Grassland', 'Grassland', '0', '4', '0', '1', '0', '0',
        '0', '0', '0'],
    ['2', 'Agriculture', 'Agriculture', '0', '0', '0', '0', '0', '8000', '8',
        '5000', '0'],
    ['3', 'Forest', 'Forest', '0', '8', '0', '1', '0', '0', '0', '0', '0'],
    ['4', 'Bareland', 'Bareland', '0', '0', '0', '0', '25', '0', '5',
        '10000', '']]

land_suitability_factors_table = [
    ['id', 'Factorname', 'Layer', 'Wt', 'Suitfield', 'Dist', 'Cover'],
    ['2', 'roads', 'Roads.shp', '5', '', '10000', 'Smallscl']]

priority_table = [
    ['id', 'Name', 'Cover A', 'Cover B', 'Cover C', 'Priority'],
    ['1', 'Cover A', '1', '', '', ''],
    ['2', 'Cover B', '0.5', '1', '', ''],
    ['3', 'Cover C', '0.1', '5', '1', '']]

pairwise_comparison_table = [
    ['Record', 'Item', 'DistRoads', 'DistTown', 'Slope', 'PRIORITY'],
    ['1', 'DistRoads', '1', '', '', ''],
    ['2', 'DistTowns', '0.33', '1', '', ''],
    ['3', 'Slope', '0.1', '5', '1', '']]

transition_matrix = [
    ['', 'Cover A', 'Cover B', 'Cover C', 'Change'],
    ['Cover A', '0', '4', '0', '30%'],
    ['Cover B', '0', '0', '0', '0'],
    ['Cover C', '10', '2', '0', '-10%']]


def create_raster(raster_uri, array):
    pass


def create_shapefile(shapefile_uri, geometry):
    pass


def create_csv_table(table_uri, list):
    pass


def get_args():
    """Create test-case arguments for Scenario Generator model."""
    workspace_dir = os.path.join(os.path.dirname(__file__), 'workspace')
    if not os.path.exists(workspace_dir):
        os.mkdir(workspace_dir)

    land_cover_raster_uri = os.path.join(workspace_dir, 'lulc.tif')
    create_raster(land_cover_raster_uri, array)

    transition_matrix_uri = os.path.join(workspace_dir, 'transition.csv')
    create_raster(transition_matrix_uri, array)

    suitability_dir = os.path.join(
        os.path.dirname(__file__), 'workspace', 'suitability')
    if not os.path.exists(suitability_dir):
        os.mkdir(suitability_dir)

    priorities_csv_uri = os.path.join(workspace_dir, 'priorities.csv')
    create_csv_table(priorities_csv_uri, priorities_csv_list)

    suitability_factors_csv_uri = os.path.join(
        workspace_dir, 'suitability.csv')
    create_csv_table(suitability_factors_csv_uri, suitability_csv_list)

    constraints_shapefile_uri = os.path.join(workspace_dir, 'constraints.shp')
    constraints_geometry = None
    create_shapefile(constraints_shapefile_uri, constraints_geometry)

    override_shapefile_uri = os.path.join(workspace_dir, 'override.shp')
    override_geometry = None
    create_shapefile(override_shapefile_uri, override_geometry)

    args = {
        'workspace_dir': workspace_dir,
        'suffix': '',
        'landcover': land_cover_raster_uri,
        'transition': transition_matrix_uri,
        'calculate_priorities': True,
        'priorities_csv_uri': priorities_csv_uri,
        'calculate_proximity': True,
        'proximity_weight': 0.3,
        'calculate_transition': True,
        'transition_id': 'ID',
        'percent_field': 'Percent Change',
        'area_field': 'Area Change',
        'priority_field': 'Priority',
        'proximity_field': 'Proximity',
        'calculate_factors': True,
        'suitability_folder': suitability_dir,
        'suitability': suitability_factors_csv_uri,
        'weight': 0.5,
        'factor_inclusion': 0,
        'factors_field_container': True,
        'suitability_id': '',
        'suitability_layer': '',
        'suitability_field': '',
        'distance_field': '',
        'calculate_constraints': True,
        'constraints': constraints_shapefile_uri,
        'constraints_field': '',
        'override_layer': True,
        'override': override_shapefile_uri,
        'override_field': '',
        'override_inclusion': 0
    }

    return args


class UnitTests(unittest.TestCase):

    """Test functions in scenario generator model."""

    def test_calculate_weights(self):
        from natcap.invest import scenario_generator
        array = np.array([1, 2, 3])
        weights_list = scenario_generator.calculate_weights(array)

    def test_calculate_priority(self):
        from natcap.invest import scenario_generator
        args = get_args()
        priority_table_uri = ''
        priority_dict = scenario_generator.calculate_priority(
            priority_table_uri)
        shutil.rmtree(args['workspace_dir'])

    def test_calculate_distance_raster_uri(self):
        from natcap.invest import scenario_generator
        args = get_args()
        dataset_in_uri = ''
        dataset_out_uri = ''
        scenario_generator.calculate_distance_raster_uri(
            dataset_in_uri, dataset_out_uri)
        shutil.rmtree(args['workspace_dir'])

    def test_get_geometry_type_from_uri(self):
        from natcap.invest import scenario_generator
        args = get_args()
        datasource_uri = ''
        shape_type = scenario_generator.get_geometry_type_from_uri(
            datasource_uri)
        shutil.rmtree(args['workspace_dir'])

    def test_get_transition_set_count_from_uri(self):
        from natcap.invest import scenario_generator
        args = get_args()
        dataset_uri_list = ''
        unique_raster_values_count, transitions = \
            scenario_generator.get_transition_set_count_from_uri(
                dataset_uri_list)
        shutil.rmtree(args['workspace_dir'])

    def test_generate_chart_html(self):
        from natcap.invest import scenario_generator
        args = get_args()
        cover_dict = {}
        cover_names_dict = {}
        workspace_dir = ''
        chart_html = scenario_generator.generate_chart_html(
            cover_dict, cover_names_dict, workspace_dir)
        shutil.rmtree(args['workspace_dir'])

    def test_filter_fragments(self):
        from natcap.invest import scenario_generator
        args = get_args()
        input_uri = ''
        size = None
        output_uri = ''
        scenario_generator.filter_fragments(input_uri, size, output_uri)
        shutil.rmtree(args['workspace_dir'])
