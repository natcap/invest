import importlib
import itertools
import logging
import os
import pkgutil
import shutil
import tempfile
import unittest
import csv
import pprint as pp

import numpy as np
from osgeo import gdal, ogr, osr
import shapely
from pygeoprocessing import geoprocessing as geoprocess
import pygeoprocessing.testing as pygeotest


SAMPLE_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'invest-data')

LOGGER = logging.getLogger('test_scenario_generator')

land_cover_array = [
    [1., 2., 3., 4.],
    [1., 2., 3., 4.],
    [1., 2., 3., 4.],
    [1., 2., 3., 4.]]

transition_likelihood_table = [
    ['Id', 'Name',        'Short',       '1', '2', '3', '4', 'Percent Change', 'Area Change', 'Priority', 'Proximity', 'Patch ha'],
    ['1',  'Grassland',   'Grassland',   '0', '4', '0', '1', '0',              '0',           '0',        '0',         '0'],
    ['2',  'Agriculture', 'Agriculture', '0', '0', '0', '0', '0',              '8000',        '8',        '5000',      '0'],
    ['3',  'Forest',      'Forest',      '0', '8', '0', '1', '0',              '0',           '0',        '0',         '0'],
    ['4',  'Bareland',    'Bareland',    '0', '0', '0', '0', '25',             '0',           '5',        '10000',     '0']]

land_suitability_factors_table = [
    ['Id', 'Cover ID', 'Factorname', 'Layer',     'Wt', 'Suitfield', 'Dist',  'Cover'],
    ['2',  '9', 'roads',      'roads.shp', '5',  '',          '10000', 'Smallscl']]

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
    ['',            'Grassland', 'Agriculture', 'Forest', 'Baseland', 'Change'],
    ['Grassland',   '0',         '4',           '0',      '0'         '30%'],
    ['Agriculture', '0',         '0',           '0',      '0',        '0'],
    ['Forest',      '10',        '2',           '0',      '0',        '-10%'],
    ['Baseland',    '0',         '0',           '0',      '0'         '0']]


def create_raster(raster_uri, array):
    """Create test raster."""
    srs = pygeotest.sampledata.SRS_WILLAMETTE
    pygeotest.create_raster_on_disk(
        [array],
        srs.origin,
        srs.projection,
        -1,
        srs.pixel_size(100),
        datatype=gdal.GDT_Int32,
        filename=raster_uri)
    return raster_uri


def create_shapefile(shapefile_uri, geometries, fields=None):
    """Create test shapefile."""
    srs = pygeotest.sampledata.SRS_WILLAMETTE
    return pygeotest.create_vector_on_disk(
        geometries,
        srs.projection,
        fields=fields,
        attributes=None,
        vector_format='ESRI Shapefile',
        filename=shapefile_uri)


def create_csv_table(table_uri, rows_list):
    """Create csv file from list of lists."""
    with open(table_uri, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows_list)
    return table_uri


def get_args():
    """Create test-case arguments for Scenario Generator model."""
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

    srs = pygeotest.sampledata.SRS_WILLAMETTE
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
        'workspace_dir': workspace_dir,                  # workspace directory
        'suffix': '',                                    #
        'landcover': land_cover_raster_uri,              # land cover raster
        'transition': transition_likelihood_uri,         # transition matrix
        'calculate_priorities': True,                    # use relative priorities
        'priorities_csv_uri': priorities_csv_uri,        #   relative priorities
        'calculate_proximity': True,                     #
        'calculate_transition': True,                    #
        'calculate_factors': True,                       #
        'suitability_folder': suitability_dir,           # suitability shapefiles folder
        'suitability': suitability_factors_csv_uri,      # suitability factors
        'weight': 0.5,                                   # factor weight
        'factor_inclusion': 0,                           # all_touched=True for vectorize_datasets
        'factors_field_container': True,                 #
        'calculate_constraints': True,                   #
        'constraints': constraints_shapefile_uri,        #
        'constraints_field': 'protlevel',                #
        'override_layer': True,                          #
        'override': override_shapefile_uri,              #
        'override_field': 'newclass',                    #
        'override_inclusion': 0                          #
    }

    return args


class ModelTests(unittest.TestCase):

    """Test execute function in scenario generator model."""

    def test_execute(self):
        import natcap.invest.scenario_generator as sg
        args = get_args()
        sg.scenario_generator.execute(args)
        shutil.rmtree(args['workspace_dir'])


class UnitTests(unittest.TestCase):

    """Test functions in scenario generator model."""

    def test_calculate_weights(self):
        from natcap.invest import scenario_generator as sg
        array = np.array([1, 2, 3])
        weights_list = sg.scenario_generator.calculate_weights(array)

    def test_calculate_priority(self):
        from natcap.invest import scenario_generator as sg
        args = get_args()
        priority_table_uri = ''
        priority_dict = sg.scenario_generator.calculate_priority(
            priority_table_uri)
        shutil.rmtree(args['workspace_dir'])

    def test_calculate_distance_raster_uri(self):
        from natcap.invest import scenario_generator as sg
        args = get_args()
        dataset_in_uri = ''
        dataset_out_uri = ''
        sg.scenario_generator.calculate_distance_raster_uri(
            dataset_in_uri, dataset_out_uri)
        shutil.rmtree(args['workspace_dir'])

    def test_get_geometry_type_from_uri(self):
        from natcap.invest import scenario_generator as sg
        args = get_args()
        datasource_uri = ''
        shape_type = sg.scenario_generator.get_geometry_type_from_uri(
            datasource_uri)
        shutil.rmtree(args['workspace_dir'])

    def test_get_transition_set_count_from_uri(self):
        from natcap.invest import scenario_generator as sg
        args = get_args()
        dataset_uri_list = ''
        unique_raster_values_count, transitions = \
            sg.scenario_generator.get_transition_set_count_from_uri(
                dataset_uri_list)
        shutil.rmtree(args['workspace_dir'])

    def test_generate_chart_html(self):
        from natcap.invest import scenario_generator as sg
        args = get_args()
        cover_dict = {}
        cover_names_dict = {}
        workspace_dir = ''
        chart_html = sg.scenario_generator.generate_chart_html(
            cover_dict, cover_names_dict, workspace_dir)
        shutil.rmtree(args['workspace_dir'])

    def test_filter_fragments(self):
        from natcap.invest import scenario_generator as sg
        args = get_args()
        input_uri = ''
        size = None
        output_uri = ''
        sg.scenario_generator.filter_fragments(input_uri, size, output_uri)
        shutil.rmtree(args['workspace_dir'])


if __name__ == '__main__':
    unittest.main()
