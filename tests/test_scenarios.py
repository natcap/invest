import os
import unittest
import tempfile
import shutil
import json
import glob
import functools

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'invest-data')
FW_DATA = os.path.join(DATA_DIR, 'Base_Data', 'Freshwater')


class ScenariosTest(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_collect_simple_parameters(self):
        from natcap.invest import scenarios
        params = {
            'a': 1,
            'b': u'hello there',
            'c': 'plain bytestring'
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

        scenarios.collect_parameters(params, archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        scenarios.extract_archive(out_directory, archive_path)
        self.assertEqual(len(os.listdir(out_directory)), 2)

        self.assertEqual(
            json.load(open(os.path.join(out_directory, 'parameters.json'))),
            {'a': 1, 'b': u'hello there', 'c': u'plain bytestring'})

    @scm.skip_if_data_missing(FW_DATA)
    def test_collect_multipart_gdal_raster(self):
        from natcap.invest import scenarios
        params = {
            'raster': os.path.join(FW_DATA, 'dem'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.collect_parameters(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        scenarios.extract_archive(out_directory, archive_path)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))

        self.assertEqual(len(archived_params), 1)
        pygeoprocessing.testing.assert_rasters_equal(
            params['raster'], os.path.join(out_directory,
                                           archived_params['raster']))

    @scm.skip_if_data_missing(FW_DATA)
    def test_collect_multipart_ogr_vector(self):
        from natcap.invest import scenarios
        params = {
            'vector': os.path.join(FW_DATA, 'watersheds.shp'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.collect_parameters(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        scenarios.extract_archive(out_directory, archive_path)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        pygeoprocessing.testing.assert_vectors_equal(
            params['vector'], os.path.join(out_directory,
                                           archived_params['vector']),
            field_tolerance=1e-6,
        )

        self.assertEqual(len(archived_params), 1)  # sanity check

    @scm.skip_if_data_missing(FW_DATA)
    def test_collect_ogr_table(self):
        from natcap.invest import scenarios
        params = {
            'table': os.path.join(DATA_DIR, 'Carbon', 'carbon_pools_samp.csv'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.collect_parameters(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        scenarios.extract_archive(out_directory, archive_path)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        pygeoprocessing.testing.assert_csv_equal(
            params['table'], os.path.join(out_directory,
                                          archived_params['table'])
        )

        self.assertEqual(len(archived_params), 1)  # sanity check
