import os
import unittest
import tempfile
import shutil
import json
import glob

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

        # get the checksum of the dem
        dem_checksum = pygeoprocessing.testing.digest_file_list(
            glob.glob(os.path.join(params['raster'], '*')))

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.collect_parameters(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        scenarios.extract_archive(out_directory, archive_path)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        archived_raster_path = os.path.join(out_directory,
                                            archived_params['raster'])

        # there's an extra dem.aux file that appears in the file list from
        # using the GDAL internal method for getting the component filenames
        # with an ESRI binary grid.
        raster_file_list = [filename for filename in
                            glob.glob(os.path.join(archived_raster_path, '*'))
                            if not filename.endswith('dem.aux')]
        self.assertEqual(
            dem_checksum,
            pygeoprocessing.testing.digest_file_list(raster_file_list))
        self.assertEqual(len(archived_params), 1)

    @scm.skip_if_data_missing(FW_DATA)
    def test_collect_multipart_ogr_vector(self):
        from natcap.invest import scenarios
        params = {
            'vector': os.path.join(FW_DATA, 'watersheds.shp'),
        }

        # get the checksum of the dem
        checksum = pygeoprocessing.testing.digest_file_list(
            glob.glob(os.path.join(FW_DATA, 'watersheds.*')))

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.collect_parameters(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        scenarios.extract_archive(out_directory, archive_path)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        archived_vector_path = os.path.join(out_directory,
                                            archived_params['vector'])

        vector_file_list = [filename for filename in
                            glob.glob(os.path.join(archived_vector_path, '*'))]
        self.assertEqual(
            checksum,
            pygeoprocessing.testing.digest_file_list(vector_file_list))
        self.assertEqual(len(archived_params), 1)
