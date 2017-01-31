import os
import unittest
import tempfile
import shutil
import json
import tarfile

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

        scenarios.build_scenario(params, archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')

        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        self.assertEqual(len(os.listdir(out_directory)), 3)

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
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')

        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

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
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

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
            'table': os.path.join(DATA_DIR, 'carbon', 'carbon_pools_samp.csv'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        pygeoprocessing.testing.assert_csv_equal(
            params['table'], os.path.join(out_directory,
                                          archived_params['table'])
        )

        self.assertEqual(len(archived_params), 1)  # sanity check

    def test_nonspatial_single_file(self):
        from natcap.invest import scenarios

        params = {
            'some_file': os.path.join(self.workspace, 'foo.txt')
        }
        with open(params['some_file'], 'w') as textfile:
            textfile.write('some text here!')

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        pygeoprocessing.testing.assert_text_equal(
            params['some_file'], os.path.join(out_directory,
                                              archived_params['some_file'])
        )

        self.assertEqual(len(archived_params), 1)  # sanity check

    def test_data_dir(self):
        from natcap.invest import scenarios
        params = {
            'data_dir': os.path.join(self.workspace, 'data_dir')
        }
        os.makedirs(params['data_dir'])
        for filename in ('foo.txt', 'bar.txt', 'baz.txt'):
            data_filepath = os.path.join(params['data_dir'], filename)
            with open(data_filepath, 'w') as textfile:
                textfile.write(filename)

        # make a folder within the data folder.
        nested_folder = os.path.join(params['data_dir'], 'nested')
        os.makedirs(nested_folder)
        with open(os.path.join(nested_folder, 'nested.txt'), 'w') as textfile:
            textfile.write('hello, world!')

        src_datadir_digest = pygeoprocessing.testing.digest_folder(
            params['data_dir'])

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        dest_datadir_digest = pygeoprocessing.testing.digest_folder(
            os.path.join(out_directory, archived_params['data_dir']))

        self.assertEqual(len(archived_params), 1)  # sanity check
        if src_datadir_digest != dest_datadir_digest:
            self.fail('Digest mismatch: src:%s != dest:%s' % (
                src_datadir_digest, dest_datadir_digest))

    def test_list_of_inputs(self):
        from natcap.invest import scenarios
        params = {
            'file_list': [
                os.path.join(self.workspace, 'foo.txt'),
                os.path.join(self.workspace, 'bar.txt'),
            ]
        }
        for filename in params['file_list']:
            with open(filename, 'w') as textfile:
                textfile.write(filename)

        src_digest = pygeoprocessing.testing.digest_file_list(
            params['file_list'])

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))
        dest_digest = pygeoprocessing.testing.digest_file_list(
            [os.path.join(out_directory, filename)
             for filename in archived_params['file_list']])

        self.assertEqual(len(archived_params), 1)  # sanity check
        if src_digest != dest_digest:
            self.fail('Digest mismatch: src:%s != dest:%s' % (
                src_digest, dest_digest))

    def test_duplicate_filepaths(self):
        from natcap.invest import scenarios
        params = {
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
        }
        with open(params['foo'], 'w') as textfile:
            textfile.write('hello world!')

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))

        # Assert that the archived 'foo' and 'bar' params point to the same
        # file.
        self.assertEqual(archived_params['foo'], archived_params['bar'])

        # Assert we have the expected number of files in the archive
        self.assertEqual(len(os.listdir(os.path.join(out_directory))), 3)

        # Assert we have the expected number of files in the data dir.
        self.assertEqual(
            len(os.listdir(os.path.join(out_directory, 'data'))), 1)

    def test_archive_extraction(self):
        from natcap.invest import scenarios
        params = {
            'a': 1,
            'b': u'hello there',
            'c': 'plain bytestring',
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
            'file_list': [
                os.path.join(self.workspace, 'file1.txt'),
                os.path.join(self.workspace, 'file2.txt'),
            ],
            'data_dir': os.path.join(self.workspace, 'data_dir'),
            'raster': os.path.join(FW_DATA, 'dem'),
            'vector': os.path.join(FW_DATA, 'watersheds.shp'),
            'table': os.path.join(DATA_DIR, 'carbon', 'carbon_pools_samp.csv'),
        }
        # synthesize sample data
        os.makedirs(params['data_dir'])
        for filename in ('foo.txt', 'bar.txt', 'baz.txt'):
            data_filepath = os.path.join(params['data_dir'], filename)
            with open(data_filepath, 'w') as textfile:
                textfile.write(filename)

        with open(params['foo'], 'w') as textfile:
            textfile.write('hello world!')

        for filename in params['file_list']:
            with open(filename, 'w') as textfile:
                textfile.write(filename)

        # collect parameters:
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = scenarios.extract_parameters_archive(archive_path,
                                                              out_directory)
        pygeoprocessing.testing.assert_rasters_equal(
            archive_params['raster'], params['raster'])
        pygeoprocessing.testing.assert_vectors_equal(
            archive_params['vector'], params['vector'], field_tolerance=1e-6)
        pygeoprocessing.testing.assert_csv_equal(
            archive_params['table'], params['table'])
        for key in ('a', 'b', 'c'):
            self.assertEqual(archive_params[key],
                             params[key],
                             'Params differ for key %s' % key)

        for key in ('foo', 'bar'):
            pygeoprocessing.testing.assert_text_equal(
                archive_params[key], params[key])

        self.assertEqual(
            pygeoprocessing.testing.digest_file_list(
                archive_params['file_list']),
            pygeoprocessing.testing.digest_file_list(params['file_list']))

    def test_archive_extraction_symlink(self):
        from natcap.invest import scenarios
        params = {
            'a': 1,
            'b': u'hello there',
            'c': 'plain bytestring',
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
            'file_list': [
                os.path.join(self.workspace, 'file1.txt'),
                os.path.join(self.workspace, 'file2.txt'),
            ],
            'data_dir': os.path.join(self.workspace, 'data_dir'),
            'raster': os.path.join(FW_DATA, 'dem'),
            'vector': os.path.join(FW_DATA, 'watersheds.shp'),
            'table': os.path.join(DATA_DIR, 'carbon', 'carbon_pools_samp.csv'),
        }
        # synthesize sample data
        os.makedirs(params['data_dir'])
        for filename in ('foo.txt', 'bar.txt', 'baz.txt'):
            data_filepath = os.path.join(params['data_dir'], filename)
            with open(data_filepath, 'w') as textfile:
                textfile.write(filename)

        with open(params['foo'], 'w') as textfile:
            textfile.write('hello world!')

        for filename in params['file_list']:
            with open(filename, 'w') as textfile:
                textfile.write(filename)

        # collect parameters:
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario(params, archive_path, link_data=True)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = scenarios.extract_parameters_archive(archive_path,
                                                              out_directory)
        pygeoprocessing.testing.assert_rasters_equal(
            archive_params['raster'], params['raster'])
        pygeoprocessing.testing.assert_vectors_equal(
            archive_params['vector'], params['vector'], field_tolerance=1e-6)
        pygeoprocessing.testing.assert_csv_equal(
            archive_params['table'], params['table'])
        for key in ('a', 'b', 'c'):
            self.assertEqual(archive_params[key],
                             params[key],
                             'Params differ for key %s' % key)

        for key in ('foo', 'bar'):
            pygeoprocessing.testing.assert_text_equal(
                archive_params[key], params[key])

        self.assertEqual(
            pygeoprocessing.testing.digest_file_list(
                archive_params['file_list']),
            pygeoprocessing.testing.digest_file_list(params['file_list']))


    def test_sandbox_manager(self):
        from natcap.invest import scenarios

        with scenarios.sandbox_tempdir(suffix='foo',
                                       prefix='bar',
                                       dir=self.workspace) as new_dir:
            self.assertTrue(new_dir.startswith(self.workspace))
            basename = os.path.basename(new_dir)
            self.assertTrue(basename.startswith('bar'))
            self.assertTrue(basename.endswith('foo'))

            # trigger the exception handling for coverage.
            shutil.rmtree(new_dir)
