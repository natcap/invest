import os
import unittest
import tempfile
import shutil
import json
import tarfile
import textwrap

import pygeoprocessing.testing
from pygeoprocessing.testing import scm
from osgeo import ogr
import six
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'invest-data')
FW_DATA = os.path.join(DATA_DIR, 'Base_Data', 'Freshwater')
POLLINATION_DATA = os.path.join(DATA_DIR, 'pollination')


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
            'c': 'plain bytestring',
            'd': '',
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

        scenarios.build_scenario_archive(params, 'sample_model', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')

        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        self.assertEqual(len(os.listdir(out_directory)), 3)

        self.assertEqual(
            json.load(open(os.path.join(out_directory,
                                        'parameters.json')))['args'],
            {'a': 1, 'b': u'hello there', 'c': u'plain bytestring', 'd': ''})

    @scm.skip_if_data_missing(FW_DATA)
    def test_collect_multipart_gdal_raster(self):
        from natcap.invest import scenarios
        params = {
            'raster': os.path.join(FW_DATA, 'dem'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')

        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))['args']

        self.assertEqual(len(archived_params), 1)
        pygeoprocessing.testing.assert_rasters_equal(
            params['raster'], os.path.join(out_directory,
                                           archived_params['raster']))

    @scm.skip_if_data_missing(POLLINATION_DATA)
    def test_collect_geotiff(self):
        # Necessary test, as this is proving to be an issue.
        from natcap.invest import scenarios
        params = {
            'raster': os.path.join(POLLINATION_DATA, 'landcover.tif'),
        }
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        dest_dir = os.path.join(self.workspace, 'extracted_archive')
        archived_params = scenarios.extract_scenario_archive(archive_path,
                                                             dest_dir)
        pygeoprocessing.testing.assert_rasters_equal(
            params['raster'],
            os.path.join(dest_dir, 'data', archived_params['raster']))

    @scm.skip_if_data_missing(FW_DATA)
    def test_collect_ogr_vector(self):
        from natcap.invest import scenarios
        source_vector_path = os.path.join(FW_DATA, 'watersheds.shp')
        source_vector = ogr.Open(source_vector_path)

        for format_name, extension in (('ESRI Shapefile', 'shp'),
                                       ('GeoJSON', 'geojson')):
            dest_dir = os.path.join(self.workspace, format_name)
            os.makedirs(dest_dir)
            dest_vector_path = os.path.join(dest_dir,
                                            'vector.%s' % extension)
            params = {
                'vector': dest_vector_path,
            }
            driver = ogr.GetDriverByName(format_name)
            driver.CopyDataSource(source_vector, dest_vector_path)

            archive_path = os.path.join(dest_dir,
                                        'archive.invs.tar.gz')

            # Collect the vector's files into a single archive
            scenarios.build_scenario_archive(params, 'sample_model',
                                             archive_path)

            # extract the archive
            out_directory = os.path.join(dest_dir, 'extracted_archive')
            with tarfile.open(archive_path) as tar:
                tar.extractall(out_directory)

            archived_params = json.load(
                open(os.path.join(out_directory, 'parameters.json')))['args']
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
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))['args']
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
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))['args']
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
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))['args']
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
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))['args']
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
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory, 'parameters.json')))['args']

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
            'blank': '',
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
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = scenarios.extract_scenario_archive(
            archive_path, out_directory)
        pygeoprocessing.testing.assert_rasters_equal(
            archive_params['raster'], params['raster'])
        pygeoprocessing.testing.assert_vectors_equal(
            archive_params['vector'], params['vector'], field_tolerance=1e-6)
        pygeoprocessing.testing.assert_csv_equal(
            archive_params['table'], params['table'])
        for key in ('blank', 'a', 'b', 'c'):
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

    def test_nested_args_keys(self):
        from natcap.invest import scenarios

        params = {
            'a': {
                'b': 1
            }
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        scenarios.build_scenario_archive(params, 'sample_model', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = scenarios.extract_scenario_archive(
            archive_path, out_directory)
        self.assertEqual(archive_params, params)

    def test_scenario_parameter_set(self):
        from natcap.invest import scenarios, __version__

        params = {
            'a': 1,
            'b': u'hello there',
            'c': 'plain bytestring',
            'nested': {
                'level1': 123,
            },
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
        modelname = 'natcap.invest.foo'
        paramset_filename = os.path.join(self.workspace, 'paramset.json')

        # Write the parameter set
        scenarios.write_parameter_set(paramset_filename, params, modelname)

        # Read back the parameter set
        args, invest_version, callable_name = scenarios.read_parameter_set(
            paramset_filename)

        # parameter set calculations normalizes all paths.
        # These are relative paths and must be patched.
        normalized_params = params.copy()
        for key in ('raster', 'vector', 'table'):
            normalized_params[key] = os.path.normpath(normalized_params[key])

        self.assertEqual(args, normalized_params)
        self.assertEqual(invest_version, __version__)
        self.assertEqual(callable_name, modelname)

    def test_relative_parameter_set(self):
        from natcap.invest import scenarios, __version__

        params = {
            'a': 1,
            'b': u'hello there',
            'c': 'plain bytestring',
            'nested': {
                'level1': 123,
            },
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
            'file_list': [
                os.path.join(self.workspace, 'file1.txt'),
                os.path.join(self.workspace, 'file2.txt'),
            ],
            'data_dir': os.path.join(self.workspace, 'data_dir'),
            'temp_workspace': self.workspace
        }
        modelname = 'natcap.invest.foo'
        paramset_filename = os.path.join(self.workspace, 'paramset.json')

        # make the sample data so filepaths are interpreted correctly
        for file_base in ('foo', 'bar', 'file1', 'file2'):
            test_filepath = os.path.join(self.workspace, file_base + '.txt')
            open(test_filepath, 'w').write('hello!')
        os.makedirs(params['data_dir'])

        # Write the parameter set
        scenarios.write_parameter_set(
            paramset_filename, params, modelname, relative=True)

        # Check that the written parameter set file contains relative paths
        raw_args = json.load(open(paramset_filename))['args']
        self.assertEqual(raw_args['foo'], 'foo.txt')
        self.assertEqual(raw_args['bar'], 'foo.txt')
        self.assertEqual(raw_args['file_list'], ['file1.txt', 'file2.txt'])
        self.assertEqual(raw_args['data_dir'], 'data_dir')
        self.assertEqual(raw_args['temp_workspace'], '.')

        # Read back the parameter set and verify the returned paths are
        # absolute
        args, invest_version, callable_name = scenarios.read_parameter_set(
            paramset_filename)

        self.assertEqual(args, params)
        self.assertEqual(invest_version, __version__)
        self.assertEqual(callable_name, modelname)

    def test_read_parameters_from_logfile(self):
        """Scenarios: Verify we can read args from a logfile."""
        from natcap.invest import scenarios
        logfile_path = os.path.join(self.workspace, 'logfile')
        with open(logfile_path, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                07/20/2017 16:37:48  natcap.invest.ui.model INFO
                Arguments:
                suffix                           foo
                some_int                         1
                some_float                       2.33
                workspace_dir                    some_workspace_dir

                07/20/2017 16:37:48  natcap.invest.ui.model INFO post args.
            """))

        params = scenarios.read_parameters_from_logfile(logfile_path)

        self.assertEqual(params, {u'suffix': u'foo',
                                  u'some_int': 1,
                                  u'some_float': 2.33,
                                  u'workspace_dir': u'some_workspace_dir'})

    def test_read_parameters_from_logfile_valueerror(self):
        """Scenarios: verify that valuerror raised when no params found."""
        from natcap.invest import scenarios
        logfile_path = os.path.join(self.workspace, 'logfile')
        with open(logfile_path, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                07/20/2017 16:37:48  natcap.invest.ui.model INFO
                07/20/2017 16:37:48  natcap.invest.ui.model INFO post args.
            """))

        with self.assertRaises(ValueError):
            scenarios.read_parameters_from_logfile(logfile_path)


class IsProbablyScenarioTests(unittest.TestCase):
    """Tests for our quick check for whether a file is a scenario."""

    def setUp(self):
        """Create a new workspace for each test."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the workspace created for each test."""
        shutil.rmtree(self.workspace)

    def test_json_extension(self):
        """Scenarios: invest.json extension is probably a scenario"""
        from natcap.invest import scenarios

        filepath = 'some_model.invest.json'
        self.assertTrue(scenarios.is_probably_scenario(filepath))

    def test_tar_gz_extension(self):
        """Scenarios: invest.tar.gz extension is probably a scenario"""
        from natcap.invest import scenarios

        filepath = 'some_model.invest.tar.gz'
        self.assertTrue(scenarios.is_probably_scenario(filepath))

    def test_parameter_set(self):
        """Scenarios: a parameter set should be a scenario."""
        from natcap.invest import scenarios

        filepath = os.path.join(self.workspace, 'paramset.json')
        args = {'foo': 'foo', 'bar': 'bar'}
        scenarios.write_parameter_set(filepath, args, 'test_model')

        self.assertTrue(scenarios.is_probably_scenario(filepath))

    def test_parameter_archive(self):
        """Scenarios: a parameter archive should be a scenario."""
        from natcap.invest import scenarios

        filepath = os.path.join(self.workspace, 'paramset.tar.gz')
        args = {'foo': 'foo', 'bar': 'bar'}
        scenarios.build_scenario_archive(args, 'test_model', filepath)

        self.assertTrue(scenarios.is_probably_scenario(filepath))

    def test_parameter_logfile(self):
        """Scenarios: a textfile should be interpreted as a scenario."""
        from natcap.invest import scenarios

        filepath = os.path.join(self.workspace, 'logfile.txt')
        with open(filepath, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                07/20/2017 16:37:48  natcap.invest.ui.model INFO
                Arguments:
                suffix                           foo
                some_int                         1
                some_float                       2.33
                workspace_dir                    some_workspace_dir

                07/20/2017 16:37:48  natcap.invest.ui.model INFO post args.
            """))

        self.assertTrue(scenarios.is_probably_scenario(filepath))

    def test_csv_not_a_parameter(self):
        """Scenarios: a CSV is probably not a parameter set."""
        from natcap.invest import scenarios

        filepath = os.path.join(self.workspace, 'sample.csv')
        with open(filepath, 'w') as sample_csv:
            sample_csv.write('A,B,C\n')
            sample_csv.write('"aaa","bbb","ccc"\n')

        self.assertFalse(scenarios.is_probably_scenario(filepath))


class UtilitiesTest(unittest.TestCase):
    def test_print_args(self):
        """Scenarios: verify that we format args correctly."""
        from natcap.invest.scenarios import format_args_dict

        args_dict = {
            'some_arg': [1, 2, 3, 4],
            'foo': 'bar',
        }

        args_string = format_args_dict(args_dict=args_dict)
        expected_string = six.text_type(
            'Arguments:\n'
            'foo      bar\n'
            'some_arg [1, 2, 3, 4]\n')
        self.assertEqual(args_string, expected_string)
