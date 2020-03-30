import os
import unittest
import tempfile
import shutil
import json
import tarfile
import textwrap

import pygeoprocessing.testing
from osgeo import ogr
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'invest-test-data', 'data_stack')


class DatastacksTest(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace)

    def test_collect_simple_parameters(self):
        from natcap.invest import datastack
        params = {
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'd': '',
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

        datastack.build_datastack_archive(params, 'sample_model', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')

        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        self.assertEqual(len(os.listdir(out_directory)), 3)

        self.assertEqual(
            json.load(open(
                os.path.join(out_directory,
                             datastack.DATASTACK_PARAMETER_FILENAME)))['args'],
            {'a': 1, 'b': 'hello there', 'c': 'plain bytestring', 'd': ''})

    def test_collect_multipart_gdal_raster(self):
        from natcap.invest import datastack
        params = {
            'raster': os.path.join(DATA_DIR, 'dem'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')

        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory,
                              datastack.DATASTACK_PARAMETER_FILENAME)))['args']

        self.assertEqual(len(archived_params), 1)
        pygeoprocessing.testing.assert_rasters_equal(
            params['raster'], os.path.join(out_directory,
                                           archived_params['raster']))

    def test_collect_geotiff(self):
        # Necessary test, as this is proving to be an issue.
        from natcap.invest import datastack
        params = {
            'raster': os.path.join(DATA_DIR, 'landcover.tif'),
        }
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        dest_dir = os.path.join(self.workspace, 'extracted_archive')
        archived_params = datastack.extract_datastack_archive(archive_path,
                                                              dest_dir)
        pygeoprocessing.testing.assert_rasters_equal(
            params['raster'],
            os.path.join(dest_dir, 'data', archived_params['raster']))

    def test_collect_ogr_vector(self):
        from natcap.invest import datastack
        source_vector_path = os.path.join(DATA_DIR, 'watersheds.shp')
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
            datastack.build_datastack_archive(params, 'sample_model',
                                              archive_path)

            # extract the archive
            out_directory = os.path.join(dest_dir, 'extracted_archive')
            with tarfile.open(archive_path) as tar:
                tar.extractall(out_directory)

            archived_params = json.load(
                open(os.path.join(
                    out_directory,
                    datastack.DATASTACK_PARAMETER_FILENAME)))['args']
            pygeoprocessing.testing.assert_vectors_equal(
                params['vector'], os.path.join(out_directory,
                                               archived_params['vector']),
                field_tolerance=1e-6,
            )

            self.assertEqual(len(archived_params), 1)  # sanity check

    def test_collect_ogr_table(self):
        from natcap.invest import datastack
        params = {
            'table': os.path.join(DATA_DIR, 'carbon_pools_samp.csv'),
        }

        # Collect the raster's files into a single archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(
                out_directory,
                datastack.DATASTACK_PARAMETER_FILENAME)))['args']
        pygeoprocessing.testing.assert_csv_equal(
            params['table'], os.path.join(out_directory,
                                          archived_params['table'])
        )

        self.assertEqual(len(archived_params), 1)  # sanity check

    def test_nonspatial_single_file(self):
        from natcap.invest import datastack

        params = {
            'some_file': os.path.join(self.workspace, 'foo.txt')
        }
        with open(params['some_file'], 'w') as textfile:
            textfile.write('some text here!')

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory,
                              datastack.DATASTACK_PARAMETER_FILENAME)))['args']
        pygeoprocessing.testing.assert_text_equal(
            params['some_file'], os.path.join(out_directory,
                                              archived_params['some_file'])
        )

        self.assertEqual(len(archived_params), 1)  # sanity check

    def test_data_dir(self):
        from natcap.invest import datastack
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
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory,
                              datastack.DATASTACK_PARAMETER_FILENAME)))['args']
        dest_datadir_digest = pygeoprocessing.testing.digest_folder(
            os.path.join(out_directory, archived_params['data_dir']))

        self.assertEqual(len(archived_params), 1)  # sanity check
        if src_datadir_digest != dest_datadir_digest:
            self.fail('Digest mismatch: src:%s != dest:%s' % (
                src_datadir_digest, dest_datadir_digest))

    def test_list_of_inputs(self):
        from natcap.invest import datastack
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
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory,
                              datastack.DATASTACK_PARAMETER_FILENAME)))['args']
        dest_digest = pygeoprocessing.testing.digest_file_list(
            [os.path.join(out_directory, filename)
             for filename in archived_params['file_list']])

        self.assertEqual(len(archived_params), 1)  # sanity check
        if src_digest != dest_digest:
            self.fail('Digest mismatch: src:%s != dest:%s' % (
                src_digest, dest_digest))

    def test_duplicate_filepaths(self):
        from natcap.invest import datastack
        params = {
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
        }
        with open(params['foo'], 'w') as textfile:
            textfile.write('hello world!')

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        with tarfile.open(archive_path) as tar:
            tar.extractall(out_directory)

        archived_params = json.load(
            open(os.path.join(out_directory,
                              datastack.DATASTACK_PARAMETER_FILENAME)))['args']

        # Assert that the archived 'foo' and 'bar' params point to the same
        # file.
        self.assertEqual(archived_params['foo'], archived_params['bar'])

        # Assert we have the expected number of files in the archive
        self.assertEqual(len(os.listdir(os.path.join(out_directory))), 3)

        # Assert we have the expected number of files in the data dir.
        self.assertEqual(
            len(os.listdir(os.path.join(out_directory, 'data'))), 1)

    def test_archive_extraction(self):
        from natcap.invest import datastack
        params = {
            'blank': '',
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
            'file_list': [
                os.path.join(self.workspace, 'file1.txt'),
                os.path.join(self.workspace, 'file2.txt'),
            ],
            'data_dir': os.path.join(self.workspace, 'data_dir'),
            'raster': os.path.join(DATA_DIR, 'dem'),
            'vector': os.path.join(DATA_DIR, 'watersheds.shp'),
            'table': os.path.join(DATA_DIR, 'carbon_pools_samp.csv'),
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
        datastack.build_datastack_archive(params, 'sample_model', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = datastack.extract_datastack_archive(
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
        from natcap.invest import datastack

        params = {
            'a': {
                'b': 1
            }
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = datastack.extract_datastack_archive(
            archive_path, out_directory)
        self.assertEqual(archive_params, params)

    def test_datastack_parameter_set(self):
        from natcap.invest import datastack, __version__

        params = {
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'd': 'true',
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
            'raster': os.path.join(DATA_DIR, 'dem'),
            'vector': os.path.join(DATA_DIR, 'watersheds.shp'),
            'table': os.path.join(DATA_DIR, 'carbon', 'carbon_pools_samp.csv'),
        }
        modelname = 'natcap.invest.foo'
        paramset_filename = os.path.join(self.workspace, 'paramset.json')

        # Write the parameter set
        datastack.build_parameter_set(params, modelname, paramset_filename)

        # Read back the parameter set
        args, callable_name, invest_version = datastack.extract_parameter_set(
            paramset_filename)

        # parameter set calculations normalizes all paths.
        # These are relative paths and must be patched.
        normalized_params = params.copy()
        normalized_params['d'] = True  # should be read in as a bool
        for key in ('raster', 'vector', 'table'):
            normalized_params[key] = os.path.normpath(normalized_params[key])

        self.assertEqual(args, normalized_params)
        self.assertEqual(invest_version, __version__)
        self.assertEqual(callable_name, modelname)

    def test_relative_parameter_set(self):
        from natcap.invest import datastack, __version__

        params = {
            'a': 1,
            'b': 'hello there',
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
        datastack.build_parameter_set(
            params, modelname, paramset_filename, relative=True)

        # Check that the written parameter set file contains relative paths
        raw_args = json.load(open(paramset_filename))['args']
        self.assertEqual(raw_args['foo'], 'foo.txt')
        self.assertEqual(raw_args['bar'], 'foo.txt')
        self.assertEqual(raw_args['file_list'], ['file1.txt', 'file2.txt'])
        self.assertEqual(raw_args['data_dir'], 'data_dir')
        self.assertEqual(raw_args['temp_workspace'], '.')

        # Read back the parameter set and verify the returned paths are
        # absolute
        args, callable_name, invest_version = datastack.extract_parameter_set(
            paramset_filename)

        self.assertEqual(args, params)
        self.assertEqual(invest_version, __version__)
        self.assertEqual(callable_name, modelname)

    def test_extract_parameters_from_logfile(self):
        """Datastacks: Verify we can read args from a logfile."""
        from natcap.invest import datastack
        logfile_path = os.path.join(self.workspace, 'logfile')
        with open(logfile_path, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                07/20/2017 16:37:48  natcap.invest.ui.model INFO
                Arguments for InVEST some_model some_version:
                suffix                           foo
                some_int                         1
                some_float                       2.33
                workspace_dir                    some_workspace_dir

                07/20/2017 16:37:48  natcap.invest.ui.model INFO post args.
            """))

        params = datastack.extract_parameters_from_logfile(logfile_path)

        expected_params = datastack.ParameterSet(
            {'suffix': 'foo',
             'some_int': 1,
             'some_float': 2.33,
             'workspace_dir': 'some_workspace_dir'},
            'some_model',
            'some_version')

        self.assertEqual(params, expected_params)

    def test_extract_parameters_from_logfile_valueerror(self):
        """Datastacks: verify that valuerror raised when no params found."""
        from natcap.invest import datastack
        logfile_path = os.path.join(self.workspace, 'logfile')
        with open(logfile_path, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                07/20/2017 16:37:48  natcap.invest.ui.model INFO
                07/20/2017 16:37:48  natcap.invest.ui.model INFO post args.
            """))

        with self.assertRaises(ValueError):
            datastack.extract_parameters_from_logfile(logfile_path)

    def test_get_datastack_info_archive(self):
        """Datastacks: verify we can get info from an archive."""
        import natcap.invest
        from natcap.invest import datastack

        params = {
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'd': '',
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        datastack.build_datastack_archive(params, 'sample_model', archive_path)

        stack_type, stack_info = datastack.get_datastack_info(archive_path)

        self.assertEqual(stack_type, 'archive')
        self.assertEqual(stack_info, datastack.ParameterSet(
            params, 'sample_model', natcap.invest.__version__))

    def test_get_datatack_info_parameter_set(self):
        import natcap.invest
        from natcap.invest import datastack

        params = {
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'd': '',
        }

        json_path = os.path.join(self.workspace, 'archive.invs.json')
        datastack.build_parameter_set(params, 'sample_model', json_path)

        stack_type, stack_info = datastack.get_datastack_info(json_path)
        self.assertEqual(stack_type, 'json')
        self.assertEqual(stack_info, datastack.ParameterSet(
            params, 'sample_model', natcap.invest.__version__))

    def test_get_datastack_info_logfile_new_style(self):
        import natcap.invest
        from natcap.invest import datastack
        args = {
            'a': 1,
            'b': 2.7,
            'c': [1, 2, 3.55],
            'd': 'hello, world!',
            'e': False,
        }

        logfile_path = os.path.join(self.workspace, 'logfile.txt')
        with open(logfile_path, 'w') as logfile:
            logfile.write(datastack.format_args_dict(args, 'some_modelname'))

        stack_type, stack_info = datastack.get_datastack_info(logfile_path)
        self.assertEqual(stack_type, 'logfile')
        self.assertEqual(stack_info, datastack.ParameterSet(
            args, 'some_modelname', natcap.invest.__version__))

    def test_get_datastack_info_logfile_iui_style(self):
        from natcap.invest import datastack

        logfile_path = os.path.join(self.workspace, 'logfile.txt')
        with open(logfile_path, 'w') as logfile:
            logfile.write(textwrap.dedent("""
                Arguments:
                suffix                           foo
                some_int                         1
                some_float                       2.33
                workspace_dir                    some_workspace_dir

                some other logging here.
            """))

        expected_args = {
            'suffix': 'foo',
            'some_int': 1,
            'some_float': 2.33,
            'workspace_dir': 'some_workspace_dir',
        }

        stack_type, stack_info = datastack.get_datastack_info(logfile_path)
        self.assertEqual(stack_type, 'logfile')
        self.assertEqual(stack_info, datastack.ParameterSet(
            expected_args, datastack.UNKNOWN, datastack.UNKNOWN))

    def test_mixed_path_separators_in_paramset(self):
        """Datastacks: parameter sets must handle windows and linux paths."""
        from natcap.invest import datastack

        args = {
            'windows_path': os.path.join(self.workspace,
                                         'dir1\\filepath1.txt'),
            'linux_path': os.path.join(self.workspace,
                                       'dir2/filepath2.txt'),
        }
        for filepath in args.values():
            normalized_path = os.path.normpath(filepath.replace('\\', os.sep))
            try:
                os.makedirs(os.path.dirname(normalized_path))
            except OSError:
                pass

            with open(normalized_path, 'w') as open_file:
                open_file.write('the contents of this file do not matter.')

        paramset_path = os.path.join(self.workspace, 'paramset.invest.json')
        datastack.build_parameter_set(args, 'sample_model', paramset_path,
                                      relative=True)

        with open(paramset_path) as saved_parameters:
            args = json.loads(saved_parameters.read())['args']
            expected_args = {
                'windows_path': os.path.join('dir1', 'filepath1.txt'),
                'linux_path': os.path.join('dir2', 'filepath2.txt'),
            }
            self.assertEqual(expected_args, args)

        expected_args = {
            'windows_path': os.path.join(self.workspace, 'dir1',
                                         'filepath1.txt'),
            'linux_path': os.path.join(self.workspace, 'dir2',
                                       'filepath2.txt'),
        }

        extracted_paramset = datastack.extract_parameter_set(paramset_path)
        self.assertEqual(extracted_paramset.args, expected_args)

    def test_mixed_path_separators_in_archive(self):
        """Datastacks: datastack archives must handle windows, linux paths."""
        from natcap.invest import datastack

        args = {
            'windows_path': os.path.join(self.workspace,
                                         'dir1\\filepath1.txt'),
            'linux_path': os.path.join(self.workspace,
                                       'dir2/filepath2.txt'),
        }
        for filepath in args.values():
            normalized_path = os.path.normpath(filepath.replace('\\', os.sep))
            try:
                os.makedirs(os.path.dirname(normalized_path))
            except OSError:
                pass

            with open(normalized_path, 'w') as open_file:
                open_file.write('the contents of this file do not matter.')

        datastack_path = os.path.join(self.workspace, 'archive.invest.tar.gz')
        datastack.build_datastack_archive(args, 'sample_model', datastack_path)

        extraction_path = os.path.join(self.workspace, 'extracted_dir')
        extracted_args = datastack.extract_datastack_archive(datastack_path,
                                                             extraction_path)

        expected_args = {
            'windows_path': os.path.join(extraction_path, 'data',
                                          'filepath1.txt'),
            'linux_path': os.path.join(extraction_path, 'data',
                                        'filepath2.txt'),
        }
        self.maxDiff = None  # show whole exception on failure
        self.assertEqual(extracted_args, expected_args)


class UtilitiesTest(unittest.TestCase):
    def test_print_args(self):
        """Datastacks: verify that we format args correctly."""
        from natcap.invest.datastack import format_args_dict, __version__

        args_dict = {
            'some_arg': [1, 2, 3, 4],
            'foo': 'bar',
        }

        args_string = format_args_dict(args_dict=args_dict,
                                       model_name='test_model')
        expected_string = str(
            'Arguments for InVEST test_model %s:\n'
            'foo      bar\n'
            'some_arg [1, 2, 3, 4]\n') % __version__
        self.assertEqual(args_string, expected_string)
