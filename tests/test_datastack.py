"""Testing Module for Datastack."""
import filecmp
import importlib
import json
import os
import pprint
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy
import pandas
import pygeoprocessing
import shapely.geometry
from osgeo import gdal
from osgeo import ogr


gdal.UseExceptions()
_TEST_FILE_CWD = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_TEST_FILE_CWD,
                        '..', 'data', 'invest-test-data', 'data_stack')
SAMPLE_DATA_DIR = os.path.join(
    _TEST_FILE_CWD, '..', 'data', 'invest-sample-data')

# These modules live in tests/test_datastack_modules
# Each contains a different MODEL_SPEC for the purpose of datastack testing
MOCK_MODEL_ID_TO_PYNAME = {
    name: f'test_datastack_modules.{name}' for name in [
        'archive_extraction',
        'duplicate_filepaths',
        'nonspatial_files',
        'raster',
        'simple_parameters',
        'ui_parameter_archive',
        'vector'
    ]
}

# Allow our tests to import the test modules in the test directory.
sys.path.append(_TEST_FILE_CWD)


class DatastackArchiveTests(unittest.TestCase):
    """Test Datastack Archives."""

    def setUp(self):
        """Create temporary workspace."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary workspace."""
        shutil.rmtree(self.workspace)

    @staticmethod
    def execute_model(workspace, source_parameter_set):
        """Helper function to run a model from its parameter set file.

        Args:
            workspace (str): The path to the workspace to use for the test run.
                All files will be written here.
            source_parameter_set (str): The path to the parameter set from
                which the args dict and model name should be loaded.

        Returns:
            ``None``
        """
        from natcap.invest import datastack

        source_args = datastack.extract_parameter_set(source_parameter_set)
        model_name = source_args.model_name

        datastack_archive_path = os.path.join(
            workspace, 'datastack.invs.tar.gz')
        datastack.build_datastack_archive(
            source_args.args, model_name, datastack_archive_path)

        extraction_dir = os.path.join(workspace, 'archived_data')
        args = datastack.extract_datastack_archive(
            datastack_archive_path, extraction_dir)
        args['workspace_dir'] = os.path.join(workspace, 'workspace')

        # validate the args for good measure
        module = importlib.import_module(name=model_name)
        errors = module.validate(args)
        if errors != []:
            raise AssertionError(
                f"Errors founds: {pprint.pformat(errors)}")

        module.execute(args)

    @unittest.skip('Sample data not usually cloned for test runs.')
    def test_coastal_blue_carbon(self):
        """Datastack: Test CBC."""
        source_parameter_set_path = os.path.join(
            SAMPLE_DATA_DIR, 'CoastalBlueCarbon',
            'cbc_galveston_bay.invs.json')
        DatastackArchiveTests.execute_model(
            self.workspace, source_parameter_set_path)

    @unittest.skip('Sample data not usually cloned for test runs.')
    def test_habitat_quality(self):
        """Datastack: Test Habitat Quality."""
        source_parameter_set_path = os.path.join(
            SAMPLE_DATA_DIR, 'HabitatQuality',
            'habitat_quality_willamette.invs.json')
        DatastackArchiveTests.execute_model(
            self.workspace, source_parameter_set_path)

    @unittest.skip('Sample data not usually cloned for test runs.')
    def test_cv(self):
        """Datastack: Test Coastal Vulnerability."""
        source_parameter_set_path = os.path.join(
            SAMPLE_DATA_DIR, 'CoastalVulnerability',
            'coastal_vuln_grandbahama.invs.json')
        DatastackArchiveTests.execute_model(
            self.workspace, source_parameter_set_path)

    @unittest.skip('Sample data not usually cloned for test runs.')
    def test_recreation(self):
        source_parameter_set_path = os.path.join(
            SAMPLE_DATA_DIR, 'recreation', 'recreation_andros.invs.json')
        DatastackArchiveTests.execute_model(
            self.workspace, source_parameter_set_path)

    def test_collect_simple_parameters(self):
        """Datastack: test collect simple parameters."""
        from natcap.invest import datastack
        params = {
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'd': '',
            'workspace_dir': os.path.join(self.workspace),
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

        with patch('natcap.invest.datastack.models') as p:
            p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
            datastack.build_datastack_archive(
                params, 'simple_parameters', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        datastack._tarfile_safe_extract(archive_path, out_directory)

        self.assertEqual(len(os.listdir(out_directory)), 3)

        # We expect the workspace to be excluded from the resulting args dict.
        with open(os.path.join(
                out_directory, datastack.DATASTACK_PARAMETER_FILENAME)) as file:
            self.assertEqual(
                json.load(file)['args'],
                {'a': 1, 'b': 'hello there', 'c': 'plain bytestring', 'd': ''})

    def test_collect_rasters(self):
        """Datastack: test collect GDAL rasters."""
        from natcap.invest import datastack
        for raster_filename in (
                'dem',  # This is a multipart raster
                'landcover.tif'):  # This is a single-file raster

            params = {
                'raster': os.path.join(DATA_DIR, raster_filename),
            }

            # Collect the raster's files into a single archive
            archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
            with patch('natcap.invest.datastack.models') as p:
                p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
                datastack.build_datastack_archive(
                    params, 'raster', archive_path)

            # extract the archive
            out_directory = os.path.join(self.workspace, 'extracted_archive')
            datastack._tarfile_safe_extract(archive_path, out_directory)

            with open(os.path.join(
                    out_directory,
                    datastack.DATASTACK_PARAMETER_FILENAME)) as datastack_file:
                archived_params = json.load(datastack_file)['args']

            self.assertEqual(len(archived_params), 1)
            model_array = pygeoprocessing.raster_to_numpy_array(
                params['raster'])
            reg_array = pygeoprocessing.raster_to_numpy_array(
                os.path.join(out_directory, archived_params['raster']))
            numpy.testing.assert_allclose(model_array, reg_array)

    def test_collect_vectors(self):
        """Datastack: test collect ogr vector."""
        from natcap.invest import datastack
        from natcap.invest.utils import _assert_vectors_equal
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

            with patch('natcap.invest.datastack.models') as p:
                p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
                # Collect the vector's files into a single archive
                datastack.build_datastack_archive(
                    params, 'vector', archive_path)

            # extract the archive
            out_directory = os.path.join(dest_dir, 'extracted_archive')
            datastack._tarfile_safe_extract(archive_path, out_directory)

            with open(os.path.join(
                    out_directory,
                    datastack.DATASTACK_PARAMETER_FILENAME)) as datastack_file:
                archived_params = json.load(datastack_file)['args']
            _assert_vectors_equal(
                params['vector'],
                os.path.join(out_directory, archived_params['vector']))

            self.assertEqual(len(archived_params), 1)  # sanity check

    def test_datastack_metadata(self):
        """Test correct metadata is created for datastack

        Copy files into a temp directory, create metadata for 1 file 

        """
        from natcap.invest import datastack
        import geometamaker

        params = {
            'raster': os.path.join(DATA_DIR, "landcover.tif"),
            'simple_table': os.path.join(DATA_DIR, "carbon_pools_samp.csv"),
        }

        # Copy params into new dir
        temp_dir = os.path.join(self.workspace, "temp_dir")
        os.mkdir(temp_dir)

        for name, f in params.items():
            shutil.copyfile(f, os.path.join(temp_dir, os.path.basename(f)))

        params = {k: os.path.join(temp_dir, os.path.basename(f))
                  for k, f in params.items()}

        # generate custom metadata for 1 file before building datastack
        resource = geometamaker.describe(params['raster'])
        resource.set_description("foo")
        resource.set_keywords(["bar"])
        resource.write()

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

        with patch('natcap.invest.datastack.models') as p:
            p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
            datastack.build_datastack_archive(
                params, 'archive_extraction', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        datastack._tarfile_safe_extract(archive_path, out_directory)

        # validate metadata in directory to ensure 2 yamls exist
        files, messages = geometamaker.validate_dir(out_directory)
        self.assertEqual(len(files), 2)
        self.assertFalse(any(messages))

        # test that custom description and keyword are not overwritten and new
        # keywords are added
        raster_path = os.path.join(out_directory, "data",
                                   "raster_raster", "landcover.tif")
        resource = geometamaker.describe(raster_path)
        self.assertEqual(resource.get_description(), "foo")
        self.assertCountEqual(resource.get_keywords(),
                              ["archive_extraction_model", "InVEST", "bar"])

    def test_nonspatial_files(self):
        """Datastack: test nonspatial files."""
        from natcap.invest import datastack

        params = {
            'some_file': os.path.join(self.workspace, 'foo.txt'),
            'data_dir': os.path.join(self.workspace, 'data_dir')
        }
        with open(params['some_file'], 'w') as textfile:
            textfile.write('some text here!')

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

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        with patch('natcap.invest.datastack.models') as p:
            p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
            datastack.build_datastack_archive(
                params, 'nonspatial_files', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        datastack._tarfile_safe_extract(archive_path, out_directory)

        with open(os.path.join(
                out_directory,
                datastack.DATASTACK_PARAMETER_FILENAME)) as datastack_file:
            archived_params = json.load(datastack_file)['args']
        self.assertTrue(filecmp.cmp(
            params['some_file'],
            os.path.join(out_directory, archived_params['some_file']),
            shallow=False))
        self.assertEqual(len(archived_params), 2)  # sanity check

        common_files = ['foo.txt', 'bar.txt', 'baz.txt', 'nested/nested.txt']
        matched_files, mismatch_files, error_files = filecmp.cmpfiles(
            params['data_dir'],
            os.path.join(out_directory, archived_params['data_dir']),
            common_files, shallow=False)
        if mismatch_files or error_files:
            self.fail('Directory mismatch or error. The mismatches are'
                      f' {mismatch_files} ; and the errors are {error_files}')

    def test_duplicate_filepaths(self):
        """Datastack: test duplicate filepaths."""
        from natcap.invest import datastack
        params = {
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
        }
        with open(params['foo'], 'w') as textfile:
            textfile.write('hello world!')

        # Collect the file into an archive
        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        with patch('natcap.invest.datastack.models') as p:
            p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
            datastack.build_datastack_archive(
                params, 'duplicate_filepaths', archive_path)

        # extract the archive
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        datastack._tarfile_safe_extract(archive_path, out_directory)

        with open(os.path.join(
                out_directory,
                datastack.DATASTACK_PARAMETER_FILENAME)) as datastack_file:
            archived_params = json.load(datastack_file)['args']

        # Assert that the archived 'foo' and 'bar' params point to the same
        # file.
        self.assertEqual(archived_params['foo'], archived_params['bar'])

        # Assert we have the expected directory contents.
        self.assertEqual(
            sorted(os.listdir(out_directory)),
            ['data', 'log.txt', 'parameters.invest.json'])
        self.assertTrue(os.path.isdir(os.path.join(out_directory, 'data')))

        # Assert we have the expected number of files in the data dir.
        self.assertEqual(
            len(os.listdir(os.path.join(out_directory, 'data'))), 1)

    def test_archive_extraction(self):
        """Datastack: test archive extraction."""
        from natcap.invest import datastack
        from natcap.invest import utils
        from natcap.invest import spec
        from natcap.invest import validation

        params = {
            'blank': '',
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'foo.txt'),
            'data_dir': os.path.join(self.workspace, 'data_dir'),
            'raster': os.path.join(DATA_DIR, 'dem'),
            'vector': os.path.join(DATA_DIR, 'watersheds.shp'),
            'simple_table': os.path.join(DATA_DIR, 'carbon_pools_samp.csv'),
            'spatial_table': os.path.join(self.workspace, 'spatial_table.csv'),
        }
        # synthesize sample data
        os.makedirs(params['data_dir'])
        for filename in ('foo.txt', 'bar.txt', 'baz.txt'):
            data_filepath = os.path.join(params['data_dir'], filename)
            with open(data_filepath, 'w') as textfile:
                textfile.write(filename)

        with open(params['foo'], 'w') as textfile:
            textfile.write('hello world!')

        with open(params['spatial_table'], 'w') as spatial_csv:
            # copy existing DEM
            # copy existing watersheds
            # new raster
            # new vector
            spatial_csv.write('ID,path\n')
            spatial_csv.write(f"1,{params['raster']}\n")
            spatial_csv.write(f"2,{params['vector']}\n")

            # Create a raster only referenced by the CSV
            target_csv_raster_path = os.path.join(
                self.workspace, 'new_raster.tif')
            pygeoprocessing.new_raster_from_base(
                params['raster'], target_csv_raster_path, gdal.GDT_UInt16, [0])
            spatial_csv.write(f'3,{target_csv_raster_path}\n')

            # Create a vector only referenced by the CSV
            target_csv_vector_path = os.path.join(
                self.workspace, 'new_vector.geojson')
            pygeoprocessing.shapely_geometry_to_vector(
                [shapely.geometry.Point(100, 100)],
                target_csv_vector_path,
                pygeoprocessing.get_raster_info(
                    params['raster'])['projection_wkt'],
                'GeoJSON',
                ogr_geom_type=ogr.wkbPoint)
            spatial_csv.write(f'4,{target_csv_vector_path}\n')

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')
        with patch('natcap.invest.datastack.models') as p:
            p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
            datastack.build_datastack_archive(
                params, 'archive_extraction', archive_path)
        out_directory = os.path.join(self.workspace, 'extracted_archive')
        archive_params = datastack.extract_datastack_archive(
            archive_path, out_directory)
        model_array = pygeoprocessing.raster_to_numpy_array(
            archive_params['raster'])
        reg_array = pygeoprocessing.raster_to_numpy_array(params['raster'])
        numpy.testing.assert_allclose(model_array, reg_array)
        utils._assert_vectors_equal(
            archive_params['vector'], params['vector'])
        pandas.testing.assert_frame_equal(
            pandas.read_csv(archive_params['simple_table']),
            pandas.read_csv(params['simple_table']))
        for key in ('blank', 'a', 'b', 'c'):
            self.assertEqual(archive_params[key],
                             params[key],
                             f'Params differ for key {key}')

        for key in ('foo', 'bar'):
            self.assertTrue(
                filecmp.cmp(archive_params[key], params[key], shallow=False))

        spatial_csv_dict = spec.CSVInput(
            id='',
            index_col='id',
            columns=[
                spec.IntegerInput(id='id'),
                spec.FileInput(id='path')]
        ).get_validated_dataframe(
            archive_params['spatial_table']
        ).to_dict(orient='index')
        spatial_csv_dir = os.path.dirname(archive_params['spatial_table'])
        numpy.testing.assert_allclose(
            pygeoprocessing.raster_to_numpy_array(
                os.path.join(spatial_csv_dir, spatial_csv_dict[3]['path'])),
            pygeoprocessing.raster_to_numpy_array(
                target_csv_raster_path))
        utils._assert_vectors_equal(
            os.path.join(spatial_csv_dir, spatial_csv_dict[4]['path']),
            target_csv_vector_path)

    def test_relative_path_failure(self):
        """Datastack: raise error when relative path creation fails."""
        from natcap.invest import datastack
        params = {
            'workspace_dir': os.path.join(self.workspace),
        }

        archive_path = os.path.join(self.workspace, 'archive.invs.tar.gz')

        # Call build_datastack_archive and force build_parameter_set
        # to raise an error
        error_message = 'Error saving datastack'
        with self.assertRaises(ValueError):
            with patch('natcap.invest.datastack.build_parameter_set',
                       side_effect=ValueError(error_message)):
                with patch('natcap.invest.datastack.models') as p:
                    p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
                    datastack.build_datastack_archive(
                        params, 'simple_parameters',archive_path)

    def test_extract_old_style_datastack(self):
        """Datastack: extract old-style datastack that uses pyname"""
        from natcap.invest import datastack
        json_path = os.path.join(self.workspace, 'old_datastack.json')
        with open(json_path, 'w') as file:
            json.dump({
                "args": {
                    "factor": "",
                    "raster_path": "",
                    "results_suffix": "",
                    "workspace_dir": ""
                },
                "invest_version": "3.14.2",
                "model_name": "natcap.invest.carbon"
            }, file)
        datastack_info = datastack.extract_parameter_set(json_path)
        self.assertEqual(datastack_info.model_id, "carbon")


class ParameterSetTest(unittest.TestCase):
    """Test Datastack."""
    def setUp(self):
        """Create temporary workspace."""
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary workspace."""
        shutil.rmtree(self.workspace)

    def test_datastack_parameter_set(self):
        """Datastack: test datastack parameter set."""
        from natcap.invest import __version__
        from natcap.invest import datastack

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
        args, callable_name = datastack.extract_parameter_set(
            paramset_filename)

        # parameter set calculations normalizes all paths.
        # These are relative paths and must be patched.
        normalized_params = params.copy()
        normalized_params['d'] = True  # should be read in as a bool
        for key in ('raster', 'vector', 'table'):
            normalized_params[key] = os.path.normpath(normalized_params[key])

        self.assertEqual(args, normalized_params)
        self.assertEqual(callable_name, modelname)

    def test_relative_parameter_set(self):
        """Datastack: test relative parameter set."""
        from natcap.invest import __version__
        from natcap.invest import datastack

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
            with open(test_filepath, 'w') as file:
                file.write('hello!')
        os.makedirs(params['data_dir'])

        # Write the parameter set
        datastack.build_parameter_set(
            params, modelname, paramset_filename, relative=True)

        # Check that the written parameter set file contains relative paths
        with open(paramset_filename) as param_file:
            raw_args = json.load(param_file)['args']
        self.assertEqual(raw_args['foo'], 'foo.txt')
        self.assertEqual(raw_args['bar'], 'foo.txt')
        self.assertEqual(raw_args['file_list'], ['file1.txt', 'file2.txt'])
        self.assertEqual(raw_args['data_dir'], 'data_dir')
        self.assertEqual(raw_args['temp_workspace'], '.')

        # Read back the parameter set and verify the returned paths are
        # absolute
        args, callable_name = datastack.extract_parameter_set(
            paramset_filename)

        self.assertEqual(args, params)
        self.assertEqual(callable_name, modelname)

    def test_relative_path_failure(self):
        """Datastack: raise error when relative path creation fails."""
        from natcap.invest import datastack

        params = {
            'data_dir': os.path.join(self.workspace, 'data_dir'),
        }
        modelname = 'natcap.invest.foo'
        paramset_filename = os.path.join(self.workspace, 'paramset.json')

        # make the sample data so filepaths are interpreted correctly
        os.makedirs(params['data_dir'])

        # Call build_parameter_set and force it into an error state
        with self.assertRaises(ValueError):
            with patch('natcap.invest.os.path.relpath',
                       side_effect=ValueError):
                datastack.build_parameter_set(
                    params, modelname, paramset_filename, relative=True)

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_relative_parameter_set_windows(self):
        """Datastack: test relative parameter set paths saved linux style."""
        from natcap.invest import __version__
        from natcap.invest import datastack

        params = {
            'foo': os.path.join(self.workspace, 'foo.txt'),
            'bar': os.path.join(self.workspace, 'inter_dir', 'bar.txt'),
            'doh': os.path.join(
                self.workspace, 'inter_dir', 'inter_inter_dir', 'doh.txt'),
            'data_dir': os.path.join(self.workspace, 'data_dir'),
        }
        os.makedirs(
            os.path.join(self.workspace, 'inter_dir', 'inter_inter_dir'))
        modelname = 'natcap.invest.foo'
        paramset_filename = os.path.join(self.workspace, 'paramset.json')

        # make the sample data so filepaths are interpreted correctly
        for base_name in ('foo', 'bar', 'doh'):
            with open(params[base_name], 'w') as file:
                file.write('hello!')
        os.makedirs(params['data_dir'])

        # Write the parameter set
        datastack.build_parameter_set(
            params, modelname, paramset_filename, relative=True)

        # Check that the written parameter set file contains relative paths
        with open(paramset_filename) as param_file:
            raw_args = json.load(param_file)['args']
        self.assertEqual(raw_args['foo'], 'foo.txt')
        # Expecting linux style path separators for Windows
        self.assertEqual(raw_args['bar'], 'inter_dir/bar.txt')
        self.assertEqual(raw_args['doh'], 'inter_dir/inter_inter_dir/doh.txt')
        self.assertEqual(raw_args['data_dir'], 'data_dir')

        # Read back the parameter set and verify the returned paths are
        # absolute
        args, callable_name = datastack.extract_parameter_set(
            paramset_filename)

        self.assertEqual(args, params)
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
            'some_model')

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

        with patch('natcap.invest.datastack.models') as p:
            p.model_id_to_pyname = MOCK_MODEL_ID_TO_PYNAME
            datastack.build_datastack_archive(
                params, 'simple_parameters', archive_path)

        stack_type, stack_info = datastack.get_datastack_info(
            archive_path, extract_path=os.path.join(self.workspace, 'archive'))

        self.assertEqual(stack_type, 'archive')
        self.assertEqual(stack_info, datastack.ParameterSet(
            params, 'simple_parameters'))

    def test_get_datastack_info_parameter_set(self):
        """Datastack: test get datastack info parameter set."""
        import natcap.invest
        from natcap.invest import datastack

        params = {
            'a': 1,
            'b': 'hello there',
            'c': 'plain bytestring',
            'd': '',
        }

        test_module_name = 'simple_parameters'
        json_path = os.path.join(self.workspace, 'archive.invs.json')
        datastack.build_parameter_set(
            params, test_module_name, json_path)

        stack_type, stack_info = datastack.get_datastack_info(json_path)
        self.assertEqual(stack_type, 'json')
        self.assertEqual(
            stack_info,
            datastack.ParameterSet(
                params, test_module_name))

    def test_get_datastack_info_logfile_new_style(self):
        """Datastack: test get datastack info logfile new style."""
        import natcap.invest
        from natcap.invest import datastack
        from natcap.invest import utils
        args = {
            'a': 1,
            'b': 2.7,
            'c': [1, 2, 3.55],
            'd': 'hello, world!',
            'e': False,
        }

        logfile_path = os.path.join(self.workspace, 'logfile.txt')
        with open(logfile_path, 'w') as logfile:
            logfile.write(utils.format_args_dict(args, 'some_modelname'))

        stack_type, stack_info = datastack.get_datastack_info(logfile_path)
        self.assertEqual(stack_type, 'logfile')
        self.assertEqual(stack_info, datastack.ParameterSet(
            args, 'some_modelname'))

    def test_get_datastack_info_logfile_old_style(self):
        """Datastack: test get datastack info logfile old style."""
        import natcap.invest
        from natcap.invest import datastack
        from natcap.invest import utils
        args = {
            'a': 1,
            'b': 2.7,
            'c': [1, 2, 3.55],
            'd': 'hello, world!',
            'e': False,
        }

        logfile_path = os.path.join(self.workspace, 'logfile.txt')
        with open(logfile_path, 'w') as logfile:
            # Old style of log files include the pyname instead of model ID
            logfile.write(utils.format_args_dict(args, 'natcap.invest.carbon'))

        stack_type, stack_info = datastack.get_datastack_info(logfile_path)
        self.assertEqual(stack_type, 'logfile')
        self.assertEqual(stack_info, datastack.ParameterSet(
            args, 'carbon'))

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_mixed_path_separators_in_paramset_windows(self):
        """Datastacks: parameter sets must handle windows and linux paths."""
        from natcap.invest import datastack

        args = {
            'windows_path': os.path.join(
                self.workspace, 'dir1\\filepath1.txt'),
            'linux_path': os.path.join(
                self.workspace, 'dir2/filepath2.txt'),
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
        # Windows paths should be saved with linux-style separators
        datastack.build_parameter_set(
            args, 'sample_model', paramset_path, relative=True)

        with open(paramset_path) as saved_parameters:
            args = json.loads(saved_parameters.read())['args']
            # Expecting window_path to have linux style line seps
            expected_args = {
                'windows_path': 'dir1/filepath1.txt',
                'linux_path': 'dir2/filepath2.txt',
            }
            self.assertEqual(expected_args, args)

        expected_args = {
            'windows_path': os.path.join(
                self.workspace, 'dir1', 'filepath1.txt'),
            'linux_path': os.path.join(
                self.workspace, 'dir2', 'filepath2.txt'),
        }

        extracted_paramset = datastack.extract_parameter_set(paramset_path)
        self.assertEqual(extracted_paramset.args, expected_args)

    @unittest.skipUnless(sys.platform.startswith("darwin"), "requires macOS")
    def test_mixed_path_separators_in_paramset_mac(self):
        """Datastacks: parameter sets must handle mac and linux paths."""
        from natcap.invest import datastack

        args = {
            'mac_path': os.path.join(
                self.workspace, 'dir1/filepath1.txt'),
            'linux_path': os.path.join(
                self.workspace, 'dir2/filepath2.txt'),
        }
        for filepath in args.values():
            try:
                os.makedirs(os.path.dirname(filepath))
            except OSError:
                pass

            with open(filepath, 'w') as open_file:
                open_file.write('the contents of this file do not matter.')

        paramset_path = os.path.join(self.workspace, 'paramset.invest.json')
        datastack.build_parameter_set(
            args, 'sample_model', paramset_path, relative=True)

        with open(paramset_path) as saved_parameters:
            args = json.loads(saved_parameters.read())['args']
            expected_args = {
                'mac_path': 'dir1/filepath1.txt',
                'linux_path': 'dir2/filepath2.txt',
            }
            self.assertEqual(expected_args, args)

        expected_args = {
            'mac_path': os.path.join(
                self.workspace, 'dir1', 'filepath1.txt'),
            'linux_path': os.path.join(
                self.workspace, 'dir2', 'filepath2.txt'),
        }

        extracted_paramset = datastack.extract_parameter_set(paramset_path)
        self.assertEqual(extracted_paramset.args, expected_args)
