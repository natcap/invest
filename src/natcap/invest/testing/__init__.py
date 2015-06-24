"""The natcap.invest.testing package defines core testing routines and
functionality.

Rationale
---------

While the python standard library's ``unittest`` package provides valuable
resources for testing, GIS applications such as the various InVEST models
output GIS data that require more in-depth testing to verify equality.  For
cases such as this, ``natcap.invest.testing`` provides a ``GISTest`` class that
provides assertions for common data formats.

Writing Tests with ``natcap.invest.testing``
--------------------------------------------

The easiest way to take advantage of the functionality in natcap.invest.testing
is to use the ``GISTest`` class whenever you write a TestCase class for your
model.  Doing so will grant you access to the GIS assertions provided by
``GISTest``.

This example is relatively simplistic, since there will often be many more
assertions you may need to make to be able to test your model
effectively::

    import natcap.invest.testing
    import natcap.invest.example_model

    class ExampleTest(natcap.invest.testing.GISTest):
        def test_some_model(self):
            example_args = {
                'workspace_dir': './workspace',
                'arg_1': 'foo',
                'arg_2': 'bar',
            }
            natcap.invest.example_model.execute(example_args)

            # example GISTest assertion
            self.assertRastersEqual('workspace/raster_1.tif',
                'regression_data/raster_1.tif')


.. Writing tests programmatically
.. ------------------------------
..
.. The testing package also provides a program to create regression archives of
.. your content with relative ease.
..
.. .. warning::
..
..     **WRITE YOUR TESTS BY HAND**
..
..     This tool was built using the old paradigm of creating regression data
..     archives for all inputs and outputs.  This was a very expensive approach to
..     testing, since we sometimes output very large datasets, which were all
..     tracked in mercurial.
..
..     **This tool should not be used** until we figure out the correct project-based
..     way to test all of our inputs and outputs.
..
..
.. The tool can be invoked from the command-line like so::
..
..     $ python regression-build.py
..
.. Command-line arguments::
..
..     usage: regression-build.py [-h] [-a] [-i] [-o] [-t] [-c] [-f] [-n]
..
..     optional arguments:
..       -h, --help            show this help message and exit
..       -a , --arguments      JSON file with input arguments and model data
..       -i , --input-archive  Path to where the archived input data will be
..                             saved
..       -o , --output-archive Path to where the archived output data will be
..                             saved
..       -t , --test-file      The test file to modify
..       -c , --test-class     The test class to write or append to. A new
..                             class will be written if this name does not
..                             already exist.
..       -f , --test-func      The test function to write inside the
..                             designated test class.
..       -n, --no-confirm      Provide this flag if you do not wish to
..                             confirm before running.




"""

import csv
import filecmp
import functools
import glob
import hashlib
import json
import logging
import os
import shutil
import time
import unittest

import numpy
np = numpy
from osgeo import gdal
from osgeo import ogr


from natcap.invest.iui import executor
from natcap.invest.iui import fileio
import pygeoprocessing.geoprocessing
import data_storage

LOGGER = logging.getLogger('natcap.invest.testing')

def get_hash(uri):
    """Get the MD5 hash for a single file.  The file is read in a
        memory-efficient fashion.

        Args:
            uri (string): a string uri to the file to be tested.

        Returns:
            An md5sum of the input file"""

    block_size = 2**20
    file_handler = open(uri, 'rb')
    md5 = hashlib.md5()
    while True:
        data = file_handler.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()


def save_workspace(new_workspace):
    """Decorator to save a workspace to a new location.

        If `new_workspace` already exists on disk, it will be recursively
        removed.

        Example usage with a test case::

            import natcap.invest.testing

            @natcap.invest.testing.save_workspace('/path/to/workspace')
            def test_workspaces(self):
                model.execute(self.args)

        Note:
            + Target workspace folder must be saved to ``self.workspace_dir``
                This decorator is only designed to work with test functions
                from subclasses of ``unittest.TestCase`` such as
                ``natcap.invest.testing.GISTest``.

            + If ``new_workspace`` exists, it will be removed.
                So be careful where you save things.

        Args:
            new_workspace (string): a URI to the where the workspace should be
                copied.

        Returns:
            A composed test case function which will execute and then save your
            workspace to the specified location."""

    # item is the function being decorated
    def test_inner_func(item):

        # this decorator indicates that this innermost function is wrapping up
        # the function passed in as item.
        @functools.wraps(item)
        def test_and_remove_workspace(self, *args, **kwargs):
            # This inner function actually executes the test function and then
            # moves the workspace to the folder passed in by the user.
            item(self)

            # remove the contents of the old folder
            try:
                shutil.rmtree(new_workspace)
            except OSError:
                pass

            # copy the workspace to the target folder
            old_workspace = self.workspace_dir
            shutil.copytree(old_workspace, new_workspace)
        return test_and_remove_workspace
    return test_inner_func


def regression(input_archive, workspace_archive):
    """Decorator to unzip input data, run the regression test and compare the
        outputs against the outputs on file.

        Example usage with a test case::

            import natcap.invest.testing

            @natcap.invest.testing.regression('/data/input.tar.gz', /data/output.tar.gz')
            def test_workspaces(self):
                model.execute(self.args)

        Args:
            input_archive (string): The path to a .tar.gz archive with the input data.
            workspace_archive (string): The path to a .tar.gz archive with the workspace to
                assert.

        Returns:
            Composed function with regression testing.
         """

    # item is the function being decorated
    def test_inner_function(item):

        @functools.wraps(item)
        def test_and_assert_workspace(self, *args, **kwargs):
            workspace = pygeoprocessing.geoprocessing.temporary_folder()
            self.args = data_storage.extract_parameters_archive(workspace, input_archive)

            # Actually run the test.  Assumes that self.args is used as the
            # input arguments.
            item(self)

            # Extract the archived workspace to a new temporary folder and
            # compare the two workspaces.
            archived_workspace = pygeoprocessing.geoprocessing.temporary_folder()
            data_storage.extract_archive(archived_workspace, workspace_archive)
            self.assertWorkspace(workspace, archived_workspace)
        return test_and_assert_workspace
    return test_inner_function


def build_regression_archives(file_uri, input_archive_uri, output_archive_uri):
    """Build regression archives for a target model run.

        With a properly formatted JSON configuration file at `file_uri`, all
        input files and parameters are collected and compressed into a single
        gzip.  Then, the target model is executed and the output workspace is
        zipped up into another gzip.  These could then be used for regression
        testing, such as with the ``natcap.invest.testing.regression`` decorator.

        Example configuration file contents (serialized to JSON)::

            {
                    "model": "natcap.invest.pollination.pollination",
                    "arguments": {
                        # the full set of model arguments here
                    }
            }

        Example function usage::

            import natcap.invest.testing

            file_uri = "/path/to/config.json"
            input_archive_uri = "/path/to/archived_inputs.tar.gz"
            output_archive_uri = "/path/to/archived_outputs.tar.gz"
            natcap.invest.testing.build_regression_archives(file_uri,
                input_archive_uri, output_archive_uri)

        Args:
            file_uri (string): a URI to a json file on disk containing the
            above configuration options.

            input_archive_uri (string): the URI to where the gzip archive
            of inputs should be saved once it is created.

            output_archive_uri (string): the URI to where the gzip output
            archive of output should be saved once it is created.

        Returns:
            Nothing.
        """
    file_handler = fileio.JSONHandler(file_uri)

    saved_data = file_handler.get_attributes()

    arguments = saved_data['arguments']
    model_id = saved_data['model']

    model_list = model_id.split('.')
    model = executor.locate_module(model_list)

    # guarantee that we're running this in a new workspace
    arguments['workspace_dir'] = pygeoprocessing.geoprocessing.temporary_folder()
    workspace = arguments['workspace_dir']

    # collect the parameters into a single folder
    input_archive = input_archive_uri
    if input_archive[-7:] == '.tar.gz':
        input_archive = input_archive[:-7]
    data_storage.collect_parameters(arguments, input_archive)
    input_archive += '.tar.gz'

    model_args = data_storage.extract_parameters_archive(workspace, input_archive)

    model.execute(model_args)

    archive_uri = output_archive_uri
    if archive_uri[-7:] == '.tar.gz':
        archive_uri = archive_uri[:-7]
    LOGGER.debug('Archiving the output workspace')
    shutil.make_archive(archive_uri, 'gztar', root_dir=workspace, logger=LOGGER)


class GISTest(unittest.TestCase):
    """A test class with an emphasis on testing GIS outputs.

    The ``GISTest`` class provides many functions for asserting the equality of
    various GIS files.  This is particularly useful for GIS tool outputs, when
    we wish to assert the accuracy of very detailed outputs.

    ``GISTest`` is a subclass of ``unittest.TestCase``, so all members that
    exist in ``unittest.TestCase`` also exist here.  Read the python
    documentation on ``unittest`` for more information about these test
    fixtures and their usage.  The important thing to note is that ``GISTest``
    merely provides more assertions for the more specialized testing and
    assertions that GIS outputs require.

    Example usage of ``GISTest``::

        import natcap.invest.testing

        class ModelTest(natcap.invest.testing.GISTest):
            def test_some_function(self):
                # perform your tests here.


    Note that to take advantage of these additional assertions, you need only
    to create a subclass of ``GISTest`` in your test file to gain access to the
    ``GISTest`` assertions.
    """

    def assertRastersEqual(self, a_uri, b_uri):
        """Tests if datasets a and b are 'almost equal' to each other on a per
        pixel basis

        This assertion method asserts the equality of these raster
        characteristics:
            + Raster height and width

            + The number of layers in the raster

            + Each pixel value, out to a precision of 7 decimal places if the\
            pixel value is a float.


        Args:
            a_uri (string): a URI to a GDAL dataset
            b_uri (string): a URI to a GDAL dataset

        Returns:
            Nothing.

        Raises:
            IOError: Raised when one of the input files is not found on disk.

            AssertionError: Raised when the two rasters are found to be not\
            equal to each other.

        """

        LOGGER.debug('Asserting datasets A: %s, B: %s', a_uri, b_uri)

        for uri in [a_uri, b_uri]:
            if not os.path.exists(uri):
                raise IOError('File "%s" not found on disk' % uri)

        a_dataset = gdal.Open(a_uri)
        b_dataset = gdal.Open(b_uri)

        self.assertEqual(a_dataset.RasterXSize, b_dataset.RasterXSize,
            "x dimensions are different a=%s, second=%s" %
            (a_dataset.RasterXSize, b_dataset.RasterXSize))
        self.assertEqual(a_dataset.RasterYSize, b_dataset.RasterYSize,
            "y dimensions are different a=%s, second=%s" %
            (a_dataset.RasterYSize, b_dataset.RasterYSize))
        self.assertEqual(a_dataset.RasterCount, b_dataset.RasterCount,
            "different number of rasters a=%s, b=%s" % (
            (a_dataset.RasterCount, b_dataset.RasterCount)))

        for band_number in range(1, a_dataset.RasterCount + 1):
            band_a = a_dataset.GetRasterBand(band_number)
            band_b = b_dataset.GetRasterBand(band_number)

            a_array = band_a.ReadAsArray(0, 0, band_a.XSize, band_a.YSize)
            b_array = band_b.ReadAsArray(0, 0, band_b.XSize, band_b.YSize)

            try:
                numpy.testing.assert_array_almost_equal(a_array, b_array)
            except AssertionError:
                for row_index in xrange(band_a.YSize):
                    for pixel_a, pixel_b in zip(a_array[row_index], b_array[row_index]):
                        self.assertAlmostEqual(pixel_a, pixel_b,
                            msg='%s != %s ... Failed at row %s' %
                            (pixel_a, pixel_b, row_index))

    def assertVectorsEqual(self, aUri, bUri):
        """
        Tests if vector datasources are equal to each other.

        This assertion method asserts the equality of these vector
        characteristics:
            + Number of layers in the vector

            + Number of features in each layer

            + Feature geometry type

            + Number of fields in each feature

            + Name of each field

            + Field values for each feature

        Args:
            aUri (string): a URI to an OGR vector
            bUri (string): a URI to an OGR vector

        Raises:
            IOError: Raised if one of the input files is not found on disk.
            AssertionError: Raised if the vectors are not found to be equal to\
            one another.

        Returns
           Nothing.
        """

        for uri in [aUri, bUri]:
            if not os.path.exists(uri):
                raise IOError('File "%s" not found on disk' % uri)

        shape = ogr.Open(aUri)
        shape_regression = ogr.Open(bUri)

        # Check that the shapefiles have the same number of layers
        layer_count = shape.GetLayerCount()
        layer_count_regression = shape_regression.GetLayerCount()
        self.assertEqual(layer_count, layer_count_regression,
                         'The shapes DO NOT have the same number of layers')

        for layer_num in range(layer_count):
            # Get the current layer
            layer = shape.GetLayer(layer_num)
            layer_regression = shape_regression.GetLayer(layer_num)
            # Check that each layer has the same number of features
            feat_count = layer.GetFeatureCount()
            feat_count_regression = layer_regression.GetFeatureCount()
            self.assertEqual(feat_count, feat_count_regression,
                             'The layers DO NOT have the same number of features')

            self.assertEqual(layer.GetGeomType(), layer_regression.GetGeomType(),
                'The layers do not have the same geometry type')


            # Get the first features of the layers and loop through all the features
            feat = layer.GetNextFeature()
            feat_regression = layer_regression.GetNextFeature()
            while feat is not None:
                # Check that the field counts for the features are the same
                layer_def = layer.GetLayerDefn()
                layer_def_regression = layer_regression.GetLayerDefn()
                field_count = layer_def.GetFieldCount()
                field_count_regression = layer_def_regression.GetFieldCount()
                self.assertEqual(field_count, field_count_regression,
                                 'The shapes DO NOT have the same number of fields')

                for fld_index in range(field_count):
                    # Check that the features have the same field values
                    field = feat.GetField(fld_index)
                    field_regression = feat_regression.GetField(fld_index)
                    self.assertEqual(field, field_regression,
                                         'The field values DO NOT match')
                    # Check that the features have the same field name
                    field_ref = feat.GetFieldDefnRef(fld_index)
                    field_ref_regression = \
                        feat_regression.GetFieldDefnRef(fld_index)
                    field_name = field_ref.GetNameRef()
                    field_name_regression = field_ref_regression.GetNameRef()
                    self.assertEqual(field_name, field_name_regression,
                                         'The fields DO NOT have the same name')
                # Check that the features have the same geometry
                geom = feat.GetGeometryRef()
                geom_regression = feat_regression.GetGeometryRef()

                self.assertTrue(geom.Equals(geom_regression))

                if layer.GetGeomType() != ogr.wkbPoint:
                    # Check that the features have the same area,
                    # but only if the shapefile's geometry is not a point, since
                    # points don't have area to check.
                    self.assertEqual(geom.Area(), geom_regression.Area())

                feat = None
                feat_regression = None
                feat = layer.GetNextFeature()
                feat_regression = layer_regression.GetNextFeature()

        shape = None
        shape_regression = None

    def assertCSVEqual(self, aUri, bUri):
        """Tests if csv files a and b are 'almost equal' to each other on a per
        cell basis.  Numeric cells are asserted to be equal out to 7 decimal
        places.  Other cell types are asserted to be equal.

        Args:
            aUri (string): a URI to a csv file
            bUri (string): a URI to a csv file

        Raises:
            AssertionError: Raised when the two CSV files are found to be\
            different.

        Returns:
            Nothing.
        """

        a = open(aUri)
        b = open(bUri)

        reader_a = csv.reader(a)
        reader_b = csv.reader(b)

        for index, (a_row, b_row) in enumerate(zip(reader_a, reader_b)):
            try:
                self.assertEqual(a_row, b_row,
                    'Rows differ at row %s: a=%s b=%s' % (index, a_row, b_row))
            except AssertionError:
                for col_index, (a_element, b_element) in enumerate(zip(a_row, b_row)):
                    try:
                        a_element = float(a_element)
                        b_element = float(b_element)
                        self.assertAlmostEqual(a_element, b_element,
                            msg=('Values are significantly different at row %s col %s:'
                             ' a=%s b=%s' % (index, col_index, a_element,
                             b_element)))
                    except ValueError:
                        # we know for sure they arenot floats, so compare as
                        # non-floats.
                        self.assertEqual(a_element, b_element,
                            msg=('Elements differ at row %s col%s: a=%s b=%s' %
                            (index, col_index, a_element, b_element)))

    def assertMD5(self, uri, regression_hash):
        """Assert the MD5sum of a file against a regression MD5sum.

        This method is a convenience method that uses
        ``natcap.invest.testing.get_hash()`` to determine the MD5sum of the
        file located at `uri`.  It is functionally equivalent to calling::

            self.assertEqual(get_hash(uri), '<some md5sum>')

        Regression MD5sums can be calculated for you by using
        ``natcap.invest.testing.get_hash()`` or a system-level md5sum program.

        Args:
            uri (string): a string URI to the file to be tested.
            regression_hash (string) a string md5sum to be tested against.

        Raises:
            AssertionError: Raised when the MD5sum of  the file at `uri` \
            differs from the provided regression md5sum hash.

        Returns:
            Nothing.
        """

        self.assertEqual(get_hash(uri), regression_hash, "MD5 Hashes differ.")

    def assertMatrixes(self, matrix_a, matrix_b, decimal=6):
        """Tests if the input numpy matrices are equal up to `decimal` places.

        This is a convenience function that wraps up required functionality in
        ``numpy.testing``.

        Args:
            matrix_a (numpy.ndarray): a numpy matrix
            matrix_b (numpy.ndarray): a numpy matrix
            decimal (int): an integer of the desired precision.

        Raises:
            AssertionError: Raised when the two matrices are determined to be\
            different.

        Returns:
            Nothing.
        """

        numpy.testing.assert_array_almost_equal(matrix_a, matrix_b, decimal)

    def assertArchives(self, archive_1_uri, archive_2_uri):
        """
        Compare the contents of two archived workspaces against each other.

        Takes two archived workspaces, each generated from
        ``build_regression_archives()``, unzips them and
        compares the resulting workspaces against each other.

        Args:
            archive_1_uri (string): a URI to a .tar.gz workspace archive
            archive_2_uri (string): a URI to a .tar.gz workspace archive

        Raises:
            AssertionError: Raised when the two workspaces are found to be\
            different.

        Returns:
            Nothing.

            """

        archive_1_folder = pygeoprocessing.geoprocessing.temporary_folder()
        data_storage.extract_archive(archive_1_folder, archive_1_uri)

        archive_2_folder = pygeoprocessing.geoprocessing.temporary_folder()
        data_storage.extract_archive(archive_2_folder, archive_2_uri)

        self.assertWorkspace(archive_1_folder, archive_2_folder)

    def assertWorkspace(self, archive_1_folder, archive_2_folder,
            glob_exclude=''):
        """
        Check the contents of two folders against each other.

        This method iterates through the contents of each workspace folder and
        verifies that all files exist in both folders.  If this passes, then
        each file is compared against each other using
        ``GISTest.assertFiles()``.

        If one of these workspaces includes files that are known to be
        different between model runs (such as logs, or other files that include
        timestamps), you may wish to specify a glob pattern matching those
        filenames and passing it to `glob_exclude`.

        Args:
            archive_1_folder (string): a uri to a folder on disk
            archive_2_folder (string): a uri to a folder on disk
            glob_exclude (string): a string in glob format representing files to ignore

        Raises:
            AssertionError: Raised when the two folders are found to have\
            different contents.

        Returns:
            Nothing.
        """

        # uncompress the two archives

        archive_1_files = []
        archive_2_files = []
        for files_list, workspace in [
                (archive_1_files, archive_1_folder),
                (archive_2_files, archive_2_folder)]:
            for root, dirs, files in os.walk(workspace):
                root = root.replace(workspace + os.sep, '')
                ignored_files = glob.glob(glob_exclude)
                for filename in files:
                    if filename not in ignored_files:
                        full_path = os.path.join(root, filename)
                        files_list.append(full_path)

        archive_1_files = sorted(archive_1_files)
        archive_2_files = sorted(archive_2_files)

        archive_1_size = len(archive_1_files)
        archive_2_size = len(archive_2_files)
        if archive_1_size != archive_2_size:
            # find out which archive had more files.
            archive_1_files = map(lambda x: x.replace(archive_1_folder, ''),
                archive_1_files)
            archive_2_files = map(lambda x: x.replace(archive_2_folder, ''),
                archive_2_files)
            missing_from_archive_1 = list(set(archive_2_files) -
                set(archive_1_files))
            missing_from_archive_2 = list(set(archive_1_files) -
                set(archive_2_files))
            raise AssertionError('Elements missing from A:%s, from B:%s' %
                (missing_from_archive_1, missing_from_archive_2))
        else:
            # archives have the same number of files that we care about
            for file_1, file_2 in zip(archive_1_files, archive_2_files):
                file_1_uri = os.path.join(archive_1_folder, file_1)
                file_2_uri = os.path.join(archive_2_folder, file_2)
                LOGGER.debug('Checking %s, %s', file_1, file_2)
                self.assertFiles(file_1_uri, file_2_uri)

    def assertJSON(self, json_1_uri, json_2_uri):
        """Assert two JSON files against each other.

        The two JSON files provided will be opened, read, and their
        contents will be asserted to be equal.  If the two are found to be
        different, the diff of the two files will be printed.

        Args:
            json_1_uri (string): a uri to a JSON file.
            json_2_uri (string): a uri to a JSON file.

        Raises:
            AssertionError: Raised when the two JSON objects differ.

        Returns:
            Nothing.
        """

        dict_1 = json.loads(open(json_1_uri).read())
        dict_2 = json.loads(open(json_2_uri).read())

        self.maxDiff = None
        self.assertEqual(dict_1, dict_2)

    def assertTextEqual(self, text_1_uri, text_2_uri):
        """Assert that two text files are equal

        This comparison is done line-by-line.

        Args:
            text_1_uri (string): a python string uri to a text file. \
                Considered the file to be tested.
            text_2_uri (string): a python string uri to a text file. \
                Considered the regression file.

        Raises:
            AssertionError: Raised when a line differs in the two files.

        Returns:
            Nothing.
        """

        lines = lambda f: [line for line in open(f)]
        for index, (a_line, b_line) in enumerate(zip(lines(text_1_uri), lines(text_2_uri))):
            self.assertEqual(a_line, b_line, ('Line %s in %s does not match'
                'regression file. Output: \"%s\" Regression: \"%s\"') % (index,
                text_1_uri, a_line, b_line))

    def assertFiles(self, file_1_uri, file_2_uri):
        """Assert two files are equal.

        If the extension of the provided file is recognized, the relevant
        filetype-specific function is called and a more detailed check of the
        file can be done.  If the extension is not recognized, the MD5sums of
        the two files are compared instead.

        Known extensions: ``.json``, ``.tif``, ``.shp``, ``.csv``, ``.txt.``,
        ``.html``

        Args:
            file_1_uri (string): a string URI to a file on disk.
            file_2_uru (string): a string URI to a file on disk.

        Raises:
            AssertionError: Raised when one of the input files does not exist,\
            when the extensions of the input files differ, or if the two files\
            are found to differ.

        Returns:
            Nothing.
        """

        for uri in [file_1_uri, file_2_uri]:
            self.assertEqual(os.path.exists(uri), True,
                'File %s does not exist' % uri)

        # assert the extensions are the same
        file_1_ext = os.path.splitext(file_1_uri)[1]
        file_2_ext = os.path.splitext(file_2_uri)[1]
        self.assertEqual(file_1_ext, file_2_ext, 'Extensions differ: %s, %s' %
            (file_1_ext, file_2_ext))

        assert_funcs = {
            '.json': self.assertJSON,
            '.tif': self.assertRastersEqual,
            '.shp': self.assertVectorsEqual,
            '.csv': self.assertCSVEqual,
            '.txt': self.assertTextEqual,
            '.html': self.assertTextEqual,
        }

        try:
            assert_funcs[file_1_ext](file_1_uri, file_2_uri)
        except KeyError:
            # When we're given an extension we don't have a function for, assert
            # the MD5s.
            file_1_md5 = get_hash(file_1_uri)
            file_2_md5 = get_hash(file_2_uri)
            self.assertEqual(file_1_md5, file_2_md5,
                'Files %s and %s differ (MD5sum)' % (file_1_uri, file_2_uri))

