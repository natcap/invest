# coding=utf-8
"""Assertions for geospatial testing."""

import os
import csv
import logging
import json
import itertools

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy

from .. import geoprocessing
from . import utils


# TOLERANCE is based on the machine epsilon for single-precision floating-point
# decimal numbers.  See https://en.wikipedia.org/wiki/Machine_epsilon for more
# information on machine epsilons.  This is the default allowable relative
# error due to rounding for our assertions.
REL_TOL_DEFAULT = 1e-09
ABS_TOL_DEFAULT = 0.0
LOGGER = logging.getLogger('pygeoprocessing.testing.assertions')


def isclose(val_a, val_b, rel_tol=REL_TOL_DEFAULT, abs_tol=ABS_TOL_DEFAULT):
    """Assert that values are equal to the given tolerance.

    Adapted from the python 3.5 standard library based on the
    specification found in PEP485.

    Parameters:
        val_a (int or float): The first value to test
        val_b (int or float): The second value to test
        rel_tol (int or float): is the relative tolerance - it is the
            maximum allowed difference between a and b, relative to the
            larger absolute value of a or b. For example, to set a
            tolerance of 5%, pass `rel_tol=0.05`. The default tolerance
            is 1e-09, which assures that the two values are the same
            within about 9 decimal digits. rel_tol must be greater than
            zero.
        abs_tol (float): is the minimum absolute tolerance - useful for
            comparisons near zero. abs_tol must be at least zero.

    Returns:
        A boolean.
    """
    return abs(val_a-val_b) <= max(
        rel_tol * max(abs(val_a), abs(val_b)), abs_tol)


def assert_close(
        value_a, value_b, rel_tol=REL_TOL_DEFAULT, abs_tol=ABS_TOL_DEFAULT,
        msg=None):
    """
    Assert equality to an absolute tolerance.

    Parameters:
        value_a (int or float): The first value to test.
        value_b (int or float): The second value to test.
        rel_tol (int or float): The relative numerical tolerance.  If
            the relative difference of these values are less than this value,
            assertion will pass.
        abs_tol (int or float): The absolute numerical tolerance.  If the
            difference of the values being tested are less than this value,
            the assertion will pass.
        msg=None (string or None): The assertion message to use if value_a
            and value_b are not found to be equal out to the target tolerance.

    Returns:
        None.

    Raises:
        AssertionError: Raised when the values are not equal out to the
        desired tolerance.
    """
    if not isclose(value_a, value_b, rel_tol=rel_tol, abs_tol=abs_tol):
        if msg is None:
            msg = (
                "{a} != {b} within rel_tol {rel_tol} or "
                "abs_tol {abs_tol}".format(
                    a=value_a, b=value_b, rel_tol=rel_tol, abs_tol=abs_tol))
        raise AssertionError(msg)


def assert_rasters_equal(
        a_uri, b_uri, rel_tol=REL_TOL_DEFAULT, abs_tol=ABS_TOL_DEFAULT):
    """Assert te equality of rasters a and b out to the given tolerance.

    This assertion method asserts the equality of these raster
    characteristics:
        + Raster height and width

        + The number of layers in the raster

        + Each pixel value, such that the absolute difference between the
            pixels is less than `tolerance`.

        + Projection

    Args:
        a_uri (string): a URI to a GDAL dataset
        b_uri (string): a URI to a GDAL dataset
        rel_tol (int or float): the relative tolerance to which
            values should be asserted.  This is a numerical tolerance,
            not the number of places to which a value should be rounded.  If
            the relative value is below this value then the assert passes.
        abs_tol (int or float): absolute tolerance to which values should be
            asserted.  If absolute difference in values is below this value
            then the assert passes.

    Returns:
        None

    Raises:
        IOError: Raised when one of the input files is not found on disk.

        AssertionError: Raised when the two rasters are found to be not
        equal to each other.

    """
    for uri in [a_uri, b_uri]:
        if not os.path.exists(uri):
            raise IOError('File "%s" not found on disk' % uri)

    a_dataset = gdal.Open(a_uri)
    b_dataset = gdal.Open(b_uri)

    try:
        if a_dataset.RasterXSize != b_dataset.RasterXSize:
            raise AssertionError(
                "x dimensions are different a=%s, b=%s" %
                (a_dataset.RasterXSize, b_dataset.RasterXSize))

        if a_dataset.RasterYSize != b_dataset.RasterYSize:
            raise AssertionError(
                "y dimensions are different a=%s, b=%s" %
                (a_dataset.RasterYSize, b_dataset.RasterYSize))

        if a_dataset.RasterCount != b_dataset.RasterCount:
            raise AssertionError(
                "different number of rasters a=%s, b=%s" %
                ((a_dataset.RasterCount, b_dataset.RasterCount)))

        a_sr = osr.SpatialReference()
        a_sr.ImportFromWkt(a_dataset.GetProjection())

        b_sr = osr.SpatialReference()
        b_sr.ImportFromWkt(b_dataset.GetProjection())

        if bool(a_sr.IsSame(b_sr)) is False:
            raise AssertionError('Projections differ: %s != %s' % (
                a_sr.ExportToPrettyWkt(), b_sr.ExportToPrettyWkt()))

        for band_number in range(1, a_dataset.RasterCount + 1):
            a_band = a_dataset.GetRasterBand(band_number)
            b_band = b_dataset.GetRasterBand(band_number)

            a_blocksize = a_band.GetBlockSize()
            b_blocksize = b_band.GetBlockSize()
            if a_blocksize != b_blocksize:
                raise AssertionError(
                    'Block sizes differ for band %s: %s != %s' % (
                        band_number, a_blocksize, b_blocksize))

            for (_, a_block), (_, b_block) in zip(
                    geoprocessing.iterblocks(a_uri, [band_number]),
                    geoprocessing.iterblocks(b_uri, [band_number])):
                try:
                    numpy.testing.assert_allclose(
                        a_block, b_block, rtol=rel_tol, atol=abs_tol)
                except AssertionError:
                    iterator = numpy.nditer([a_block, b_block],
                                            flags=['multi_index'],
                                            op_flags=['readonly'])
                    while not iterator.finished:
                        col = iterator.multi_index[0]
                        row = iterator.multi_index[1]

                        pixel_a = a_block[iterator.multi_index]
                        pixel_b = b_block[iterator.multi_index]
                        assert_close(
                            pixel_a, pixel_b, rel_tol=rel_tol,
                            abs_tol=abs_tol, msg=(
                                '{a_val} != {b_val} at col {col}, '
                                'row {row} within rel_tol {rel_tol} and '
                                'abs_tol {abs_tol}').format(
                                    a_val=pixel_a, b_val=pixel_b, col=col,
                                    row=row, rel_tol=rel_tol,
                                    abs_tol=abs_tol))
                        iterator.iternext()
    finally:
        gdal.Dataset.__swig_destroy__(a_dataset)
        gdal.Dataset.__swig_destroy__(b_dataset)
        a_dataset = None
        b_dataset = None


def assert_vectors_equal(a_uri, b_uri, field_tolerance):
    """Assert that the vectors at a_uri and b_uri are equal to each other.

    This assertion method asserts the equality of these vector
    characteristics:
        + Number of layers in the vector

        + Number of features in each layer

        + Geometry type of the layer

        + Feature geometry type

        + Number of fields in each feature

        + Name of each field

        + Field values for each feature

        + Projection

    Args:
        a_uri (string): a URI to an OGR vector
        b_uri (string): a URI to an OGR vector
        field_tolerance (int or float): The relative numerical tolerance to
            which field values should be asserted.

    Raises:
        IOError: Raised if one of the input files is not found on disk.
        AssertionError: Raised if the vectors are not found to be equal to
        one another.

    Returns
        None
    """
    for uri in [a_uri, b_uri]:
        if not os.path.exists(uri):
            raise IOError('File "%s" not found on disk' % uri)

    shape = ogr.Open(a_uri)
    shape_regression = ogr.Open(b_uri)

    try:
        # Check that the shapefiles have the same number of layers
        layer_count = shape.GetLayerCount()
        layer_count_regression = shape_regression.GetLayerCount()
        if layer_count != layer_count_regression:
            raise AssertionError(
                'Number of vector layers do not match: %s != %s' % (
                    layer_count, layer_count_regression))

        for layer_num in range(layer_count):
            # Get the current layer
            layer = shape.GetLayer(layer_num)
            layer_regression = shape_regression.GetLayer(layer_num)
            # Check that each layer has the same number of features
            feat_count = layer.GetFeatureCount()
            feat_count_regression = layer_regression.GetFeatureCount()
            if feat_count != feat_count_regression:
                raise AssertionError(
                    ('The layers DO NOT have the same number of features'
                     '%s != %s' % (feat_count, feat_count_regression)))

            layer_geom = layer.GetGeomType()
            layer_reg_geom = layer_regression.GetGeomType()
            if layer_geom != layer_reg_geom:

                def _readable_geometry_type(geom_type):
                    """Determine the geometry type from the OGR wkb int.

                    Parameters:
                        geom_type: The result of ogr.Layer.GetGeomType()

                    Returns:
                        A string representation of the feature type.
                    """
                    for attr_name in dir(ogr):
                        if (attr_name.startswith('wkb') and
                                getattr(ogr, attr_name) == geom_type):
                            return attr_name.replace('wkb', '')

                layer_geom_string = _readable_geometry_type(layer_geom)
                layer_reg_geom_string = _readable_geometry_type(layer_reg_geom)
                raise AssertionError(
                    'Layers #%s have different geometry types: %s != %s' % (
                        layer_num, layer_geom_string, layer_reg_geom_string))

            a_sr = layer.GetSpatialRef()
            b_sr = layer_regression.GetSpatialRef()
            if bool(a_sr.IsSame(b_sr)) is False:
                raise AssertionError(
                    'Projections differ in layer %s: %s != %s' % (
                        layer_num, a_sr.ExportToPrettyWkt(),
                        b_sr.ExportToPrettyWkt()))

            # Get the first features of the layers and loop through all the
            # features
            feat = layer.GetNextFeature()
            feat_regression = layer_regression.GetNextFeature()
            while feat is not None:
                # Check that the field counts for the features are the same
                layer_def = layer.GetLayerDefn()
                layer_def_regression = layer_regression.GetLayerDefn()
                field_count = layer_def.GetFieldCount()
                field_count_regression = layer_def_regression.GetFieldCount()
                if field_count != field_count_regression:
                    raise AssertionError((
                        'The features DO NOT have the same number of fields '
                        'in layer %s: %s != %s') % (
                            layer_num, field_count, field_count_regression))

                for fld_index in range(field_count):
                    # Check that the features have the same field values
                    field = feat.GetField(fld_index)
                    field_regression = feat_regression.GetField(fld_index)

                    try:
                        try:
                            float_field = float(field)
                            float_field_regression = float(field_regression)
                            assert_close(
                                float_field, float_field_regression,
                                field_tolerance)
                        except (ValueError, TypeError):
                            # ValueError happens when strings cast to float
                            # TypeError when casting non-string or non-number.
                            if field != field_regression:
                                raise AssertionError
                    except AssertionError:
                        # If we found an AssertionError, raise a more helpful
                        # assertion error mesage here that includes vector
                        # information.
                        raise AssertionError(
                            'Field values %s != %s at index %s in layer %s' % (
                                field, field_regression, fld_index, layer_num))

                    # Check that the features have the same field name
                    field_ref = feat.GetFieldDefnRef(fld_index)
                    field_ref_regression = \
                        feat_regression.GetFieldDefnRef(fld_index)
                    field_name = field_ref.GetNameRef()
                    field_name_regression = field_ref_regression.GetNameRef()
                    if field_name != field_name_regression:
                        raise AssertionError(
                            'Field names %s != %s at index %s in layer %s' % (
                                field_name, field_name_regression, fld_index,
                                layer_num))

                # Check that the features have the same geometry
                geom = feat.GetGeometryRef()
                geom_regression = feat_regression.GetGeometryRef()

                if bool(geom.Equals(geom_regression)) is False:
                    feature_fid = feat.GetFID()
                    reg_feature_fid = feat_regression.GetFID()
                    raise AssertionError(
                        'Geometries are not equal in feature %s, '
                        'regression feature %s in layer %s' % (
                            feature_fid, reg_feature_fid, layer_num))

                feat = None
                feat_regression = None
                feat = layer.GetNextFeature()
                feat_regression = layer_regression.GetNextFeature()
    finally:
        ogr.DataSource.__swig_destroy__(shape)
        ogr.DataSource.__swig_destroy__(shape_regression)
        shape = None
        shape_regression = None


def assert_csv_equal(
        a_uri, b_uri, rel_tol=REL_TOL_DEFAULT, abs_tol=ABS_TOL_DEFAULT):
    """Assert the equality of CSV files at a_uri and b_uri.

    Tests if csv files a and b are 'almost equal' to each other on a per
    cell basis.  Numeric cells are asserted to be equal within the given
    tolerance.  Other cell types are asserted to be equal.

    Args:
        a_uri (string): a URI to a csv file
        b_uri (string): a URI to a csv file
        rel_tol (int or float): The relative numerical tolerance allowed, if
            relative difference of values are less than this, assertion
            passes.
        abs_tol (int or float): The absolute numerical tolerance allowed, if
            difference of values are less than this, assertion passes.

    Raises:
        AssertionError: Raised when the two CSV files are found to be
        different.

    Returns:
        None
    """
    a_file = open(a_uri, 'rb')
    b_file = open(b_uri, 'rb')

    reader_a = csv.reader(a_file)
    reader_b = csv.reader(b_file)

    for index, (a_row, b_row) in enumerate(zip(reader_a, reader_b)):
        try:
            if a_row != b_row:
                raise AssertionError('Rows differ at row'
                                     '%s: a=%s b=%s' % (index, a_row, b_row))
        except AssertionError:
            for col_index, (a_element, b_element) in enumerate(zip(a_row,
                                                                   b_row)):
                try:
                    a_element = float(a_element)
                    b_element = float(b_element)
                    assert_close(
                        a_element, b_element, rel_tol=rel_tol,
                        abs_tol=abs_tol, msg=(
                            'Values are significantly different at row %s'
                            'col %s: a=%s b=%s' % (
                                index, col_index, a_element, b_element)))
                except ValueError:
                    # we know for sure they arenot floats, so compare as
                    # non-floats.
                    if a_element != b_element:
                        raise AssertionError(
                            'Elements differ at row %s col%s: a=%s '
                            'b=%s' % (index, col_index, a_element, b_element))


def assert_md5_equal(uri, regression_hash):
    """Assert the MD5sum of a file against a regression MD5sum.

    This method is a convenience method that uses
    ``natcap.invest.testing.digest_file()`` to determine the MD5sum of the
    file located at `uri`.  It is functionally equivalent to calling::

        if digest_file(uri) != '<some md5sum>':
            raise AssertionError

    Regression MD5sums can be calculated for you by using
    ``natcap.invest.testing.digest_file()`` or a system-level md5sum program.

    Args:
        uri (string): a string URI to the file to be tested.
        regression_hash (string): a string md5sum to be tested against.

    Raises:
        AssertionError: Raised when the MD5sum of  the file at `uri`
        differs from the provided regression md5sum hash.

    Returns:
        None
    """
    digested_file = utils.digest_file(uri)
    if digested_file != regression_hash:
        raise AssertionError('MD5 hashes differ: %s != %s' % (digested_file,
                                                              regression_hash))


def assert_json_equal(json_1_uri, json_2_uri):
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
        None
    """
    dict_1 = json.loads(open(json_1_uri).read())
    dict_2 = json.loads(open(json_2_uri).read())

    if dict_1 != dict_2:
        raise AssertionError('JSON objects differ: %s\n%s' % (dict_1, dict_2))


def assert_text_equal(text_1_uri, text_2_uri):
    """Assert that two text files are equal.

    This comparison is done line-by-line.

    Args:
        text_1_uri (string): a python string uri to a text file.
            Considered the file to be tested.
        text_2_uri (string): a python string uri to a text file.
            Considered the regression file.

    Raises:
        AssertionError: Raised when a line differs in the two files.

    Returns:
        None
    """
    def lines(filepath):
        """Return a generator of lines in the opened file."""
        with open(filepath, 'rb') as opened_file:
            for line in opened_file:
                yield line

    for index, (a_line, b_line) in enumerate(itertools.izip(
            lines(text_1_uri), lines(text_2_uri))):
        if a_line != b_line:
            raise AssertionError('Line %s in %s does not match regression '
                                 'file %s. Output  "%s" Regression "%s"' % (
                                     index, text_1_uri, text_2_uri, a_line,
                                     b_line))


def assert_checksums_equal(checksum_file, base_folder=None):
    """Assert all files in the `checksum_file` have the same checksum.

    Checksum files could be created by
    `pygeoprocessing.testing.utils.checksum_folder()`, but this function
    may also assert the output of GNU `md5sum` or BSD `md5`.  Either format
    (GNU or BSD) may be provided as input to this assertion function.
    Any files not included in `checksum_file` are ignored.

    Parameters:
        checksum_file (string): the path to the snapshot file to use.
        base_folder=None (string or None): the folder to refer to as the base
            path for files in the `checksum_file`.  If `None`, the current
            working directory will be used.

    Raises:
        AssertionError: when a nonmatching md5sum is found.
    """
    if base_folder is None:
        base_folder = os.getcwd()

    snapshot = open(checksum_file)
    env_params = {}
    files = {}
    for line in snapshot:
        # environment parameters are recorded as comments.
        if line.startswith('#'):
            name, value = line.split('=')
            name = name.replace('#', '').strip()
            env_params[name.strip()] = value.strip()
        else:
            split_line = line.strip().split(' ')
            if split_line[0] == 'MD5':
                # we're reading a BSD-style checksum file with format:
                # MD5 (filename) = md5sum
                md5sum = split_line[3]
                filename = split_line[1].replace(')', '').replace('(', '')
            else:
                # We're reading a GNU-style checksum file with format:
                # md5sum  filename
                md5sum = split_line[0]
                filename = split_line[2]

            files[filename] = md5sum

    missing_files = []
    nonmatching_files = []
    for filepath, expected_md5sum in files.iteritems():
        full_filepath = os.path.join(base_folder, filepath)
        try:
            current_md5sum = utils.digest_file(full_filepath)
        except IOError:
            # When the file we're looking for doesn't exist
            # Keep around for reporting missing files later.
            missing_files.append(full_filepath)
            continue

        if current_md5sum != expected_md5sum:
            nonmatching_files.append(filepath)

    if len(missing_files) != 0:
        if len(missing_files) == len(files):
            raise AssertionError((
                'All files recorded in the snapshot are missing.  Are you '
                'testing against the right folder?  Testing {test_dir}. '
                'Snapshot taken from {snap_dir}.').format(
                    test_dir=base_folder,
                    snap_dir=env_params['orig_workspace']))
        raise AssertionError(
            ('{num_missing} files out of {num_files} are '
             'missing.').format(num_missing=len(missing_files),
                                num_files=len(files)))

    if len(nonmatching_files) != 0:
        raise AssertionError((
            '{num_err} files out of {num_files} have differing '
            'md5sums: {files}').format(num_err=len(nonmatching_files),
                                       num_files=len(files),
                                       files=nonmatching_files))
