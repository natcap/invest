import os
import shutil
import tempfile
import types
import unittest

import geometamaker
from natcap.invest import spec
from natcap.invest.unit_registry import u
from osgeo import gdal
from osgeo import ogr

gdal.UseExceptions()


class SpecUtilsUnitTests(unittest.TestCase):
    """Unit tests for natcap.invest.spec."""

    def test_format_unit(self):
        """spec: test converting units to strings with format_unit."""
        for unit_name, expected in [
                ('meter', 'm'),
                ('meter / second', 'm/s'),
                ('foot * mm', 'ft · mm'),
                ('t * hr * ha / ha / MJ / mm', 't · h · ha / (ha · MJ · mm)'),
                ('mm^3 / year', 'mm³/year')
        ]:
            unit = spec.u.Unit(unit_name)
            actual = spec.format_unit(unit)
            self.assertEqual(expected, actual)

    def test_format_unit_raises_error(self):
        """spec: format_unit raises TypeError if not a pint.Unit."""
        with self.assertRaises(TypeError):
            spec.format_unit({})


class TestDescribeArgFromSpec(unittest.TestCase):
    """Test building RST for various invest args specifications."""

    def test_number_spec(self):
        number_spec = spec.NumberInput(
            id="bar",
            name="Bar",
            about="Description",
            units=u.meter**3/u.month,
            expression="value >= 0"
        )
        out = spec.describe_arg_from_spec(number_spec.name, number_spec)
        expected_rst = ([
            '**Bar** (`number <input_types.html#number>`__, '
            'units: **m³/month**, *required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_ratio_spec(self):
        ratio_spec = spec.RatioInput(
            id="bar",
            name="Bar",
            about="Description"
        )
        out = spec.describe_arg_from_spec(ratio_spec.name, ratio_spec)
        expected_rst = (['**Bar** (`ratio <input_types.html#ratio>`__, '
                         '*required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_percent_spec(self):
        percent_spec = spec.PercentInput(
            id="bar",
            name="Bar",
            about="Description",
            required=False
        )
        out = spec.describe_arg_from_spec(percent_spec.name, percent_spec)
        expected_rst = (['**Bar** (`percent <input_types.html#percent>`__, '
                         '*optional*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_integer_spec(self):
        integer_spec = spec.IntegerInput(
            id="bar",
            name="Bar",
            about="Description",
            required=True
        )
        out = spec.describe_arg_from_spec(integer_spec.name, integer_spec)
        expected_rst = (['**Bar** (`integer <input_types.html#integer>`__, '
                         '*required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_boolean_spec(self):
        boolean_spec = spec.BooleanInput(
            id="bar",
            name="Bar",
            about="Description"
        )
        out = spec.describe_arg_from_spec(boolean_spec.name, boolean_spec)
        expected_rst = (['**Bar** (`true/false <input_types.html#truefalse>'
                         '`__): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_freestyle_string_spec(self):
        string_spec = spec.StringInput(
            id="bar",
            name="Bar",
            about="Description"
        )
        out = spec.describe_arg_from_spec(string_spec.name, string_spec)
        expected_rst = (['**Bar** (`text <input_types.html#text>`__, '
                         '*required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_option_string_spec(self):
        option_spec = spec.OptionStringInput(
            id="bar",
            name="Bar",
            about="Description",
            options=[
                spec.Option(key="option_a", display_name="A"),
                spec.Option(key="Option_b", about="do something"),
                spec.Option(
                    key="option_c",
                    display_name="c",
                    about="do something else")])
        # expect that option case is ignored
        # otherwise, c would sort before A
        out = spec.describe_arg_from_spec(option_spec.name, option_spec)
        expected_rst = ([
            '**Bar** (`option <input_types.html#option>`__, *required*): Description',
            '\tValues must be one of the following text strings:',
            '\t- "**A**"',
            '\t- "**c**": do something else',
            '\t- "**Option_b**": do something'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_raster_spec(self):
        raster_spec = spec.SingleBandRasterInput(
            id="bar",
            data_type=int,
            units=None,
            about="Description",
            name="Bar"
        )
        out = spec.describe_arg_from_spec(raster_spec.name, raster_spec)
        expected_rst = ([
            '**Bar** (`raster <input_types.html#raster>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

        raster_spec = spec.SingleBandRasterInput(
            id="bar",
            data_type=float,
            units=u.millimeter/u.year,
            about="Description",
            name="Bar"
        )
        out = spec.describe_arg_from_spec(raster_spec.name, raster_spec)
        expected_rst = ([
            '**Bar** (`raster <input_types.html#raster>`__, units: **mm/year**, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_vector_spec(self):
        vector_spec = spec.VectorInput(
            id="bar",
            fields=[],
            geometry_types={"LINESTRING"},
            about="Description",
            name="Bar"
        )
        out = spec.describe_arg_from_spec(vector_spec.name, vector_spec)
        expected_rst = ([
            '**Bar** (`vector <input_types.html#vector>`__, linestring, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

        vector_spec = spec.VectorInput(
            id="bar",
            fields=[
                spec.IntegerInput(
                    id="id",
                    about="Unique identifier for each feature"
                ),
                spec.NumberInput(
                    id="precipitation",
                    units=u.millimeter/u.year,
                    about="Average annual precipitation over the area"
                )
            ],
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            about="Description",
            name="Bar"
        )
        out = spec.describe_arg_from_spec(vector_spec.name, vector_spec)
        expected_rst = ([
            '**Bar** (`vector <input_types.html#vector>`__, polygon/multipolygon, *required*): Description',
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_csv_spec(self):
        csv_spec = spec.CSVInput(
            id="bar",
            about="Description.",
            name="Bar"
        )
        out = spec.describe_arg_from_spec(csv_spec.name, csv_spec)
        expected_rst = ([
            '**Bar** (`CSV <input_types.html#csv>`__, *required*): Description. '
            'Please see the sample data table for details on the format.'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

        # Test every type that can be nested in a CSV column:
        # number, ratio, percent, code,
        csv_spec = spec.CSVInput(
            id="bar",
            about="Description",
            name="Bar",
            columns=[
                spec.RatioInput(
                    id="b",
                    about="description"
                )
            ]
        )
        out = spec.describe_arg_from_spec(csv_spec.name, csv_spec)
        expected_rst = ([
            '**Bar** (`CSV <input_types.html#csv>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_directory_spec(self):
        self.maxDiff = None
        dir_spec = spec.DirectoryInput(
            id="bar",
            about="Description",
            name="Bar",
            contents=[]
        )
        out = spec.describe_arg_from_spec(dir_spec.name, dir_spec)
        expected_rst = ([
            '**Bar** (`directory <input_types.html#directory>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_multi_type_spec(self):
        multi_spec = spec.RasterOrVectorInput(
            id="bar",
            about="Description",
            name="Bar",
            data_type=int,
            units=None,
            geometry_types={"POLYGON"},
            fields=[]
        )
        out = spec.describe_arg_from_spec(multi_spec.name, multi_spec)
        expected_rst = ([
            '**Bar** (`raster <input_types.html#raster>`__ or `vector <input_types.html#vector>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_real_model_spec(self):
        from natcap.invest import carbon
        out = spec.describe_arg_from_name(
            'natcap.invest.carbon', 'carbon_pools_path', 'columns', 'lucode')
        expected_rst = (
            '.. _carbon-pools-path-columns-lucode:\n\n' +
            '**lucode** (`integer <input_types.html#integer>`__, *required*): ' +
            carbon.MODEL_SPEC.get_input('carbon_pools_path').get_column('lucode').about
        )
        self.assertEqual(repr(out), repr(expected_rst))


def _generate_files_from_spec(output_spec, workspace):
    """A utility function to support the metadata test."""
    for spec_data in output_spec:
        filepath = os.path.join(workspace, spec_data.path)
        filedir = os.path.dirname(filepath)
        try:
            os.makedirs(filedir)
        except OSError:
            if not os.path.isdir(filedir):
                raise
        if isinstance(spec_data, spec.SingleBandRasterOutput):
            driver = gdal.GetDriverByName('GTIFF')
            raster = driver.Create(filepath, 2, 2, 1, gdal.GDT_Byte)
            band = raster.GetRasterBand(1)
            band.SetNoDataValue(2)
        elif isinstance(spec_data, spec.VectorOutput):
            driver = gdal.GetDriverByName('GPKG')
            target_vector = driver.CreateDataSource(filepath)
            layer_name = os.path.basename(os.path.splitext(filepath)[0])
            target_layer = target_vector.CreateLayer(
                layer_name, geom_type=ogr.wkbPolygon)
            for field_spec in spec_data.fields:
                target_layer.CreateField(ogr.FieldDefn(field_spec.id, ogr.OFTInteger))
        elif isinstance(spec_data, spec.CSVOutput):
            columns = [field_spec.id for field_spec in spec_data.columns]
            with open(filepath, 'w') as file:
                file.write(','.join(columns))
        else:
            # Such as taskgraph.db, just create the file.
            with open(filepath, 'w') as file:
                pass


class TestMetadataFromSpec(unittest.TestCase):
    """Tests for metadata-generation functions."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_write_metadata_for_outputs(self):
        """Test writing metadata for an invest output workspace."""

        # An example invest output spec
        output_spec = [
            spec.SingleBandRasterOutput(
                id="urban_nature_supply_percapita",
                path="output/urban_nature_supply_percapita.tif",
                about="The calculated supply per capita of urban nature.",
                data_type=float,
                units=u.m**2
            ),
            spec.VectorOutput(
                id="admin_boundaries",
                path="output/admin_boundaries.gpkg",
                about=("A copy of the user's administrative boundaries "
                       "vector with a single layer."),
                geometry_types=spec.POLYGONS,
                fields=[
                    spec.NumberOutput(
                        id="SUP_DEMadm_cap",
                        units=u.m**2/u.person,
                        about="The average urban nature supply/demand"
                    )
                ]
            ),
            spec.CSVOutput(
                id="table",
                path="output/table.csv",
                about=("A biophysical table."),
                columns=[
                    spec.NumberOutput(
                        id="foo",
                        units=u.m**2/u.person,
                        about="bar"
                    )
                ]
            ),
            spec.SingleBandRasterOutput(
                id="mask",
                path="intermediate/mask.tif",
                about="A mask for the final raster output.",
                data_type=float,
                units=u.m**2
            ),
            spec.TASKGRAPH_CACHE.model_copy(update=dict(
                path="intermediate/taskgraph_cache/taskgraph.db")
            )
        ]
        # Generate an output workspace with real files, without
        # running an invest model.
        _generate_files_from_spec(output_spec, self.workspace_dir)

        model_spec=spec.ModelSpec(
            model_id='urban_nature_access',
            model_title='Urban Nature Access',
            userguide='',
            aliases=[],
            input_field_order=[],
            inputs=[],
            module_name='',
            outputs=output_spec
        )
        args_dict = {'workspace_dir': self.workspace_dir}
        model_spec.generate_metadata_for_outputs(args_dict)

        files, messages = geometamaker.validate_dir(self.workspace_dir)
        self.assertEqual(len(files), 4)
        self.assertFalse(any(messages))

        resource = geometamaker.describe(
            os.path.join(args_dict['workspace_dir'], 'output',
                         'urban_nature_supply_percapita.tif'))
        self.assertCountEqual(resource.get_keywords(),
                              [model_spec.model_id, 'InVEST'])


class ResultsSuffixTests(unittest.TestCase):
    """Tests for natcap.invest.spec.ResultsSuffixInput."""

    def test_suffix_string(self):
        """Utils: test suffix_string."""
        self.assertEqual(spec.SUFFIX.preprocess('suff'), '_suff')

    def test_suffix_string_underscore(self):
        """Utils: test suffix_string underscore."""
        self.assertEqual(spec.SUFFIX.preprocess('_suff'), '_suff')

    def test_suffix_string_empty(self):
        """Utils: test empty suffix_string."""
        self.assertEqual(spec.SUFFIX.preprocess(''), '')

    def test_suffix_string_no_entry(self):
        """Utils: test no suffix entry in args."""
        self.assertEqual(spec.SUFFIX.preprocess(None), '')


class InputTests(unittest.TestCase):
    """Tests for natcap.invest.spec.Input and subclasses."""

    def test_raster_input_preprocess(self):
        """Test SingleBandRasterInput.preprocess method"""
        raster_input = spec.RasterInput(
            id="foo",
            bands=[spec.RasterBand(units=None)])
        self.assertEqual(raster_input.preprocess('foo/bar.tif'), 'foo/bar.tif')
        self.assertEqual(
            raster_input.preprocess('zip+https://storage.googleapis.com/foo/bar.tif'),
            '/vsizip/vsicurl/https://storage.googleapis.com/foo/bar.tif')
        self.assertEqual(raster_input.preprocess(''), None)
        self.assertEqual(raster_input.preprocess(None), None)

    def test_single_band_raster_input_preprocess(self):
        """Test SingleBandRasterInput.preprocess method"""
        raster_input = spec.SingleBandRasterInput(
            id="foo",
            data_type=int,
            units=None)
        self.assertEqual(raster_input.preprocess('foo/bar.tif'), 'foo/bar.tif')
        self.assertEqual(
            raster_input.preprocess('zip+https://storage.googleapis.com/foo/bar.tif'),
            '/vsizip/vsicurl/https://storage.googleapis.com/foo/bar.tif')
        self.assertEqual(raster_input.preprocess(''), None)
        self.assertEqual(raster_input.preprocess(None), None)

    def test_vector_input_preprocess(self):
        """Test VectorInput.preprocess method"""
        vector_input = spec.VectorInput(
            id="foo",
            geometry_types={"POLYGON"},
            fields=[])
        self.assertEqual(vector_input.preprocess('foo/bar.gpkg'), 'foo/bar.gpkg')
        self.assertEqual(
            vector_input.preprocess('zip+https://storage.googleapis.com/foo/bar.gpkg'),
            '/vsizip/vsicurl/https://storage.googleapis.com/foo/bar.gpkg')
        self.assertEqual(vector_input.preprocess(''), None)
        self.assertEqual(vector_input.preprocess(None), None)

    def test_csv_input_preprocess(self):
        """Test CSVInput.preprocess method"""
        csv_input = spec.CSVInput(
            id="foo")
        self.assertEqual(csv_input.preprocess('foo/bar.csv'), 'foo/bar.csv')
        self.assertEqual(
            csv_input.preprocess('https://storage.googleapis.com/foo/bar.csv'),
            'https://storage.googleapis.com/foo/bar.csv')
        self.assertEqual(csv_input.preprocess(''), None)
        self.assertEqual(csv_input.preprocess(None), None)

    def test_number_input_preprocess(self):
        """Test NumberInput.preprocess method"""
        number_input = spec.NumberInput(id='foo', units=None)
        self.assertEqual(number_input.preprocess(1.5), 1.5)
        self.assertEqual(number_input.preprocess('1.5'), 1.5)
        self.assertEqual(number_input.preprocess(0), 0)
        self.assertEqual(number_input.preprocess(''), None)
        self.assertEqual(number_input.preprocess(None), None)

    def test_integer_input_preprocess(self):
        """Test IntegerInput.preprocess method"""
        integer_input = spec.IntegerInput(id='foo')
        self.assertEqual(integer_input.preprocess(1), 1)
        self.assertEqual(integer_input.preprocess('1'), 1)
        self.assertEqual(integer_input.preprocess(0), 0)
        self.assertEqual(integer_input.preprocess(''), None)
        self.assertEqual(integer_input.preprocess(None), None)

    def test_boolean_input_preprocess(self):
        """Test BooleanInput.preprocess method"""
        boolean_input = spec.BooleanInput(id='foo')
        self.assertEqual(boolean_input.preprocess(False), False)
        self.assertEqual(boolean_input.preprocess(True), True)
        self.assertEqual(boolean_input.preprocess(''), None)
        self.assertEqual(boolean_input.preprocess(None), None)

    def test_string_input_preprocess(self):
        """Test StringInput.preprocess method"""
        string_input = spec.StringInput(id='foo')
        self.assertEqual(string_input.preprocess('foo'), 'foo')
        self.assertEqual(string_input.preprocess(1), '1')
        self.assertEqual(string_input.preprocess(''), None)
        self.assertEqual(string_input.preprocess(None), None)

    def test_option_string_input_preprocess(self):
        """Test StringInput.preprocess method"""
        option_string_input = spec.OptionStringInput(
            id='foo', options=[spec.Option(key='foo'), spec.Option(key='bar')])
        self.assertEqual(option_string_input.preprocess('foo'), 'foo')
        self.assertEqual(option_string_input.preprocess('Foo'), 'foo')
        self.assertEqual(option_string_input.preprocess(''), None)
        self.assertEqual(option_string_input.preprocess(None), None)
