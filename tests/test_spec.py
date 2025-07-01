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
            '\tOptions:',
            '\t- A',
            '\t- c: do something else',
            '\t- Option_b: do something'
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
        if spec_data.__class__ is spec.DirectoryOutput:
            os.mkdir(os.path.join(workspace, spec_data.id))
            _generate_files_from_spec(
                spec_data.contents, os.path.join(workspace, spec_data.id))
        else:
            filepath = os.path.join(workspace, spec_data.id)
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
            spec.DirectoryOutput(
                id='output',
                contents=[
                    spec.SingleBandRasterOutput(
                        id="urban_nature_supply_percapita.tif",
                        about="The calculated supply per capita of urban nature.",
                        data_type=float,
                        units=u.m**2
                    ),
                    spec.VectorOutput(
                        id="admin_boundaries.gpkg",
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
                    )
                ]
            ),
            spec.DirectoryOutput(
                id='intermediate',
                contents=[spec.TASKGRAPH_DIR]
            )
        ]
        # Generate an output workspace with real files, without
        # running an invest model.
        _generate_files_from_spec(output_spec, self.workspace_dir)

        model_module = types.SimpleNamespace(
            __name__='urban_nature_access',
            execute=lambda: None,
            MODEL_SPEC=spec.ModelSpec(
                model_id='urban_nature_access',
                model_title='Urban Nature Access',
                userguide='',
                aliases=[],
                input_field_order=[],
                inputs=[],
                outputs=output_spec
            )
        )
        args_dict = {'workspace_dir': self.workspace_dir}
        spec.generate_metadata_for_outputs(model_module, args_dict)

        files, messages = geometamaker.validate_dir(
            self.workspace_dir, recursive=True)
        self.assertEqual(len(files), 2)
        self.assertFalse(any(messages))

        resource = geometamaker.describe(
            os.path.join(args_dict['workspace_dir'], 'output',
                         'urban_nature_supply_percapita.tif'))
        self.assertCountEqual(resource.get_keywords(),
                              [model_module.MODEL_SPEC.model_id, 'InVEST'])
