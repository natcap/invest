import os
import shutil
import tempfile
import types
import unittest

import geometamaker
from natcap.invest import spec_utils
from natcap.invest.unit_registry import u
from osgeo import gdal
from osgeo import ogr

gdal.UseExceptions()


class SpecUtilsUnitTests(unittest.TestCase):
    """Unit tests for natcap.invest.spec_utils."""

    def test_format_unit(self):
        """spec_utils: test converting units to strings with format_unit."""
        from natcap.invest import spec_utils
        for unit_name, expected in [
                ('meter', 'm'),
                ('meter / second', 'm/s'),
                ('foot * mm', 'ft · mm'),
                ('t * hr * ha / ha / MJ / mm', 't · h · ha / (ha · MJ · mm)'),
                ('mm^3 / year', 'mm³/year')
        ]:
            unit = spec_utils.u.Unit(unit_name)
            actual = spec_utils.format_unit(unit)
            self.assertEqual(expected, actual)

    def test_format_unit_raises_error(self):
        """spec_utils: format_unit raises TypeError if not a pint.Unit."""
        from natcap.invest import spec_utils
        with self.assertRaises(TypeError):
            spec_utils.format_unit({})


class TestDescribeArgFromSpec(unittest.TestCase):
    """Test building RST for various invest args specifications."""

    def test_number_spec(self):
        spec = spec_utils.NumberInput(
            name="Bar",
            about="Description",
            units=u.meter**3/u.month,
            expression="value >= 0"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`number <input_types.html#number>`__, '
            'units: **m³/month**, *required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_ratio_spec(self):
        spec = spec_utils.RatioInput(
            name="Bar",
            about="Description"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = (['**Bar** (`ratio <input_types.html#ratio>`__, '
                         '*required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_percent_spec(self):
        spec = spec_utils.PercentInput(
            name="Bar",
            about="Description",
            required=False
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = (['**Bar** (`percent <input_types.html#percent>`__, '
                         '*optional*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_integer_spec(self):
        spec = spec_utils.IntegerInput(
            name="Bar",
            about="Description",
            required=True
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = (['**Bar** (`integer <input_types.html#integer>`__, '
                         '*required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_boolean_spec(self):
        spec = spec_utils.BooleanInput(
            name="Bar",
            about="Description"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = (['**Bar** (`true/false <input_types.html#truefalse>'
                         '`__): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_freestyle_string_spec(self):
        spec = spec_utils.StringInput(
            name="Bar",
            about="Description"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = (['**Bar** (`text <input_types.html#text>`__, '
                         '*required*): Description'])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_option_string_spec_dictionary(self):
        spec = spec_utils.OptionStringInput(
            name="Bar",
            about="Description",
            options={
                "option_a": {
                    "display_name": "A"
                },
                "Option_b": {
                    "description": "do something"
                },
                "option_c": {
                    "display_name": "c",
                    "description": "do something else"
                }
            }
        )
        # expect that option case is ignored
        # otherwise, c would sort before A
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`option <input_types.html#option>`__, *required*): Description',
            '\tOptions:',
            '\t- A',
            '\t- c: do something else',
            '\t- Option_b: do something'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_option_string_spec_list(self):
        spec = spec_utils.OptionStringInput(
            name="Bar",
            about="Description",
            options=["option_a", "Option_b"]
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`option <input_types.html#option>`__, *required*): Description',
            '\tOptions: option_a, Option_b'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_raster_spec(self):
        spec = spec_utils.SingleBandRasterInput(
            band=spec_utils.IntegerInput(),
            about="Description",
            name="Bar"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`raster <input_types.html#raster>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

        spec = spec_utils.SingleBandRasterInput(
            band=spec_utils.NumberInput(units=u.millimeter/u.year),
            about="Description",
            name="Bar"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`raster <input_types.html#raster>`__, units: **mm/year**, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_vector_spec(self):
        spec = spec_utils.VectorInput(
            fields={},
            geometries={"LINESTRING"},
            about="Description",
            name="Bar"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`vector <input_types.html#vector>`__, linestring, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

        spec = spec_utils.VectorInput(
            fields=spec_utils.Fields(
                spec_utils.IntegerInput(
                    id="id",
                    about="Unique identifier for each feature"
                ),
                spec_utils.NumberInput(
                    id="precipitation",
                    units=u.millimeter/u.year,
                    about="Average annual precipitation over the area"
                )
            ),
            geometries={"POLYGON", "MULTIPOLYGON"},
            about="Description",
            name="Bar"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`vector <input_types.html#vector>`__, polygon/multipolygon, *required*): Description',
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_csv_spec(self):
        spec = spec_utils.CSVInput(
            about="Description.",
            name="Bar"
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`CSV <input_types.html#csv>`__, *required*): Description. '
            'Please see the sample data table for details on the format.'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

        # Test every type that can be nested in a CSV column:
        # number, ratio, percent, code,
        spec = spec_utils.CSVInput(
            about="Description",
            name="Bar",
            columns=spec_utils.Columns(
                spec_utils.RatioInput(
                    id="b",
                    about="description"
                )
            )
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`CSV <input_types.html#csv>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_directory_spec(self):
        self.maxDiff = None
        spec = spec_utils.DirectoryInput(
            about="Description",
            name="Bar",
            contents={}
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`directory <input_types.html#directory>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_multi_type_spec(self):
        spec = spec_utils.RasterOrVectorInput(
            about="Description",
            name="Bar",
            band=spec_utils.IntegerInput(),
            geometries={"POLYGON"},
            fields={}
        )
        out = spec_utils.describe_arg_from_spec(spec.name, spec)
        expected_rst = ([
            '**Bar** (`raster <input_types.html#raster>`__ or `vector <input_types.html#vector>`__, *required*): Description'
        ])
        self.assertEqual(repr(out), repr(expected_rst))

    def test_real_model_spec(self):
        from natcap.invest import carbon
        out = spec_utils.describe_arg_from_name(
            'natcap.invest.carbon', 'carbon_pools_path', 'columns', 'lucode')
        expected_rst = (
            '.. _carbon-pools-path-columns-lucode:\n\n' +
            '**lucode** (`integer <input_types.html#integer>`__, *required*): ' +
            carbon.MODEL_SPEC.inputs.get('carbon_pools_path').columns.get('lucode').about
        )
        self.assertEqual(repr(out), repr(expected_rst))


def _generate_files_from_spec(output_spec, workspace):
    """A utility function to support the metadata test."""
    for spec_data in output_spec:
        print(spec_data.__class__)
        if spec_data.__class__ is spec_utils.DirectoryOutput:
            os.mkdir(os.path.join(workspace, spec_data.id))
            _generate_files_from_spec(
                spec_data.contents, os.path.join(workspace, spec_data.id))
        else:
            filepath = os.path.join(workspace, spec_data.id)
            if hasattr(spec_data, 'band'):
                driver = gdal.GetDriverByName('GTIFF')
                raster = driver.Create(filepath, 2, 2, 1, gdal.GDT_Byte)
                band = raster.GetRasterBand(1)
                band.SetNoDataValue(2)
            elif hasattr(spec_data, 'fields'):
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

    def test_write_metadata(self):
        """Test writing metadata for an invest output workspace."""

        # An example invest output spec
        output_spec = spec_utils.ModelOutputs(
            spec_utils.DirectoryOutput(
                id='output',
                contents=spec_utils.Contents(
                    spec_utils.SingleBandRasterOutput(
                        id="urban_nature_supply_percapita.tif",
                        about="The calculated supply per capita of urban nature.",
                        band=spec_utils.NumberInput(units=u.m**2)
                    ),
                    spec_utils.VectorOutput(
                        id="admin_boundaries.gpkg",
                        about=("A copy of the user's administrative boundaries "
                               "vector with a single layer."),
                        geometries=spec_utils.POLYGONS,
                        fields=spec_utils.Fields(
                            spec_utils.NumberInput(
                                id="SUP_DEMadm_cap",
                                units=u.m**2/u.person,
                                about="The average urban nature supply/demand"
                            )
                        )
                    )
                )
            ),
            spec_utils.DirectoryOutput(
                id='intermediate',
                contents=spec_utils.Contents(
                    spec_utils.build_output_spec('taskgraph_cache', spec_utils.TASKGRAPH_DIR)
                )
            )
        )
        # Generate an output workspace with real files, without
        # running an invest model.
        _generate_files_from_spec(output_spec, self.workspace_dir)

        model_module = types.SimpleNamespace(
            __name__='urban_nature_access',
            execute=lambda: None,
            MODEL_SPEC=spec_utils.ModelSpec(
                model_id='urban_nature_access',
                model_title='Urban Nature Access',
                userguide='',
                aliases=[],
                ui_spec={},
                inputs=spec_utils.ModelInputs(),
                args_with_spatial_overlap={},
                outputs=output_spec
            )
        )

        args_dict = {'workspace_dir': self.workspace_dir}

        spec_utils.generate_metadata(model_module, args_dict)
        files, messages = geometamaker.validate_dir(
            self.workspace_dir, recursive=True)
        self.assertEqual(len(files), 2)
        self.assertFalse(any(messages))

        resource = geometamaker.describe(
            os.path.join(args_dict['workspace_dir'], 'output',
                         'urban_nature_supply_percapita.tif'))
        self.assertCountEqual(resource.get_keywords(),
                              [model_module.MODEL_SPEC.model_id, 'InVEST'])
