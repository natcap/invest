import pint
from natcap.invest import spec

ureg = pint.UnitRegistry()
ureg.define('none = []')

MODEL_SPEC = spec.ModelSpec(
    module_name=__name__,
    model_id="forest_carbon",
    model_title="Forest Carbon Edge Effect Model",
    userguide="carbon_edge.html",
    validate_spatial_overlap=["aoi_vector_path", "lulc_raster_path"],
    aliases=("fc",),
    different_projections_ok=False,
    input_field_order=[
        ["number_input", "ratio_input"],
        ["percent_input", "integer_input", "boolean_input",
         "freestyle_string_input", "option_string_input", "raster_input",
         "another_raster_input", "vector_input", "csv_input",
         "directory_input"],
    ],
    inputs=[
        spec.NumberInput(
            id="number_input",
            name="Foo",
            about=(
                "Numbers have units that are displayed in a human-readable "
                "way."),
            units=ureg.meter**3 / ureg.month,
            expression="value >= 0",
        ),
        spec.RatioInput(
            id="ratio_input",
            name="Bar",
            about="Here's a ratio",
        ),
        spec.PercentInput(
            id="percent_input",
            name="Baz",
            about="Here's a percent.",
            required=False,
        ),
        spec.IntegerInput(
            id="integer_input",
            name="Abc",
            about="Here's an integer.",
            required=True,
        ),
        spec.BooleanInput(
            id="boolean_input",
            name="Defg",
            about="Here's a boolean.",
        ),
        spec.StringInput(
            id="freestyle_string_input",
            name="Hijk",
            about=(
                "Here's a freestyle string. If its spec has a `regexp` "
                "attribute, we don't display that. The `about` attribute "
                "should describe any required pattern in a user-friendly way."
            ),
        ),
        spec.OptionStringInput(
            id="option_string_input",
            name="Lmn",
            about=(
                "For option_strings, we display the options in a bullet "
                "list."),
            options=[
                spec.Option(
                    key="option_a",
                    display_name="do something",
                ),
                spec.Option(
                    key="option_b",
                    display_name="do something else",
                ),
            ]),
        spec.SingleBandRasterInput(
            id="raster_input",
            name="Opq",
            about="Rasters are pretty simple",
            units=None,
        ),
        spec.SingleBandRasterInput(
            id="another_raster_input",
            name="Rst",
            about=(
                "If the raster's band is a `number` type, display its "
                "units."),
            data_type=float,
            units=ureg.millimeter/ureg.year,
        ),
        spec.VectorInput(
            id="vector_input",
            name="Uvw",
            about="Display vector geometries in an ordered list",
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[],
        ),
        spec.CSVInput(
            id="csv_input",
            name="☺",
            about="Unicode characters work too 😎",
            columns=[
                spec.NumberInput(
                    id="a",
                    units=ureg.second,
                    about="Here's a description"
                )
            ]
        ),
        spec.DirectoryInput(
            id="directory_input",
            name="Foo",
            about="Here's a directory",
            contents=[
                spec.CSVInput(
                    id="baz",
                    required=False,
                    columns=[
                        spec.NumberInput(
                            id="int_column",
                            units=None,
                        ),
                        spec.StringInput(
                            id="description",
                            required=False,
                            about="a description of the id"
                        ),
                        spec.SingleBandRasterInput(
                            id="raster_path",
                            units=ureg.meter,
                        ),
                    ],
                ),
            ],
        )
    ],
    outputs=[],
)
