import pint
from natcap.invest import spec

ureg = pint.UnitRegistry()
ureg.define('none = []')

MODEL_SPEC = spec.build_model_spec({
    "model_id": "forest_carbon",
    "model_title": "Forest Carbon Edge Effect Model",
    "userguide": "carbon_edge.html",
    "aliases": set(),
    "args_with_spatial_overlap": {
        "spatial_keys": ["aoi_vector_path", "lulc_raster_path"],
    },
    "ui_spec": {
        "order": []
    },
    "args": {
        "number_input": {
            "name": "Foo",
            "about": "Numbers have units that are displayed in a human-readable way.",
            "type": "number",
            "units": ureg.meter**3/ureg.month,
            "expression": "value >= 0"
        },
        "ratio_input": {
            "name": "Bar",
            "about": "Here's a ratio.",
            "type": "ratio"
        },
        "percent_input": {
            "name": "Baz",
            "about": "Here's a percent.",
            "type": "percent",
            "required": False
        },
        "integer_input": {
            "name": "Abc",
            "about": "Here's an integer.",
            "type": "integer",
            "required": True
        },
        "boolean_input": {
            "name": "Defg",
            "about": "Here's a boolean.",
            "type": "boolean"
        },
        "freestyle_string_input": {
            "name": "Hijk",
            "about": (
                "Here's a freestyle string. If its spec has a `regexp` "
                "attribute, we don't display that. The `about` attribute "
                "should describe any required pattern in a user-friendly way."
            ),
            "type": "freestyle_string"
        },
        "option_string_input": {
            "name": "Lmn",
            "about": (
                "For option_strings, we display the options in a bullet list."),
            "type": "option_string",
            "options": {
                "option_a": "do something",
                "option_b": "do something else"
            }
        },
        "raster_input": {
            "type": "raster",
            "bands": {1: {"type": "integer"}},
            "about": "Rasters are pretty simple.",
            "name": "Opq"
        },
        "another_raster_input": {
            "type": "raster",
            "bands": {1: {
                "type": "number",
                "units": ureg.millimeter/ureg.year
            }},
            "about": (
                "If the raster's band is a `number` type, display its units"),
            "name": "Rst"
        },
        "vector_input": {
            "type": "vector",
            "fields": {},
            "geometries": {"POLYGON", "MULTIPOLYGON"},
            "about": "Display vector geometries in an ordered list.",
            "name": "Uvw"
        },
        "csv_input": {
            "type": "csv",
            "about": "Unicode characters work too 😎",
            "name": "☺",
            "columns": {
                "a": {
                    "type": "number",
                    "units": ureg.second,
                    "about": "Here's a description."
                }
            }
        },
        "directory_input": {
            "type": "directory",
            "about": "Here's a directory",
            "name": "Foo",
            "contents": {
                "baz": {
                    "type": "csv",
                    "required": False,
                    "columns": {
                        "id": {"type": "integer"},
                        "description": {
                            "type": "freestyle_string",
                            "required": False,
                            "about": "a description of the id"
                        },
                        "raster_path": {
                            "type": "raster",
                            "bands": {
                                1: {"type": "number", "units": ureg.meter}
                            }
                        }
                    }
                }
            }
        }
    },
    "outputs": {}
})
