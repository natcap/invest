import os

from osgeo import gdal
from osgeo import ogr
from natcap.invest import spec
from natcap.invest.file_registry import FileRegistry
from natcap.invest.unit_registry import u


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
        id="mask_[A]",  # testing with a pattern
        path="intermediate/mask_[A].tif",
        about="A mask for the final raster output.",
        data_type=float,
        units=u.m**2
    ),
    spec.TASKGRAPH_CACHE.model_copy(update=dict(
        path="intermediate/taskgraph_cache/taskgraph.db")
    )
]

SAMPLE_MODEL_SPEC = spec.ModelSpec(
    model_id='urban_nature_access',
    model_title='Urban Nature Access',
    userguide='',
    aliases=[],
    input_field_order=[],
    inputs=[],
    module_name='',
    outputs=output_spec
)


def assert_complete_execute(raw_args, model_spec, **kwargs):
    """Assert that post-processing functions completed.

    This assertion can be used after calling ``model_spec.execute`` with
    various options to assert that expected files exist.

    Args:
        raw_args (dict): the args dict passed to ``execute``
        model_spec (natcap.invest.spec.ModelSpec): the model's specification
        kwargs (dict): kwargs that can be passed to ``execute``.

    Raises:
        AssertionError if expected files do not exist.
    """
    args = model_spec.preprocess_inputs(raw_args)
    if 'save_file_registry' in kwargs and kwargs['save_file_registry']:
        if not os.path.exists(
            os.path.join(args['workspace_dir'],
                         f'file_registry{args["results_suffix"]}.json')):
            raise AssertionError('file registry json file does not exist')
    if 'generate_report' in kwargs and kwargs['generate_report']:
        if not os.path.exists(
            os.path.join(args['workspace_dir'],
                         f'{model_spec.model_id}_report{args["results_suffix"]}.html')):
            raise AssertionError('model report html file does not exist')


def fake_execute(output_spec, workspace):
    """A function to support tests that need a real invest output workspace."""
    file_registry = FileRegistry(output_spec, workspace, '')
    for spec_data in output_spec:
        reg_key = spec_data.id
        if '[' in spec_data.id:
            reg_key = (spec_data.id, 'A')
        filepath = file_registry[reg_key]
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
    return file_registry.registry
