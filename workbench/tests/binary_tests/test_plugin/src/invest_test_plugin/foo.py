import logging

import pygeoprocessing

from natcap.invest import validation
from natcap.invest import gettext
from natcap.invest import spec

LOGGER = logging.getLogger(__name__)

MODEL_SPEC = spec.ModelSpec(
    model_id="test",
    model_title="Test Plugin",
    module_name=__name__,
    userguide='',
    input_field_order=[
        ['workspace_dir', 'results_suffix'],
        ['raster_path', 'factor']],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="raster_path",
            name="Input Raster",
            data_type=float,
            about="Raster that will be multiplied by factor, pixelwise.",
            units=None
        ),
        spec.IntegerInput(
            id="factor",
            name="Multiplication Factor"
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="result",
            path="result.tif",
            about="Raster multiplied by factor",
            data_type=float,
            units=None
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def multiply_op(raster_path, factor, target_path):
    pygeoprocessing.raster_map(
        op=lambda x: x * factor,
        rasters=[raster_path],
        target_path=target_path)


def execute(args):
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    task_graph.add_task(
        func=multiply_op,
        kwargs={
            'raster_path': args['raster_path'],
            'factor': int(args['factor']),
            'target_path': file_registry['result']
        },
        target_path_list=[file_registry['result']],
        task_name='multiply raster by factor')
    task_graph.close()
    task_graph.join()
    LOGGER.info('Done!')

    return file_registry.registry


@validation.invest_validator
def validate(args, limit_to=None):
    return validation.validate(args, MODEL_SPEC)
