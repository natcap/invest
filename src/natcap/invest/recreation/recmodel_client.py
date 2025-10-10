"""InVEST Recreation Client."""
import concurrent.futures
import json
import logging
import math
import os
import pickle
import requests
import shutil
import tempfile
import time
import uuid
import zipfile

import numpy
import numpy.linalg
import pygeoprocessing
import Pyro5
import Pyro5.api
import rtree
import shapely
import shapely.geometry
import shapely.prepared
import shapely.wkt
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

# prefer to do intrapackage imports to avoid case where global package is
# installed and we import the global version of it rather than the local
from .. import gettext
from .. import spec
from .. import utils
from .. import validation
from ..unit_registry import u

LOGGER = logging.getLogger(__name__)

# NatCap Rec Server URLs. This is a GCS bucket.
SERVER_URL = 'http://data.naturalcapitalproject.org/server_registry/invest_recreation_model_3_15_0/index.html'  # pylint: disable=line-too-long

# 'marshal' serializer lets us pass null bytes in strings unlike the default
Pyro5.config.SERIALIZER = 'marshal'

# year range supported by both the flickr and twitter databases
MIN_YEAR = 2012
MAX_YEAR = 2017
POLYGON_ID_FIELD = 'poly_id'

PREDICTOR_TABLE_COLUMNS = [
    spec.StringInput(
        id="id",
        about=gettext("A unique identifier for the predictor."),
        regexp=None
    ),
    spec.RasterOrVectorInput(
        id="path",
        about=gettext("A spatial file to use as a predictor."),
        data_type=float,
        units=u.none,
        geometry_types={
            "MULTIPOINT",
            "MULTIPOLYGON",
            "LINESTRING",
            "POINT",
            "MULTILINESTRING",
            "POLYGON",
        },
        fields=[],
        projected=None
    ),
    spec.OptionStringInput(
        id="type",
        about="The type of predictor file provided in the 'path' column.",
        options=[
            spec.Option(
                key="raster_mean",
                about=(
                    "Predictor is a raster. Metric is the mean of values"
                    " within the AOI grid cell or polygon.")),
            spec.Option(
                key="raster_sum",
                about=(
                    "Predictor is a raster. Metric is the sum of values"
                    " within the AOI grid cell or polygon.")),
            spec.Option(
                key="point_count",
                about=(
                    "Predictor is a point vector. Metric is the number of"
                    " points within each AOI grid cell or polygon.")),
            spec.Option(
                key="point_nearest_distance",
                about=(
                    "Predictor is a point vector. Metric is the Euclidean"
                    " distance between the centroid of each AOI grid cell and"
                    " the nearest point in this layer.")),
            spec.Option(
                key="line_intersect_length",
                about=(
                    "Predictor is a line vector. Metric is the total length"
                    " of the lines that fall within each AOI grid cell.")),
            spec.Option(
                key="polygon_area_coverage",
                about=(
                    "Predictor is a polygon vector. Metric is the area of"
                    " overlap between the polygon and each AOI grid cell.")),
            spec.Option(
                key="polygon_percent_coverage",
                about=(
                    "Predictor is a polygon vector. Metric is the percentage"
                    " (0-100) of overlapping area between the polygon and"
                    " each AOI grid cell."))
        ]
    )
]

MODEL_SPEC = spec.ModelSpec(
    model_id="recreation",
    model_title=gettext("Visitation: Recreation and Tourism"),
    userguide="recreation.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["aoi_path"],
        ["start_year", "end_year"],
        ["compute_regression", "predictor_table_path", "scenario_predictor_table_path"],
        ["grid_aoi", "grid_type", "cell_size"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.AOI.model_copy(update=dict(id="aoi_path")),
        spec.StringInput(
            id="hostname",
            name=gettext("hostname"),
            about=gettext(
                "FQDN to a recreation server.  If not provided, a default is assumed."
            ),
            required=False,
            hidden=True,
            regexp=None
        ),
        spec.IntegerInput(
            id="port",
            name=gettext("port"),
            about=gettext(
                "the port on ``hostname`` to use for contacting the recreation server."
            ),
            required=False,
            hidden=True,
            units=u.none,
            expression="value >= 0"
        ),
        spec.IntegerInput(
            id="start_year",
            name=gettext("start year"),
            about=gettext(
                "Year at which to start user-day calculations. Calculations start on the"
                " first day of the year. Year must be in the range 2012 - 2017, and must"
                " be less than or equal to the End Year."
            ),
            units=u.year_AD,
            expression="2012 <= value <= 2017"
        ),
        spec.IntegerInput(
            id="end_year",
            name=gettext("end year"),
            about=gettext(
                "Year at which to end user-day calculations. Calculations continue"
                " through the last day of the year. Year must be in the range 2012 -"
                " 2017, and must be greater than or equal to the Start Year."
            ),
            units=u.year_AD,
            expression="2012 <= value <= 2017"
        ),
        spec.BooleanInput(
            id="grid_aoi",
            name=gettext("grid the AOI"),
            about=gettext(
                "Divide the AOI polygons into equal-sized grid cells, and compute results"
                " for those cells instead of the original polygons."
            ),
            required=False
        ),
        spec.OptionStringInput(
            id="grid_type",
            name=gettext("grid type"),
            about=gettext(
                "The shape of grid cells to make within the AOI polygons. Required if"
                " Grid AOI is selected."
            ),
            required="grid_aoi",
            allowed="grid_aoi",
            options=[
                spec.Option(key="square"),
                spec.Option(key="hexagon")
            ]
        ),
        spec.NumberInput(
            id="cell_size",
            name=gettext("cell size"),
            about=(
                "Size of grid cells to make, measured in the projection units of the AOI."
                " If the Grid Type is 'square', this is the length of each side of the"
                " square. If the Grid Type is 'hexagon', this is the hexagon's maximal"
                " diameter."
            ),
            required="grid_aoi",
            allowed="grid_aoi",
            units=u.other,
            expression="value > 0"
        ),
        spec.BooleanInput(
            id="compute_regression",
            name=gettext("compute regression"),
            about=gettext(
                "Run the regression model using the predictor table and scenario table,"
                " if provided."
            ),
            required=False
        ),
        spec.CSVInput(
            id="predictor_table_path",
            name=gettext("predictor table"),
            about=gettext(
                "A table that maps predictor IDs to spatial files and their predictor"
                " metric types. The file paths can be absolute or relative to the table."
            ),
            required="compute_regression",
            allowed="compute_regression",
            columns=PREDICTOR_TABLE_COLUMNS,
            index_col="id"
        ),
        spec.CSVInput(
            id="scenario_predictor_table_path",
            name=gettext("scenario predictor table"),
            about=gettext(
                "A table of future or alternative scenario predictors. Maps IDs to files"
                " and their types. The file paths can be absolute or relative to the"
                " table."
            ),
            required=False,
            allowed="compute_regression",
            columns=PREDICTOR_TABLE_COLUMNS,
            index_col="id"
        )
    ],
    outputs=[
        spec.VectorOutput(
            id="pud_results",
            path="PUD_results.gpkg",
            about=gettext("Results of photo-user-days aggregations in the AOI."),
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.NumberOutput(
                    id="PUD_YR_AVG",
                    about=gettext("The average photo-user-days per year"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="PUD_[MONTH]",
                    about=gettext("The average photo-user-days for each month."),
                    units=u.none
                )
            ]
        ),
        spec.VectorOutput(
            id="tud_results",
            path="TUD_results.gpkg",
            about=gettext("Results of twitter-user-days aggregations in the AOI."),
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.NumberOutput(
                    id="PUD_YR_AVG",
                    about=gettext("The average twitter-user-days per year"),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="PUD_[MONTH]",
                    about=gettext("The average twitter-user-days for each month."),
                    units=u.none
                )
            ]
        ),
        spec.CSVOutput(
            id="pud_monthly_table",
            path="PUD_monthly_table.csv",
            about=gettext("Table of monthly photo-user-days in each AOI polygon."),
            columns=[
                spec.IntegerOutput(id="poly_id", about=gettext("Polygon ID")),
                spec.NumberOutput(
                    id="[YEAR]-[MONTH]",
                    about=gettext(
                        "Total photo-user-days counted in the polygon in the given month."
                    ),
                    units=u.none
                )
            ],
            index_col="poly_id"
        ),
        spec.CSVOutput(
            id="tud_monthly_table",
            path="TUD_monthly_table.csv",
            about=gettext("Table of monthly twitter-user-days in each AOI polygon."),
            columns=[
                spec.IntegerOutput(id="poly_id", about=gettext("Polygon ID")),
                spec.NumberOutput(
                    id="[YEAR]-[MONTH]",
                    about=gettext(
                        "Total twitter-user-days counted in the polygon in the given"
                        " month."
                    ),
                    units=u.none
                )
            ],
            index_col="poly_id"
        ),
        spec.VectorOutput(
            id="regression_data",
            path="regression_data.gpkg",
            about=gettext(
                "AOI polygons with all the variables needed to compute a regression,"
                " including predictor attributes and the user-days response variable."
            ),
            created_if="compute_regression",
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.NumberOutput(
                    id="[PREDICTOR]",
                    about=gettext("Predictor attribute value for each polygon."),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="pr_TUD",
                    about=gettext(
                        "proportion of the sum of TUD_YR_AVG across all features."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="pr_PUD",
                    about=gettext(
                        "proportion of the sum of PUD_YR_AVG across all features."
                    ),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="avg_pr_UD",
                    about=gettext(
                        "average of pr_TUD and pr_TUD. This variable is logit-transformed"
                        " and then used as the response variable in the regression model."
                    ),
                    units=u.none
                )
            ]
        ),
        spec.FileOutput(
            id="regression_summary",
            path="regression_summary.txt",
            about=gettext(
                "This is a text file output of the regression analysis. It includes"
                " estimates for each predictor variable. It also contains a “server id"
                " hash” value which can be used to correlate the PUD result with the data"
                " available on the PUD server. If these results are used in publication"
                " this hash should be included with the results for reproducibility."
            ),
            created_if="compute_regression"
        ),
        spec.CSVOutput(
            id="regression_coefficients",
            path="regression_coefficients.csv",
            about=gettext("Regression coefficients table")
        ),
        spec.VectorOutput(
            id="scenario_results",
            path="scenario_results.gpkg",
            about=gettext(
                "Results of scenario, including the predictor data used in the scenario"
                " and the predicted visitation patterns for the scenario."
            ),
            created_if="scenario_predictor_table_path",
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.NumberOutput(
                    id="[PREDICTOR]",
                    about=gettext("Predictor attribute value for each polygon."),
                    units=u.none
                ),
                spec.NumberOutput(
                    id="pr_UD_EST",
                    about=gettext(
                        "The estimated avg_pr_UD for each polygon. Estimated using the"
                        " regression coefficients for each predictor in"
                        " regression_coefficients.txt"
                    ),
                    units=u.none
                )
            ]
        ),
        spec.VectorOutput(
            id="aoi",
            path="intermediate/aoi.gpkg",
            about=gettext("Copy of the input AOI, gridded if applicable."),
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[]
        ),
        spec.FileOutput(
            id="aoi_zip",
            path="intermediate/aoi.zip",
            about=gettext("Compressed AOI")
        ),
        spec.FileOutput(
            id="[PREDICTOR]_json",
            path="intermediate/[PREDICTOR].json",
            about=gettext("aggregated predictor values within each polygon")
        ),
        spec.FileOutput(
            id="predictor_estimates",
            path="intermediate/predictor_estimates.json",
            about=gettext("Predictor estimates")
        ),
        spec.FileOutput(
            id="pud_zip",
            path="intermediate/pud.zip",
            about=gettext("Compressed photo-user-day data")
        ),
        spec.FileOutput(
            id="tud_zip",
            path="intermediate/tud.zip",
            about=gettext("Compressed twitter-user-day data")
        ),
        spec.FileOutput(
            id="response_polygons_lookup",
            path="intermediate/response_polygons_lookup.pickle",
            about=gettext(
                "Pickled dictionary mapping FIDs to shapely geometries"
            )
        ),
        spec.FileOutput(
            id="scenario_[PREDICTOR]",
            path="intermediate/scenario/[PREDICTOR].json",
            about=gettext(
                "aggregated scenario predictor values within each polygon"
            )
        ),
        spec.FileOutput(
            id="server_version",
            path="intermediate/server_version.pickle",
            about=gettext("Server version info")
        ),
        spec.TASKGRAPH_CACHE
    ]
)


# Have 5 seconds between timed progress outputs
LOGGER_TIME_DELAY = 5

RESPONSE_VARIABLE_ID = 'avg_pr_UD'
SCENARIO_RESPONSE_ID = 'pr_UD_EST'


def execute(args):
    """Recreation.

    Execute recreation client model on remote server.

    Args:
        args['workspace_dir'] (string): path to workspace directory
        args['aoi_path'] (string): path to AOI vector
        args['hostname'] (string): FQDN to recreation server
        args['port'] (string or int): port on hostname for recreation server
        args['start_year'] (string): start year in form YYYY.  This year
            is the inclusive lower bound to consider points in the PUD and
            regression.
        args['end_year'] (string): end year in form YYYY.  This year
            is the inclusive upper bound to consider points in the PUD and
            regression.
        args['grid_aoi'] (boolean): if true the polygon vector in
            ``args['aoi_path']`` should be gridded into a new vector and the
            recreation model should be executed on that
        args['grid_type'] (string): optional, but must exist if
            args['grid_aoi'] is True.  Is one of 'hexagon' or 'square' and
            indicates the style of gridding.
        args['cell_size'] (string/float): optional, but must exist if
            ``args['grid_aoi']`` is True.  Indicates the cell size of square
            pixels and the width of the horizontal axis for the hexagonal
            cells.
        args['compute_regression'] (boolean): if True, then process the
            predictor table and scenario table (if present).
        args['predictor_table_path'] (string): required if
            args['compute_regression'] is True.  Path to a table that
            describes the regression predictors, their IDs and types.  Must
            contain the fields 'id', 'path', and 'type' where:

                * 'id': is a <=10 character length ID that is used to uniquely
                  describe the predictor.  It will be added to the output
                  result shapefile attribute table which is an ESRI
                  Shapefile, thus limited to 10 characters.
                * 'path': an absolute or relative (to this table) path to the
                  predictor dataset, either a vector or raster type.
                * 'type': one of the following,

                    * 'raster_mean': mean of values in the raster under the
                      response polygon
                    * 'raster_sum': sum of values in the raster under the
                      response polygon
                    * 'point_count': count of the points contained in the
                      response polygon
                    * 'point_nearest_distance': distance to the nearest point
                      from the centroid of the response polygon
                    * 'line_intersect_length': length of lines that intersect
                      with the response polygon in projected units of AOI
                    * 'polygon_area': area of the polygon contained within
                      response polygon in projected units of AOI
                    * 'polygon_percent_coverage': percent (0-100) of area of
                      overlap between the predictor and each AOI grid cell

        args['scenario_predictor_table_path'] (string): (optional) if
            present runs the scenario mode of the recreation model with the
            datasets described in the table on this path.  Field headers
            are identical to ``args['predictor_table_path']`` and ids in the
            table are required to be identical to the predictor list.
        args['results_suffix'] (string): optional, if exists is appended to
            any output file paths.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    if args['end_year'] < args['start_year']:
        raise ValueError(
            "Start year must be less than or equal to end year.\n"
            f"start_year: {args['start_year']}\nend_year: {args['end_year']}")
    # in case the user defines a hostname
    if args['hostname'] and args['port']:
        server_url = f"PYRO:natcap.invest.recreation@{args['hostname']}:{args['port']}"
    else:
        # else use a well known path to get active server
        server_url = requests.get(SERVER_URL).text.rstrip()
        LOGGER.info(server_url)

    if args['grid_aoi']:
        prep_aoi_task = task_graph.add_task(
            func=_grid_vector,
            args=(args['aoi_path'], args['grid_type'],
                  float(args['cell_size']), file_registry['aoi']),
            target_path_list=[file_registry['aoi']],
            task_name='grid_aoi')
    else:
        # Even if we don't modify the AOI by gridding it, we still need
        # to move it to the expected location.
        prep_aoi_task = task_graph.add_task(
            func=_copy_aoi_no_grid,
            args=(args['aoi_path'], file_registry['aoi']),
            target_path_list=[file_registry['aoi']],
            task_name='copy_aoi')
    # All other tasks are dependent on this one, including tasks added
    # within _schedule_predictor_data_processing(). Rather than passing
    # this task to that function, I'm joining here.
    prep_aoi_task.join()

    # All the server communication happens in this task.
    calc_user_days_task = task_graph.add_task(
        func=_retrieve_user_days,
        args=(file_registry['aoi'],
              file_registry['aoi_zip'],
              args['start_year'], args['end_year'], args['results_suffix'],
              args['workspace_dir'], server_url, file_registry['server_version'],
              file_registry['pud_zip'], file_registry['tud_zip']),
        target_path_list=[file_registry['aoi_zip'],
                          file_registry['pud_results'],
                          file_registry['pud_monthly_table'],
                          file_registry['tud_results'],
                          file_registry['tud_monthly_table'],
                          file_registry['server_version']],
        task_name='user-day-calculation')

    assemble_userday_variables_task = task_graph.add_task(
        func=_assemble_regression_data,
        args=(file_registry['pud_results'],
              file_registry['tud_results'],
              file_registry['regression_data']),
        target_path_list=[file_registry['regression_data']],
        dependent_task_list=[calc_user_days_task],
        task_name='assemble userday variables')

    if args['compute_regression']:
        # Prepare the AOI for geoprocessing.
        prepare_response_polygons_task = task_graph.add_task(
            func=_prepare_response_polygons_lookup,
            args=(file_registry['aoi'],
                  file_registry['response_polygons_lookup']),
            target_path_list=[file_registry['response_polygons_lookup']],
            task_name='prepare response polygons for geoprocessing')

        # Build predictor data
        assemble_predictor_data_task = _schedule_predictor_data_processing(
            file_registry['aoi'],
            file_registry['response_polygons_lookup'],
            [prepare_response_polygons_task, assemble_userday_variables_task],
            args['predictor_table_path'],
            file_registry['regression_data'],
            task_graph, file_registry)

        # Compute the regression
        predictor_df = MODEL_SPEC.get_input(
            'predictor_table_path').get_validated_dataframe(
            args['predictor_table_path'])
        predictor_id_list = predictor_df.index
        compute_regression_task = task_graph.add_task(
            func=_compute_and_summarize_regression,
            args=(file_registry['regression_data'],
                  RESPONSE_VARIABLE_ID,
                  predictor_id_list,
                  file_registry['server_version'],
                  file_registry['predictor_estimates'],
                  file_registry['regression_coefficients'],
                  file_registry['regression_summary']),
            target_path_list=[file_registry['regression_coefficients'],
                              file_registry['regression_summary'],
                              file_registry['predictor_estimates']],
            dependent_task_list=[assemble_predictor_data_task],
            task_name='compute regression')

        if args['scenario_predictor_table_path']:
            driver = gdal.GetDriverByName('GPKG')
            if os.path.exists(file_registry['scenario_results']):
                driver.Delete(file_registry['scenario_results'])
            aoi_vector = gdal.OpenEx(file_registry['aoi'])
            target_vector = driver.CreateCopy(
                file_registry['scenario_results'], aoi_vector)
            target_layer = target_vector.GetLayer()
            _rename_layer_from_parent(target_layer)
            target_vector = target_layer = None
            aoi_vector = None

            build_scenario_data_task = _schedule_predictor_data_processing(
                file_registry['aoi'],
                file_registry['response_polygons_lookup'],
                [prepare_response_polygons_task],
                args['scenario_predictor_table_path'],
                file_registry['scenario_results'],
                task_graph, file_registry)

            task_graph.add_task(
                func=_calculate_scenario,
                args=(file_registry['scenario_results'],
                      SCENARIO_RESPONSE_ID, file_registry['predictor_estimates']),
                target_path_list=[file_registry['scenario_results']],
                dependent_task_list=[
                    compute_regression_task, build_scenario_data_task],
                task_name='calculate scenario')

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def _copy_aoi_no_grid(source_aoi_path, dest_aoi_path):
    """Copy a shapefile from source to destination."""
    aoi_vector = gdal.OpenEx(source_aoi_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('GPKG')
    local_aoi_vector = driver.CreateCopy(
        dest_aoi_path, aoi_vector)
    layer = local_aoi_vector.GetLayer()
    idx = layer.FindFieldIndex(POLYGON_ID_FIELD, 1)
    if idx > -1:  # -1 is index if it does not exist
        layer.DeleteField(idx)
    layer.CreateField(
        ogr.FieldDefn(POLYGON_ID_FIELD, ogr.OFTInteger64))
    layer.StartTransaction()
    for i, feature in enumerate(layer):
        feature.SetField(POLYGON_ID_FIELD, i)
        layer.SetFeature(feature)
    layer.CommitTransaction()
    layer = None
    local_aoi_vector = None
    aoi_vector = None


def _retrieve_user_days(
        local_aoi_path, compressed_aoi_path, start_year, end_year,
        file_suffix, output_dir, server_url, server_version_pickle,
        pud_userdays_path, tud_userdays_path):
    """Calculate user-days (PUD & TUD) on the server and send back results.

    All of the client-server communication happens in this scope. The local AOI
    is sent to the server for aggregations. Results are sent back when
    complete.

    Args:
        local_aoi_path (string): path to polygon vector for UD aggregation
        compressed_aoi_path (string): path to zip file where AOI will be
            compressed
        start_year (int/string): lower limit of date-range for UD queries
        end_year (int/string): upper limit of date-range for UD queries
        file_suffix (string): to append to filenames of files created by server
        output_dir (string): path to output workspace where results are
            unpacked.
        server_url (string): URL for connecting to the server
        server_version_pickle (string): path to a pickle that stores server
            version and workspace id info.
        pud_userdays_path (string): path to compressed photo-user-days data
        tud_userdays_path (string): path to compressed twitter-user-days data

    Returns:
        None

    """
    LOGGER.info('Contacting server, please wait.')
    recmodel_manager = Pyro5.api.Proxy(server_url)

    dataset_tuples = [
        ('flickr', 'PUD', pud_userdays_path),
        ('twitter', 'TUD', tud_userdays_path)
    ]

    aoi_info = pygeoprocessing.get_vector_info(local_aoi_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    target_proj = srs.ExportToWkt()
    aoi_bounding_box = pygeoprocessing.transform_bounding_box(
        aoi_info['bounding_box'], aoi_info['projection_wkt'], target_proj)

    for dataset, _, _ in dataset_tuples:
        # validate available year range
        min_year, max_year = recmodel_manager.get_valid_year_range(dataset)
        LOGGER.info(
            f"{dataset} server supports year queries between {min_year} and {max_year}")
        if not min_year <= int(start_year) <= max_year:
            raise ValueError(
                f"Start year must be between {min_year} and {max_year}.\n"
                f" User input: ({start_year})")
        if not min_year <= int(end_year) <= max_year:
            raise ValueError(
                f"End year must be between {min_year} and {max_year}.\n"
                f" User input: ({end_year})")

        # Check for a reasonably-sized AOI
        n_points, max_allowable = recmodel_manager.estimate_aoi_query_size(
            aoi_bounding_box, dataset)
        if n_points > max_allowable:
            raise ValueError(
                f'The AOI extent is too large. The bounding box '
                f'{aoi_bounding_box} contains up to {n_points} {dataset} points. '
                f'Please reduce the extent of the AOI until it contains '
                f'fewer than {max_allowable} points.')
        LOGGER.info(f'AOI accepted. Fewer than {n_points} {dataset} points '
                    f'found within AOI extent: {aoi_bounding_box}')

    aoi_vector = gdal.OpenEx(local_aoi_path, gdal.OF_VECTOR)
    with zipfile.ZipFile(compressed_aoi_path, 'w') as aoizip:
        for filename in aoi_vector.GetFileList():
            LOGGER.info(f'archiving {filename}')
            aoizip.write(filename, os.path.basename(filename))
    aoi_vector = None

    # convert compressed AOI to binary string for serialization
    with open(compressed_aoi_path, 'rb') as aoifile:
        zip_file_binary = aoifile.read()

    start_time = time.time()
    LOGGER.info('Please wait for server to calculate PUD and TUD...')
    client_id = uuid.uuid4()

    def wrap_calculate_userdays():
        with Pyro5.api.Proxy(server_url) as proxy:
            return proxy.calculate_userdays(
                zip_file_binary, os.path.basename(local_aoi_path),
                start_year, end_year, [tup[0] for tup in dataset_tuples], client_id)

    # Use a separate thread for the long-running remote function call so
    # that we can make concurrent requests for the logging messages
    # queued on the server during that call.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(wrap_calculate_userdays)
        while not future.done():
            record_dict = recmodel_manager.log_to_client(client_id)
            if record_dict:
                try:
                    # Strip workspace_id from the name for readability
                    record_dict['name'] = record_dict['name'].split('_')[0]
                except (KeyError, AttributeError, IndexError):
                    pass
                LOGGER.handle(logging.makeLogRecord(record_dict))
        result_dict = future.result()

    for dataset, result_key, compressed_userdays_path in dataset_tuples:
        result = result_dict[result_key]

        # If an exception occurred on the server's worker, we returned it
        # as a 2-tuple: ('ERROR', 'traceback as formatted string')
        if result[0] == 'ERROR':
            LOGGER.error(result[1])
            raise RuntimeError('An error occurred on the server.')

        result_zip_file_binary, workspace_id, server_version = result

        LOGGER.info(f'Server version {dataset}: {server_version}')
        LOGGER.info(f'workspace_id: {workspace_id}')
        LOGGER.info(f'received result, took {time.time() - start_time} seconds')
        # store server version info in a file so we can list it in results summary.
        with open(server_version_pickle, 'ab') as f:
            pickle.dump({
                dataset: {
                    'server_version': server_version,
                    'workspace_id': workspace_id}}, f)

        # unpack result
        with open(compressed_userdays_path, 'wb') as pud_file:
            pud_file.write(result_zip_file_binary)
        temporary_output_dir = tempfile.mkdtemp(dir=output_dir)
        zipfile.ZipFile(compressed_userdays_path, 'r').extractall(
            temporary_output_dir)

        for filename in os.listdir(temporary_output_dir):
            # Results are returned from the server without a results_suffix.
            pre, post = os.path.splitext(filename)
            target_filepath = os.path.join(
                output_dir, f'{pre}{file_suffix}{post}')
            shutil.copy(
                os.path.join(temporary_output_dir, filename),
                target_filepath)
            # Get the suffix attached to the layername too
            if target_filepath.endswith('gpkg'):
                vector = gdal.OpenEx(target_filepath, gdal.OF_UPDATE)
                layer = vector.GetLayer()
                _rename_layer_from_parent(layer)
                layer = vector = None
        shutil.rmtree(temporary_output_dir)

    LOGGER.info('connection release')
    recmodel_manager._pyroRelease()


def _grid_vector(vector_path, grid_type, cell_size, out_grid_vector_path):
    """Convert vector to a regular grid.

    Here the vector is gridded such that all cells are contained within the
    original vector.  Cells that would intersect with the boundary are not
    produced.

    Args:
        vector_path (string): path to an OGR compatible polygon vector type
        grid_type (string): one of "square" or "hexagon"
        cell_size (float): dimensions of the grid cell in the projected units
            of ``vector_path``; if "square" then this indicates the side length,
            if "hexagon" indicates the width of the horizontal axis.
        out_grid_vector_path (string): path to the output Geopackage
            vector that contains a gridded version of ``vector_path``.

    Returns:
        None

    """
    LOGGER.info("gridding aoi")
    driver = gdal.GetDriverByName('GPKG')
    if os.path.exists(out_grid_vector_path):
        driver.Delete(out_grid_vector_path)

    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    vector_layer = vector.GetLayer()
    spat_ref = vector_layer.GetSpatialRef()

    original_vector_shapes = []
    for feature in vector_layer:
        wkt_feat = shapely.wkt.loads(feature.geometry().ExportToWkt())
        original_vector_shapes.append(wkt_feat)
    vector_layer.ResetReading()
    original_polygon = shapely.prepared.prep(
        shapely.ops.unary_union(original_vector_shapes))

    out_grid_vector = driver.Create(
        out_grid_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    grid_layer = out_grid_vector.CreateLayer(
        os.path.splitext(os.path.basename(out_grid_vector_path))[0],
        spat_ref, ogr.wkbPolygon)
    grid_layer.CreateField(
        ogr.FieldDefn(POLYGON_ID_FIELD, ogr.OFTInteger64))
    grid_layer_defn = grid_layer.GetLayerDefn()

    extent = vector_layer.GetExtent()  # minx maxx miny maxy
    if grid_type == 'hexagon':
        # calculate the inner dimensions of the hexagons
        grid_width = extent[1] - extent[0]
        grid_height = extent[3] - extent[2]
        delta_short_x = cell_size * 0.25
        delta_long_x = cell_size * 0.5
        delta_y = cell_size * 0.25 * (3 ** 0.5)

        # Since the grid is hexagonal it's not obvious how many rows and
        # columns there should be just based on the number of squares that
        # could fit into it.  The solution is to calculate the width and
        # height of the largest row and column.
        n_cols = int(math.floor(grid_width / (3 * delta_long_x)) + 1)
        n_rows = int(math.floor(grid_height / delta_y) + 1)

        def _generate_polygon(col_index, row_index):
            """Generate a points for a closed hexagon."""
            if (row_index + 1) % 2:
                centroid = (
                    extent[0] + (delta_long_x * (1 + (3 * col_index))),
                    extent[2] + (delta_y * (row_index + 1)))
            else:
                centroid = (
                    extent[0] + (delta_long_x * (2.5 + (3 * col_index))),
                    extent[2] + (delta_y * (row_index + 1)))
            x_coordinate, y_coordinate = centroid
            hexagon = [(x_coordinate - delta_long_x, y_coordinate),
                       (x_coordinate - delta_short_x, y_coordinate + delta_y),
                       (x_coordinate + delta_short_x, y_coordinate + delta_y),
                       (x_coordinate + delta_long_x, y_coordinate),
                       (x_coordinate + delta_short_x, y_coordinate - delta_y),
                       (x_coordinate - delta_short_x, y_coordinate - delta_y),
                       (x_coordinate - delta_long_x, y_coordinate)]
            return hexagon
    elif grid_type == 'square':
        def _generate_polygon(col_index, row_index):
            """Generate points for a closed square."""
            square = [
                (extent[0] + col_index * cell_size + x,
                 extent[2] + row_index * cell_size + y)
                for x, y in [
                    (0, 0), (cell_size, 0), (cell_size, cell_size),
                    (0, cell_size), (0, 0)]]
            return square
        n_rows = int((extent[3] - extent[2]) / cell_size)
        n_cols = int((extent[1] - extent[0]) / cell_size)
    else:
        raise ValueError(f'Unknown polygon type: {grid_type}')

    poly_id = 0
    for row_index in range(n_rows):
        for col_index in range(n_cols):
            polygon_points = _generate_polygon(col_index, row_index)
            shapely_feature = shapely.geometry.Polygon(polygon_points)
            if original_polygon.contains(shapely_feature):
                poly = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
                poly_feature = ogr.Feature(grid_layer_defn)
                poly_feature.SetGeometry(poly)
                poly_feature.SetField(POLYGON_ID_FIELD, poly_id)
                grid_layer.CreateFeature(poly_feature)
                poly_id += 1


def _schedule_predictor_data_processing(
        response_vector_path, response_polygons_pickle_path,
        dependent_task_list, predictor_table_path,
        target_predictor_vector_path, task_graph, file_registry):
    """Summarize spatial predictor data by polygons in the response vector.

    Build a shapefile with geometry from the response vector, and tabular
    data from aggregate metrics of spatial predictor datasets in
    ``predictor_table_path``.

    Args:
        response_vector_path (string): path to a single layer polygon vector.
        response_polygons_pickle_path (string): path to pickle that stores a
            dict which maps each feature FID from ``response_vector_path`` to
            its shapely geometry.
        dependent_task_list (list): list of Taskgraph.Task objects.
        predictor_table_path (string): path to a CSV file with three columns
            'id', 'path' and 'type'.  'id' is the unique ID for that predictor
            and must be less than 10 characters long. 'path' indicates the
            full or relative path to the ``predictor_table_path`` table for the
            spatial predictor dataset. 'type' is one of:
                'point_count': count # of points per response polygon
                'point_nearest_distance': distance from nearest point to the
                    centroid of the response polygon
                'line_intersect_length': length of line segments intersecting
                    each response polygon
                'polygon_area': area of predictor polygon intersecting the
                    response polygon
                'polygon_percent_coverage': percent of predictor polygon
                    intersecting the response polygon
                'raster_sum': sum of predictor raster under the response
                    polygon
                'raster_mean': average of predictor raster under the
                    response polygon
        target_predictor_vector_path (string): path to a copy of
            ``response_vector_path`` with a column for each id in
            predictor_table_path. Overwritten if exists.
        task_graph (Taskgraph): the graph that was initialized in execute()
        file_registry (FileRegistry): used to look up predictor json paths

    Returns:
        The ultimate task object from this branch of the taskgraph.

    """
    LOGGER.info('Processing predictor datasets')

    # lookup functions for response types
    # polygon predictor types are a special case because the polygon_area
    # function requires a 'mode' argument that these fucntions do not.
    predictor_functions = {
        'point_count': _point_count,
        'point_nearest_distance': _point_nearest_distance,
        'line_intersect_length': _line_intersect_length,
    }

    predictor_df = MODEL_SPEC.get_input(
        'predictor_table_path').get_validated_dataframe(predictor_table_path)
    predictor_task_list = []
    predictor_json_list = []  # tracks predictor files to add to gpkg
    predictor_ids = []
    for predictor_id, row in predictor_df.iterrows():
        LOGGER.info(f"Building predictor {predictor_id}")
        predictor_type = row['type']
        predictor_target_path = file_registry['[PREDICTOR]_json', predictor_id]
        predictor_ids.append(predictor_id)
        predictor_json_list.append(predictor_target_path)
        if predictor_type.startswith('raster'):
            # type must be one of raster_sum or raster_mean
            raster_op_mode = predictor_type.split('_')[1]
            func = _raster_sum_mean
            args = (row['path'], raster_op_mode,
                    response_vector_path, predictor_target_path)
        # polygon types are a special case because the polygon_area
        # function requires an additional 'mode' argument.
        elif predictor_type.startswith('polygon'):
            func = _polygon_area
            args = (predictor_type, response_polygons_pickle_path,
                    row['path'], predictor_target_path)
        else:
            func = predictor_functions[predictor_type]
            args = (response_polygons_pickle_path,
                    row['path'], predictor_target_path)

        predictor_task_list.append(task_graph.add_task(
            func=func,
            args=args,
            target_path_list=[predictor_target_path],
            dependent_task_list=dependent_task_list,
            task_name=f'predictor {predictor_id}'))

    assemble_predictor_data_task = task_graph.add_task(
        func=_json_to_gpkg_table,
        args=(target_predictor_vector_path,
              predictor_ids,
              file_registry),
        target_path_list=[target_predictor_vector_path],
        dependent_task_list=predictor_task_list,
        task_name='assemble predictor data')

    return assemble_predictor_data_task


def _prepare_response_polygons_lookup(
        response_vector_path, target_pickle_path):
    """Translate a shapefile to a dictionary that maps FIDs to geometries."""
    response_vector = gdal.OpenEx(response_vector_path, gdal.OF_VECTOR)
    response_layer = response_vector.GetLayer()
    response_polygons_lookup = {}  # maps FID to prepared geometry
    for response_feature in response_layer:
        feature_geometry = response_feature.GetGeometryRef()
        feature_polygon = shapely.wkt.loads(feature_geometry.ExportToWkt())
        feature_geometry = None
        response_polygons_lookup[response_feature.GetFID()] = feature_polygon
    response_layer = None
    response_vector = None
    with open(target_pickle_path, 'wb') as pickle_file:
        pickle.dump(response_polygons_lookup, pickle_file)


def _json_to_gpkg_table(regression_vector_path, predictor_ids, file_registry):
    """Create a GeoPackage and a field with data from each json file."""
    target_vector = gdal.OpenEx(
        regression_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_layer = target_vector.GetLayer()

    for predictor_id in predictor_ids:
        json_filename = file_registry['[PREDICTOR]_json', predictor_id]
        # Create a new field for the predictor
        # Delete the field first if it already exists
        field_index = target_layer.FindFieldIndex(
            str(predictor_id), 1)
        if field_index >= 0:
            target_layer.DeleteField(field_index)
        predictor_field = ogr.FieldDefn(str(predictor_id), ogr.OFTReal)
        target_layer.CreateField(predictor_field)

        with open(json_filename, 'r') as file:
            predictor_results = json.load(file)
        for feature_id, value in predictor_results.items():
            feature = target_layer.GetFeature(int(feature_id))
            feature.SetField(str(predictor_id), value)
            target_layer.SetFeature(feature)

    target_layer = None
    target_vector = None


def _raster_sum_mean(
        raster_path, op_mode, response_vector_path,
        predictor_target_path):
    """Sum or mean for all non-nodata values in the raster under each polygon.

    Args:
        raster_path (string): path to a raster.
        op_mode (string): either 'mean' or 'sum'.
        response_vector_path (string): path to response polygons
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from ``response_vector_path``
            to values of the raster under the polygon.

    Returns:
        None

    """
    aggregate_results = pygeoprocessing.zonal_statistics(
        (raster_path, 1), response_vector_path,
        polygons_might_overlap=False)
    # remove results for a feature when the pixel count is 0.
    # we don't have non-nodata predictor values for those features.
    aggregate_results = {
        str(fid): stats for fid, stats in aggregate_results.items()
        if stats['count'] != 0}
    if not aggregate_results:
        LOGGER.warning('raster predictor does not intersect with vector AOI')
        # Create an empty file so that Taskgraph has its target file.
        predictor_results = {}
        with open(predictor_target_path, 'w') as jsonfile:
            json.dump(predictor_results, jsonfile)
        return None

    fid_raster_values = {
        'fid': aggregate_results.keys(),
        'sum': [fid['sum'] for fid in aggregate_results.values()],
        'count': [fid['count'] for fid in aggregate_results.values()],
    }

    if op_mode == 'mean':
        mean_results = (
            numpy.array(fid_raster_values['sum']) /
            numpy.array(fid_raster_values['count']))
        predictor_results = dict(
            zip(fid_raster_values['fid'],
                (i.item() for i in mean_results)))
    else:
        predictor_results = dict(
            zip(fid_raster_values['fid'],
                (i.item() for i in fid_raster_values['sum'])))
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(predictor_results, jsonfile)


def _polygon_area(
        mode, response_polygons_pickle_path, polygon_vector_path,
        predictor_target_path):
    """Calculate polygon area overlap.

    Calculates the amount of projected area overlap from ``polygon_vector_path``
    with ``response_polygons_lookup``.

    Args:
        mode (string): one of 'area' or 'percent'.  How this is set affects
            the metric that's output.  'area' is the area covered in projected
            units while 'percent' is percent of the total response area
            covered.
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        polygon_vector_path (string): path to a single layer polygon vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to polygon area coverage.

    Returns:
        None

    """
    start_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    polygons = _ogr_to_geometry_list(polygon_vector_path)
    prepped_polygons = [shapely.prepared.prep(polygon) for polygon in polygons]
    polygon_spatial_index = rtree.index.Index()
    for polygon_index, polygon in enumerate(polygons):
        polygon_spatial_index.insert(polygon_index, polygon.bounds)
    polygon_coverage_lookup = {}  # map FID to point count

    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        if time.time() - start_time > 5:
            LOGGER.info(
                f"{os.path.basename(polygon_vector_path)} polygon area: "
                f"{(100*index)/len(response_polygons_lookup):.2f}% complete")
            start_time = time.time()

        potential_intersecting_poly_ids = polygon_spatial_index.intersection(
            geometry.bounds)
        intersecting_polygons = [
            polygons[polygon_index]
            for polygon_index in potential_intersecting_poly_ids
            if prepped_polygons[polygon_index].intersects(geometry)]
        union = shapely.ops.unary_union(intersecting_polygons)
        polygon_area_coverage = geometry.intersection(union).area

        if mode == 'polygon_area_coverage':
            polygon_coverage_lookup[feature_id] = polygon_area_coverage
        elif mode == 'polygon_percent_coverage':
            polygon_coverage_lookup[str(feature_id)] = (
                polygon_area_coverage / geometry.area * 100)
    LOGGER.info(f"{os.path.basename(polygon_vector_path)} polygon area: "
                f"100.00% complete")

    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(polygon_coverage_lookup, jsonfile)


def _line_intersect_length(
        response_polygons_pickle_path,
        line_vector_path, predictor_target_path):
    """Calculate the length of the intersecting lines on the response polygon.

    Args:
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        line_vector_path (string): path to a single layer line vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to line intersect length.

    Returns:
        None

    """
    last_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    lines = _ogr_to_geometry_list(line_vector_path)
    line_length_lookup = {}  # map FID to intersecting line length

    line_spatial_index = rtree.index.Index()
    for line_index, line in enumerate(lines):
        line_spatial_index.insert(line_index, line.bounds)

    feature_count = None
    for feature_count, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                f"{os.path.basename(line_vector_path)} line intersect length: "
                f"{(100 * feature_count)/len(response_polygons_lookup):.2f}% complete"))
        potential_intersecting_lines = line_spatial_index.intersection(
            geometry.bounds)
        line_length = sum([
            (lines[line_index].intersection(geometry)).length
            for line_index in potential_intersecting_lines if
            geometry.intersects(lines[line_index])])
        line_length_lookup[str(feature_id)] = line_length
    LOGGER.info(f"{os.path.basename(line_vector_path)} line intersect length: "
                "100.00% complete")
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(line_length_lookup, jsonfile)


def _point_nearest_distance(
        response_polygons_pickle_path, point_vector_path,
        predictor_target_path):
    """Calculate distance to nearest point for the centroid of all polygons.

    Args:
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        point_vector_path (string): path to a single layer point vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to distance to nearest point.

    Returns:
        None

    """
    last_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    points = _ogr_to_geometry_list(point_vector_path)
    point_distance_lookup = {}  # map FID to point count

    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        last_time = delay_op(
            last_time, 5, lambda: LOGGER.info(
                f"{os.path.basename(point_vector_path)} point distance: "
                f"{(100*index)/len(response_polygons_lookup):.2f}% complete"))

        point_distance_lookup[str(feature_id)] = min([
            geometry.centroid.distance(point) for point in points])
    LOGGER.info(f"{os.path.basename(point_vector_path)} point distance: "
                "100.00% complete")
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(point_distance_lookup, jsonfile)


def _point_count(
        response_polygons_pickle_path, point_vector_path,
        predictor_target_path):
    """Calculate number of points contained in each response polygon.

    Args:
        response_polygons_pickle_path (str): path to a pickled dictionary which
            maps response polygon feature ID to prepared shapely.Polygon.
        point_vector_path (string): path to a single layer point vector
            object.
        predictor_target_path (string): path to json file to store result,
            which is a dictionary mapping feature IDs from
            ``response_polygons_pickle_path`` to the number of points in that
            polygon.

    Returns:
        None

    """
    last_time = time.time()
    with open(response_polygons_pickle_path, 'rb') as pickle_file:
        response_polygons_lookup = pickle.load(pickle_file)
    points = _ogr_to_geometry_list(point_vector_path)
    point_count_lookup = {}  # map FID to point count

    index = None
    for index, (feature_id, geometry) in enumerate(
            response_polygons_lookup.items()):
        last_time = delay_op(
            last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                f"{os.path.basename(point_vector_path)} point count: "
                f"{(100*index)/len(response_polygons_lookup):.2f}% complete"))
        point_count = len([
            point for point in points if geometry.contains(point)])
        point_count_lookup[str(feature_id)] = point_count
    LOGGER.info(f"{os.path.basename(point_vector_path)} point count: "
                "100.00% complete")
    with open(predictor_target_path, 'w') as jsonfile:
        json.dump(point_count_lookup, jsonfile)


def _ogr_to_geometry_list(vector_path):
    """Convert an OGR type with one layer to a list of shapely geometry.

    Iterates through the features in the ``vector_path``'s first layer and
    converts them to ``shapely`` geometry objects.  if the objects are not
    valid geometry, an attempt is made to buffer the object by 0 units
    before adding to the list.

    Args:
        vector_path (string): path to an OGR vector

    Returns:
        list of shapely geometry objects representing the features in the
        ``vector_path`` layer.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    geometry_list = []
    for feature in layer:
        feature_geometry = feature.GetGeometryRef()
        shapely_geometry = shapely.wkt.loads(feature_geometry.ExportToWkt())
        if not shapely_geometry.is_valid:
            shapely_geometry = shapely_geometry.buffer(0)
        geometry_list.append(shapely_geometry)
        feature_geometry = None
    layer = None
    vector = None
    return geometry_list


def _assemble_regression_data(
        pud_vector_path, tud_vector_path, target_vector_path):
    """Update the vector with the predictor data, adding response variables.

    Args:
        pud_vector_path (string): Path to the vector polygon
            layer with PUD_YR_AVG.
        tud_vector_path (string): Path to the vector polygon
            layer with TUD_YR_AVG.
        target_vector_path (string): The response polygons with predictor data.
            Fields will be added in order to compute the linear regression:
                * pr_PUD
                * pr_TUD
                * avg_pr_UD (the response variable for linear regression)

    Returns:
        None

    """
    pud_vector = gdal.OpenEx(
        pud_vector_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)
    pud_layer = pud_vector.GetLayer()
    tud_vector = gdal.OpenEx(
        tud_vector_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)
    tud_layer = tud_vector.GetLayer()

    driver = gdal.GetDriverByName('GPKG')
    if os.path.exists(target_vector_path):
        driver.Delete(target_vector_path)
    target_vector = driver.CreateCopy(
        target_vector_path, pud_vector)

    target_layer = target_vector.GetLayer()
    _rename_layer_from_parent(target_layer)

    for field in target_layer.schema:
        if field.name != POLYGON_ID_FIELD:
            target_layer.DeleteField(
                target_layer.FindFieldIndex(field.name, 1))

    def _create_field(fieldname):
        field = ogr.FieldDefn(str(fieldname), ogr.OFTReal)
        target_layer.CreateField(field)

    tud_variable_id = 'pr_TUD'
    pud_variable_id = 'pr_PUD'
    _create_field(tud_variable_id)
    _create_field(pud_variable_id)
    _create_field(RESPONSE_VARIABLE_ID)

    # Calculate response variable as the average of two proportions:
    # the proportion of PUD_YR_AVG for each feature
    # the proportion of TUD_YR_AVG for each feature
    sql = f'SELECT SUM(PUD_YR_AVG) sum_PUD FROM {pud_layer.GetName()}'
    with pud_vector.ExecuteSQL(sql) as pud_sum_layer:
        pud_sum = pud_sum_layer.GetNextFeature().GetField('sum_PUD')
    sql = f'SELECT SUM(TUD_YR_AVG) sum_TUD FROM {tud_layer.GetName()}'
    with tud_vector.ExecuteSQL(sql) as tud_sum_layer:
        tud_sum = tud_sum_layer.GetNextFeature().GetField('sum_TUD')

    if pud_sum == 0 or tud_sum == 0:
        raise RuntimeError(
            'Cannot compute regression because either PUD_YR_AVG'
            'or TUD_YR_AVG are all 0s')

    for feature in target_layer:
        pud_feature = pud_layer.GetFeature(feature.GetFID())
        tud_feature = tud_layer.GetFeature(feature.GetFID())
        pr_pud = pud_feature.GetField('PUD_YR_AVG') / pud_sum
        pr_tud = tud_feature.GetField('TUD_YR_AVG') / tud_sum
        feature.SetField(pud_variable_id, pr_pud)
        feature.SetField(tud_variable_id, pr_tud)
        feature.SetField(
            RESPONSE_VARIABLE_ID, (pr_pud + pr_tud) / 2)
        target_layer.SetFeature(feature)

    pud_feature = tud_feature = None
    pud_layer = pud_vector = None
    tud_layer = tud_vector = None
    target_layer = None
    target_vector = None


def _compute_and_summarize_regression(
        data_vector_path, response_id, predictor_list, server_version_path,
        target_coefficient_json_path, target_coefficient_csv_path,
        target_regression_summary_path):
    """Compute a regression and summary statistics and generate a report.

    Args:
        data_vector_path (string): path to polygon vector containing the
            ``response_id`` field and predictor data
        response_id (string): column name from ``data_vector_path`` table
            that contains the response variable.
        predictor_list (list): list of strings with names of predictor
            variables. These must be columns contained in ``data_vector_path``.
        server_version_path (string): path to pickle file containing the
            rec server id hash.
        target_coefficient_json_path (string): path to json file to store a
            dictionary that maps a predictor id its coefficient estimate.
            This file is created by this function.
        target_coefficient_csv_path (string): path to csv file with predictor
            estimates and errors. Created by this function.
        target_regression_summary_path (string): path to txt file for the report.
            This file is created by this function.

    Returns:
        None

    """
    predictor_id_list, coefficients, ssres, r_sq, r_sq_adj, std_err, dof, se_est, eps = (
        _build_regression(
            data_vector_path, predictor_list, response_id))

    t_values = [coef / se for coef, se in zip(coefficients, se_est)]
    # Generate a nice looking regression result and write to log and file
    coefficients_string = '               estimate     stderr    t value\n'
    # The last coefficient is the y-intercept,
    # but we want it at the top of the report, thus [-1] on lists
    coefficients_string += (
        f'{predictor_id_list[-1]:12} {coefficients[-1]:+.3e} '
        f'{se_est[-1]:+.3e} {t_values[-1]:+.3e}\n')
    # Since the intercept has already been reported, [:-1] on all the lists
    coefficients_string += '\n'.join(
        f'{p_id:12} {coefficient:+.3e} {se_est_factor:+.3e} '
        f'{t:+.3e}'
        for p_id, coefficient, se_est_factor, t in zip(
            predictor_id_list[:-1], coefficients[:-1], se_est[:-1], t_values[:-1]))

    # Include the server version and hash in the report:
    with open(server_version_path, 'rb') as f:
        server_version = pickle.load(f)
    report_string = (
        f'\n******************************\n'
        f'{coefficients_string}\n'
        f'---\n\n'
        f'Residual standard error: {std_err:.4f} on {dof} degrees of freedom\n'
        f'Multiple R-squared: {r_sq:.4f}\n'
        f'Adjusted R-squared: {r_sq_adj:.4f}\n'
        f'SSres: {ssres:.4f}\n'
        f'server id hash: {server_version}\n'
        f'********************************\n')
    LOGGER.info(report_string)
    with open(target_regression_summary_path, 'w') as \
            regression_log:
        regression_log.write(report_string + '\n')

    # Predictor coefficients and epsilon are needed for _calculate_scenario()
    predictor_estimates = dict(zip(predictor_id_list, coefficients))
    predictor_estimates['epsilon'] = eps
    with open(target_coefficient_json_path, 'w') as json_file:
        json.dump(predictor_estimates, json_file)
    # It's also convenient for end-users to have coefficients in a table.
    with open(target_coefficient_csv_path, 'w') as csvfile:
        csvfile.write('predictor,estimate,stderr,t-value\n')
        for p_id, coef, se, t in zip(predictor_id_list, coefficients, se_est, t_values):
            csvfile.write(','.join([p_id, str(coef), str(se), str(t)])+'\n')


def _build_regression(
        data_vector_path, predictor_id_list,
        response_id):
    """Multiple least-squares regression with logit-transformed response.

    The regression is built such that each feature in the single layer vector
    ``data_vector_path`` corresponds to one data point.
    ``response_id`` is the response variable found in ``response_vector_path``.
    The response variable is an average of two proportions and is
    logit-transformed following Warton & Hui 2011 (doi: 10.1890/10-0340.1).

    Predictor variables are found in``predictor_vector_path`` and are not
    transformed. Features with incomplete data are dropped prior to
    computing the regression.

    Args:
        data_vector_path (string): path to polygon vector with fields for
            each predictor in ``predictor_id_list`` and ``response_id``
        predictor_id_list (list): list of strings that are the predictor
            variable names.
        response_id (string): field name in ``data_vector_path`` whose
            values correspond to the regression response variable.

    Returns:
        predictor_names: A list of predictor id strings. Length matches
            coeffiecients.
        coefficients: A list of coefficients in the least-squares solution
            including the y intercept as the last element.
        ssres: sums of squared residuals
        r_sq: R^2 value
        r_sq_adj: adjusted R^2 value
        std_err: residual standard error
        dof: degrees of freedom
        se_est: A list of standard error estimate, length matches coefficients.
        epsilon: value used to adjust response variable before logit-transform

    """
    LOGGER.info("Computing regression")
    data_vector = gdal.OpenEx(data_vector_path, gdal.OF_VECTOR)
    data_layer = data_vector.GetLayer()

    n_features = data_layer.GetFeatureCount()

    # Response data matrix
    response_array = numpy.empty((n_features, 1))
    for row_index, feature in enumerate(data_layer):
        response_array[row_index, :] = feature.GetField(str(response_id))

    if (response_array == 1).any():
        raise ValueError('Cannot compute a regression because there is only '
                         f'one non-zero observation in {response_id}')
    # Logit transformation.
    # Zeros are replaced with half the smallest non-zero value
    # because zero cannot be log-transformed.
    epsilon = 0
    if (response_array == 0).any():
        epsilon = response_array[response_array > 0].min() / 2
        response_array[response_array == 0] = epsilon
    transformed_array = numpy.log(response_array / (1 - response_array))

    # Y-Intercept data matrix
    intercept_array = numpy.ones_like(transformed_array)
    # Predictor data matrix
    predictor_matrix = numpy.empty((n_features, len(predictor_id_list)))
    for row_index, feature in enumerate(data_layer):
        predictor_matrix[row_index, :] = numpy.array(
            [feature.GetField(str(key)) for key in predictor_id_list])

    # If some predictor has no data across all features, drop that predictor:
    valid_pred = ~numpy.isnan(predictor_matrix).all(axis=0)
    predictor_matrix = predictor_matrix[:, valid_pred]
    predictor_names = [
        pred for (pred, valid) in zip(predictor_id_list, valid_pred)
        if valid]
    n_predictors = predictor_matrix.shape[1]

    # add columns for response variable and y-intercept
    data_matrix = numpy.concatenate(
        (transformed_array, predictor_matrix, intercept_array), axis=1)
    predictor_names.append('(Intercept)')

    # if any variable is missing data for some feature, drop that feature:
    incomplete_rows = numpy.isnan(data_matrix).any(axis=1)
    data_matrix = data_matrix[~incomplete_rows]
    n_features = data_matrix.shape[0]
    n_missing = numpy.count_nonzero(incomplete_rows)
    if n_missing:
        LOGGER.warning(
            f'{n_missing} features are missing data for at least one '
            'predictor and will be ommitted from the regression model. '
            'See regression_data.gpkg to see the missing values.')
    y_factors = data_matrix[:, 0]  # useful to have this as a 1-D array
    coefficients, _, _, _ = numpy.linalg.lstsq(
        data_matrix[:, 1:], y_factors, rcond=-1)

    # numpy lstsq will not always return residuals, but they can be calculated.
    ssres = numpy.sum((
        y_factors -
        numpy.sum(data_matrix[:, 1:] * coefficients, axis=1)) ** 2)
    sstot = numpy.sum((
        numpy.average(y_factors) - y_factors) ** 2)
    dof = n_features - n_predictors - 1
    if sstot == 0 or dof <= 0:
        # this can happen if there is only one sample
        r_sq = 1
        r_sq_adj = 1
    else:
        r_sq = 1 - ssres / sstot
        r_sq_adj = 1 - (1 - r_sq) * (n_features - 1) / dof

    if dof > 0:
        std_err = numpy.sqrt(ssres / dof)
        sigma2 = numpy.sum((
            y_factors - numpy.sum(
                data_matrix[:, 1:] * coefficients, axis=1)) ** 2) / dof
        var_est = sigma2 * numpy.diag(numpy.linalg.pinv(
            numpy.dot(
                data_matrix[:, 1:].T, data_matrix[:, 1:])))
        # negative values that are effectively equal to 0
        # have been observed on some platforms.
        var_est[var_est < 0] = 0
        se_est = numpy.sqrt(var_est)

    else:
        LOGGER.warning(f"Linear model is under constrained with DOF={dof}")
        std_err = sigma2 = numpy.nan
        se_est = var_est = [numpy.nan] * data_matrix.shape[1]
    return (
        predictor_names, coefficients, ssres, r_sq,
        r_sq_adj, std_err, dof, se_est, epsilon)


def _calculate_scenario(
        scenario_results_path, response_id, coefficient_json_path):
    """Estimate the PUD of a scenario given an existing regression equation.

    It is expected that the predictor coefficients have been derived from a
    log normal distribution.

    Args:
        scenario_results_path (string): path to desired output scenario
            vector result which will be geometrically a copy of the input
            AOI but contain the scenario predictor data fields as well as the
            scenario estimated response.
        response_id (string): text ID of response variable to write to
            the scenario result.
        coefficient_json_path (string): path to json file with the pre-existing
            regression results. It contains a dictionary that maps
            predictor id strings to coefficient values. Includes Y-Intercept
            and the epsilon used to adjust the response variable.

    Returns:
        None

    """
    LOGGER.info("Calculating scenario results")

    # Open for writing
    scenario_coefficient_vector = gdal.OpenEx(
        scenario_results_path, gdal.OF_VECTOR | gdal.GA_Update)
    scenario_coefficient_layer = scenario_coefficient_vector.GetLayer()

    # delete the response ID if it's already there because it must have been
    # copied from the base layer
    response_index = scenario_coefficient_layer.FindFieldIndex(response_id, 1)
    if response_index >= 0:
        scenario_coefficient_layer.DeleteField(response_index)

    response_field = ogr.FieldDefn(response_id, ogr.OFTReal)
    response_field.SetWidth(24)
    response_field.SetPrecision(11)
    scenario_coefficient_layer.CreateField(response_field)

    # Load the pre-existing predictor coefficients to build the regression
    # equation.
    with open(coefficient_json_path, 'r') as json_file:
        predictor_estimates = json.load(json_file)

    y_intercept = predictor_estimates.pop('(Intercept)')
    epsilon = predictor_estimates.pop('epsilon')

    for feature in scenario_coefficient_layer:
        feature_id = feature.GetFID()
        response_value = 0
        try:
            for predictor_id, coefficient in predictor_estimates.items():
                response_value += (
                    coefficient *
                    feature.GetField(str(predictor_id)))
        except TypeError:
            # TypeError will happen if GetField returned None
            LOGGER.warning('incomplete predictor data for feature_id '
                           f'{feature_id}, not estimating UD_EST')
            feature = None
            continue  # without writing to the feature
        response_value += y_intercept
        # recall that the response was logit-transformed, so inverse it
        value = (
            numpy.exp(response_value) / (numpy.exp(response_value) + 1)
        ) - epsilon
        feature.SetField(response_id, value)
        scenario_coefficient_layer.SetFeature(feature)
        feature = None

    scenario_coefficient_layer = None
    scenario_coefficient_vector.FlushCache()
    scenario_coefficient_vector = None


def _validate_same_id_lengths(table_path):
    """Ensure a predictor table has ids of length less than 10.

    Parameter:
        table_path (string):  path to a csv table that has at least
            the field 'id'

    Return:
        string message if IDs are too long

    """
    predictor_df = MODEL_SPEC.get_input(
        'predictor_table_path').get_validated_dataframe(table_path)
    too_long = set()
    for p_id in predictor_df.index:
        if len(p_id) > 10:
            too_long.add(p_id)
    if len(too_long) > 0:
        return (
            f'The following IDs are more than 10 characters long: {too_long}')


def _validate_same_ids_and_types(
        predictor_table_path, scenario_predictor_table_path):
    """Ensure both tables have same ids and types.

    Assert that both the elements of the 'id' and 'type' fields of each table
    contain the same elements and that their values are the same.  This
    ensures that a user won't get an accidentally incorrect simulation result.

    Args:
        predictor_table_path (string): path to a csv table that has at least
            the fields 'id' and 'type'
        scenario_predictor_table_path (string):  path to a csv table that has
            at least the fields 'id' and 'type'

    Returns:
        string message if any of the fields in 'id' and 'type' don't match
        between tables.
    """
    predictor_df = MODEL_SPEC.get_input(
        'predictor_table_path').get_validated_dataframe(
        predictor_table_path)

    scenario_predictor_df = MODEL_SPEC.get_input(
        'scenario_predictor_table_path').get_validated_dataframe(
        scenario_predictor_table_path)

    predictor_pairs = set([
        (p_id, row['type']) for p_id, row in predictor_df.iterrows()])
    scenario_predictor_pairs = set([
        (p_id, row['type']) for p_id, row in scenario_predictor_df.iterrows()])
    if predictor_pairs != scenario_predictor_pairs:
        return (f'table pairs unequal. predictor: {predictor_pairs} '
                f'scenario: {scenario_predictor_pairs}')


def _validate_same_projection(base_vector_path, table_path):
    """Assert the GIS data in the table are in the same projection as the AOI.

    Args:
        base_vector_path (string): path to a GIS vector
        table_path (string): path to a csv table that has at least
            the field 'path'

    Returns:
        string message if the projections in each of the GIS types in the table
            are not identical to the projection in base_vector_path
    """
    # This will load the table as a list of paths which we can iterate through
    # without bothering the rest of the table structure
    data_paths = MODEL_SPEC.get_input(
        'predictor_table_path').get_validated_dataframe(
        table_path)['path'].tolist()

    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()
    base_ref = osr.SpatialReference(base_layer.GetSpatialRef().ExportToWkt())
    base_layer = None
    base_vector = None

    invalid_projections = False
    for path in data_paths:
        gis_type = pygeoprocessing.get_gis_type(path)
        if gis_type == pygeoprocessing.UNKNOWN_TYPE:
            return f"{path} did not load"
        elif gis_type == pygeoprocessing.RASTER_TYPE:
            raster = gdal.OpenEx(path, gdal.OF_RASTER)
            projection_as_str = raster.GetProjection()
            ref = osr.SpatialReference()
            ref.ImportFromWkt(projection_as_str)
        else:
            vector = gdal.OpenEx(path, gdal.OF_VECTOR)
            layer = vector.GetLayer()
            ref = osr.SpatialReference(layer.GetSpatialRef().ExportToWkt())
        if not base_ref.IsSame(ref):
            invalid_projections = True
    if invalid_projections:
        return (
            f"One or more of the projections in the table ({path}) did not "
            f"match the projection of the base vector ({base_vector_path})")


def _validate_predictor_types(table_path):
    """Validate the type values in a predictor table.

    Args:
        table_path (string): path to a csv table that has at least
            the field 'type'

    Returns:
        string message if any value in the ``type`` column does not match a
        valid type, ignoring leading/trailing whitespace.
    """
    df = MODEL_SPEC.get_input(
        'predictor_table_path').get_validated_dataframe(table_path)
    # ignore leading/trailing whitespace because it will be removed
    # when the type values are used
    valid_types = set({'raster_mean', 'raster_sum', 'point_count',
                       'point_nearest_distance', 'line_intersect_length',
                       'polygon_area_coverage', 'polygon_percent_coverage'})
    difference = set(df['type']).difference(valid_types)
    if difference:
        return (f'The table contains invalid type value(s): {difference}. '
                f'The allowed types are: {valid_types}')


def delay_op(last_time, time_delay, func):
    """Execute ``func`` if last_time + time_delay >= current time.

    Args:
        last_time (float): last time in seconds that ``func`` was triggered
        time_delay (float): time to wait in seconds since last_time before
            triggering ``func``
        func (function): parameterless function to invoke if
         current_time >= last_time + time_delay

    Returns:
        If ``func`` was triggered, return the time which it was triggered in
        seconds, otherwise return ``last_time``.

    """
    if time.time() - last_time > time_delay:
        func()
        return time.time()
    return last_time


def _rename_layer_from_parent(layer):
    """Rename a GDAL vector layer to match the dataset filename."""
    lyrname = os.path.splitext(
        os.path.basename(layer._parent_ds().GetName()))[0]
    layer.Rename(lyrname)


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to ``execute``'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in ``execute`` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire ``args`` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.

    """
    validation_messages = validation.validate(args, MODEL_SPEC)
    sufficient_valid_keys = (validation.get_sufficient_keys(args) -
                             validation.get_invalid_keys(validation_messages))

    validation_tuples = []
    if 'predictor_table_path' in sufficient_valid_keys:
        validation_tuples += [
            (_validate_same_id_lengths, ['predictor_table_path']),
            (_validate_predictor_types, ['predictor_table_path'])]
        if 'aoi_path' in sufficient_valid_keys:
            validation_tuples.append(
                (_validate_same_projection, ['aoi_path', 'predictor_table_path']))
        if 'scenario_predictor_table_path' in sufficient_valid_keys:
            validation_tuples.append((
                _validate_same_ids_and_types,
                ['predictor_table_path', 'scenario_predictor_table_path']))
    if 'scenario_predictor_table_path' in sufficient_valid_keys:
        validation_tuples.append((
            _validate_predictor_types, ['scenario_predictor_table_path']))
        if 'aoi_path' in sufficient_valid_keys:
            validation_tuples.append((_validate_same_projection,
                ['aoi_path', 'scenario_predictor_table_path']))

    for validate_func, key_list in validation_tuples:
        msg = validate_func(*[args[key] for key in key_list])
        if msg:
            validation_messages.append((key_list, msg))

    if 'start_year' in sufficient_valid_keys and 'end_year' in sufficient_valid_keys:
        if int(args['end_year']) < int(args['start_year']):
            validation_messages.append((
                ['start_year', 'end_year'],
                "Start year must be less than or equal to end year."))

    return validation_messages
