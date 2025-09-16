"""InVEST Urban Mental Health model."""
import logging
import math
import os
import pickle
import shutil
import tempfile


import numpy
import pandas
import pygeoprocessing
import pygeoprocessing.kernels
import taskgraph
from osgeo import gdal
from osgeo import ogr, osr

from . import gettext
from . import spec
from . import utils
from . import validation
from .unit_registry import u

LOGGER = logging.getLogger(__name__)

SCENARIO_OPTIONS = [
    spec.Option(key="tcc_ndvi", display_name="TCC & Baseline NDVI"),
    spec.Option(key="lulc", display_name="Baseline & Alternate LULC"),
    spec.Option(key="ndvi", display_name="Baseline & Alternate NDVI")
    ]
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)

MODEL_SPEC = spec.ModelSpec(
    model_id="urban_mental_health",
    model_title=gettext("Urban Mental Health"),
    userguide="",  # TODO - add this model to UG
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["aoi_vector_path", "population_raster", "search_radius"],
        ["effect_size_csv", "baseline_prevalence_vector",
         "health_cost_rate_csv"],
        ["scenario", "tc_raster", "tc_target", "ndvi_base", "ndvi_alt",
         "lulc_base", "lulc_alt", "lulc_attr_csv"]

        #  ["scenario", "ndvi_base"],
        # ["tc_raster", "tc_target"],
        # ["lulc_base", "lulc_alt", "lulc_attr_csv"],
        # ["ndvi_alt"]
        #  ^ could either be a dropdown menu of scenario options (i.e., "TC + baseline NDVI", "baseline and alternate LULC", "baseline and alternate NDVI")
    ],
    validate_spatial_overlap=True,  # TODO
    different_projections_ok=True,  # TODO
    aliases=("umh",),
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.AOI.model_copy(update=dict(  #  TODO: potentially want to have this req to be census tract pop. shp? or only req that if opt 1 but optional for opts 2-3?
            id="aoi_vector_path",
            about=gettext(
                "Map of the area over which to run the model. It is"
                "recommended that the AOI is smaller than the raster inputs"
                "by at least the search radius to ensure correct edge pixel"
                "calculation."),
            projected=True,
            projection_units=u.meter
        )),
        spec.SingleBandRasterInput(
            id="population_raster",
            name=gettext("population"),
            about=gettext(
                "Gridded population data representing the number of people "
                "per pixel, ideally at the pixel level. Where available, "
                "additional demographic attributes (e.g., age, income) can "
                "enhance model accuracy and equity assessments. For "
                "population count rasters, users are recommended to use the "
                "WorldPop data, which has global coverage at 100 meter "
                "resolution https://hub.worldpop.org/project/categories?id=3"
            ),
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.NumberInput(
            id="search_radius",
            name=gettext("search radius"),
            about=gettext(
                "Distance used to define the surrounding area of a "
                "person's residence that best represents daily exposure "
                "to nearby nature."),
            # type=float,
            units=u.meter,
            expression="value > 0"
        ),
        spec.CSVInput(
            id="effect_size_csv",
            name=gettext("health effect size table"),
            about=gettext(
                "Table containing health indicator-specific effect sizes, "
                "such as risk ratios or odds ratios, representing the "
                "relationship between nature exposure and mental health "
                "outcomes."
            ),
            columns=[ # TODO: update these column names? or give options?
                spec.StringInput(
                    id="health_indicator",
                    about=gettext("Name of the health indicator"),
                    regexp=None
                ),
                spec.PercentInput(
                    id="exposure_metric_percent", #  TODO: format in ppt is 0.1NDVI eg....
                    about=gettext("Percent to multiply exposure metric by") #TODO fix this desc
                ),
                spec.StringInput(
                    id="exposure_metric_name",
                    about=gettext("Name of exposure data"), #  TODO fix this desc and reg exp/determine what this is
                    regexp=None
                ),
                spec.StringInput(
                    id="ratio_type",
                    about=gettext("Whether ratio provided is a risk ratio or odds ratio"),
                    regexp="risk|odds"  # TODO: fix this regexp?
                ),
                spec.RatioInput(
                    id="effect_size",
                    about=gettext("Effect size ratio")
                )
            ]
        ),
        spec.VectorInput(
            id="baseline_prevalence_vector",
            name=gettext("baseline prevalence"),
            about=gettext(
                "Baseline prevalence (or incidence) rates of specific mental "
                "health outcomes (e.g., depression or anxiety) across "
                "administrative units within the study area. This data allows "
                "the model to estimate preventable cases by comparing current "
                "rates with those projected under improved nature exposure "
                "scenarios."),
            geometry_types={"MULTIPOLYGON", "POLYGON"},
            fields=[
                spec.RatioInput(
                    id="risk_rate",  # health_risk_rate
                    about=gettext("Health risk rate")
                )
            ],
            projected=None
        ),
        spec.CSVInput(
            id="health_cost_rate_csv",
            name="health cost rate",
            about=gettext(
                "Table providing the societal cost per case (e.g., in USD "
                "PPP) for specific mental health outcomes. This data enables "
                "the model to estimate the economic value of preventable "
                "cases under different urban nature scenarios. Costs can be "
                "specified at national, regional, or local levels depending "
                "on data availability."
            ),
            columns=[
                spec.NumberInput(
                    id="cost_value",
                    about=gettext("Societal cost per case"),
                    units=None
                ),
                # TODO: add region/spatial unit input or change to number input
            ],
            required=False
        ),
        spec.OptionStringInput(
            id="scenario",
            name=gettext("scenario"),
            about=gettext(
                "Land use scenario options which incorporate the following "
                "rasters as inputs: (1) Tree Canopy Cover and baseline NDVI, "
                "(2) baseline and alternate Land Use/Land Cover, or (3) "
                "baseline and alternate NDVI."
            ),
            options=[
                spec.Option(key="", display_name=gettext("--Select a scenario--"))
            ] + SCENARIO_OPTIONS
        ),
        spec.PercentInput(
            id="tc_target",
            name="Tree Cover Target",
            about=gettext(
                "Target for tree canopy cover within the city or study area."
                "This value represents a desired scenario and will be used to "
                "compare against the baseline tree cover to estimate "
                "potential health benefits."
            ),
            required="scenario=='tcc_ndvi'",
            allowed="scenario=='tcc_ndvi'"
        ),
        spec.SingleBandRasterInput(
            id="tc_raster",
            name=gettext("Tree Canopy Cover (TCC)"),
            about=gettext(
                "Map of the percentage of pixel area covered by trees."
            ),
            data_type=float,
            units=None,
            projected=None,
            required="scenario=='tcc_ndvi'", #  TODO: check this is correct
            allowed="scenario=='tcc_ndvi'"
        ),
        spec.SingleBandRasterInput(
            id="ndvi_base",
            name=gettext("baseline NDVI"),
            about=gettext(
                "Map of NDVI under current or baseline conditions."
            ),
            data_type=float,
            units=None,
            projected=True,
            required="scenario!='lulc' or not lulc_attr_csv",
        ),
        spec.SingleBandRasterInput(
            id="ndvi_alt",
            name=gettext("alternate NDVI"),
            about=gettext(
                "Map of NDVI under future or counterfactual conditions."
            ),
            data_type=float,
            units=None,
            projected=True,
            required="scenario=='ndvi'", # TODO add allowed key for this and all keys below
            allowed="scenario=='ndvi'"
        ),
        spec.SingleBandRasterInput(
            id="lulc_base",
            name=gettext("baseline land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes under current or baseline "
                "conditions. Each land use/land cover type must be assigned "
                "a unique integer code. If an LULC attribute table is used, "
                "all values in this raster must have corresponding entries."
            ),
            data_type=int,
            units=None,
            projected=True,
            required="scenario=='lulc'",
            allowed="scenario=='lulc'"
        ),
        spec.SingleBandRasterInput(
            id="lulc_alt",
            name=gettext("alternate land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes under future or "
                "counterfactual conditions. Each land use/land cover type "
                "must be assigned a unique integer code. If an LULC attribute "
                "table is used, all values in this raster must have "
                "corresponding entries."
            ),
            data_type=int,
            units=None,
            projected=True,
            required="scenario=='lulc'",
            allowed="scenario=='lulc'"
        ),
        spec.CSVInput(
            id="lulc_attr_csv",
            name=gettext("LULC Attribute Table"),
            about=gettext(
                "A table mapping LULC codes to NDVI values and stating "
                "whether the LULC class should be exluded (0 for keeping, and "
                "1 for excluding)."
            ),
            columns=[
                spec.IntegerInput(
                    id="lucode",
                    about=gettext("LULC code")
                ),
                spec.BooleanInput(
                    id="exclude",
                    about=gettext(
                        "Whether to exclude the LULC class (e.g., if water) "
                        "or keep it")
                ),
                spec.NumberInput(
                    id="ndvi",
                    about=gettext("NDVI value"),
                    units=None,
                    # type=float
                )
            ],
            required="scenario=='lulc' and not ndvi_base",
            allowed="scenario=='lulc'"
        )
        ],
    outputs=[
            spec.SingleBandRasterOutput(
                id="preventable_cases",
                path="preventable_cases.tif",
                about=gettext("Preventable cases at pixel level"),
                data_type=int,
            ),
            spec.SingleBandRasterOutput(
                id="preventable_cost",
                path="preventable_cost.tif",
                about=gettext(
                    "Preventable cost at pixel level. The currency unit will "
                    "be the same as that in the health cost rate table."),
                data_type=int,
            ),
            spec.VectorOutput(
                id="preventable_cases_cost_sum_vector",
                path="preventable_cases_cost_sum.shp",
                about=gettext(
                    "Aggregated total preventable cases, and total "
                    "preventable costs by sub-region (e.g., census tract or "
                    "zip code) within the area of interest."
                ),
                fields=[
                    spec.IntegerOutput(
                        id="total_preventable_cases",
                        about=gettext("Aggregated total preventable cases"),
                        units=None
                    ),
                    spec.IntegerOutput(
                        id="preventable_costs_by_subregion",
                        about=gettext("Total preventable costs by subregion"),
                        units=None
                    )
                ]
            ),
            spec.CSVOutput(
                id="prevantable_cases_cost_sum_table",
                path="prevantable_cases_cost_sum.csv",
                about=gettext(
                    "Aggregated total preventable cases and total preventable "
                    "costs by sub-region (e.g., census tract or zip code) "
                    "within the area of interest, with an additional row "
                    "showing the total cases for the entire area (e.g., city)."
                ),
                columns=[
                    spec.IntegerOutput(
                        id="total_preventable_cases",
                        about=gettext("Aggregated total preventable cases"),
                    ),
                    spec.IntegerOutput(
                        id="preventable_costs_by_subregion",
                        about=gettext("Total preventable costs by subregion"),
                    ),
                    spec.IntegerOutput(
                        id="total_cases",
                        about=gettext(
                            "Total cases for the entire area (e.g., a city)."),
                    )
                ]
            )
        ]
)

# TODO: use FileRegistry
_OUTPUT_BASE_FILES = {
    "preventable_cases_path": "preventable_cases.tif",
    "preventable_cost_path": "preventable_cost.tif",
    "preventable_cases_cost_sum_vector_path":"preventable_cases_cost_sum.shp",
    "prevantable_cases_cost_sum_table_path":"prevantable_cases_cost_sum.csv",
}

_INTERMEDIATE_BASE_FILES = {
    "tc_aligned": "tc_aligned.tif",
    "lulc_base_aligned": "lulc_base_aligned.tif",
    "lulc_alt_aligned": "lulc_alt_aligned.tif",
    "ndvi_base_aligned": "ndvi_base_aligned.tif",
    "ndvi_alt_aligned": "ndvi_alt_aligned.tif",
    "population_aligned": "population_aligned.tif",
    "kernel": "kernel.tif",
    "ndvi_base_buffer_mean": "ndvi_base_buffer_mean.tif", #mean NDVI within a circular buffer
    "ndvi_alt_buffer_mean": "ndvi_alt_buffer_mean.tif",
    "delta_ndvi": "delta_ndvi.tif",
    "baseline_prevalence_raster": "baseline_prevalence.tif",
    "baseline_cases": "baseline_cases.tif"
}

def execute(args):
    """Urban Mental Health.

    The model estimates the impacts of nature exposure, and more specifically
    residential greenness, on mental health. Residential nature exposure is
    defined as the average NDVI within a distance of a residence that benefits
    human mental health. The mental health model calculates the preventable
    mental disorder cases at the pixel level, based on the selected urban
    greening scenario.

    Args:
        args['workspace_dir'] (str): (required) a path to the directory that
            will write output and other temporary files during calculation.
        args['results_suffix'] (str): (optional) appended to any output
            filename.
        args['aoi_vector_path'] (str): (required) path to a polygon vector
            that is projected in a coordinate system with units of meters.
            The polygon should intersect the baseline prevalence vector and
            the population raster.
        args['population_raster] (str): (required) a path to a raster of
            gridded population data representing the number of people per pixel,
            ideally at the pixel level.
        args['search_radius'] (float): (required) the distance used to define
            the surrounding area of a person's residence that best represents
            daily exposure to nearby nature.
        args['effect_size_csv'] (str): (required) a path to a CSV table
            containing health indicator-specific effect sizes, such as risk
            ratios or odds ratios, representing the relationship between nature
            exposure and mental health outcomes.
        args['baseline_prevalence_vector'] (str): (required) a path to a
            vector providing the baseline prevalence (or incidence) rate of
            a specific mental health outcome (e.g., depression or anxiety)
            across administrative units within the study area. This data allows
            the model to estimate preventable cases by comparing current rates
            with those projected under improved nature exposure scenarios. The
            vector must contain field `risk_rate`.
        args['health_cost_rate_csv'] (str): (optional) a path to a CSV table
            providing the societal cost per case (e.g., in USD PPP) for the
            mental health outcome described by the `baseline_prevalence_vector`.
            This data enables the model to estimate the economic value of
            preventable cases under different urban nature scenarios. Costs can
            be specified at national, regional, or local levels depending on
            data availability.
        args['scenario'] (str): (required) which of the three land use scenarios
            to model.
        args['tc_raster'] (str): required if args['scenario'] == 'tc_ndvi',
            a path to a raster providing tree canopy cover under current or
            baseline conditions.
        args['tc_target'] (float): required if args['scenario'] == 'tc_ndvi',
            user-defined target for tree canopy cover within area of interest.
            This value represents a desired scenario and will be used to
            compare against the baseline tree cover to estimate potential
            health benefits.
        args['ndvi_base'] (str): required if args['scenario'] != 'lulc' or
            not args['lulc_attr_csv'], a path to a Normalized Difference Vegetation
            Index raster representing current or baseline conditions, which gives
            the greenness of vegetation in a given cell.
        args['ndvi_alt'] (str): required if args['scenario'] == 'ndvi', a path
            to an NDVI raster under future or counterfactual conditions.
        args['lulc_base'] (str): required if args['scenario'] == 'lulc', a path
            to a Land Use/Land Cover raster showing current or baseline conditions
        args['lulc_alt'] (str): required if args['scenario'] == 'lulc', a path
            to a Land Use/Land Cover raster showing a future or counterfactural
            scenario.
        args['lulc_attr_csv'] (str): required if args['scenario'] == 'lulc' and
            not args['ndvi_base'], a path to a CSV table that maps LULC codes to
            corresponding NDVI values and specifies whether to exclude the LULC
            class from analysis. The following fields are required: 
            'lucode' (int): unique LULC class identifier
            'ndvi' (float): NDVI value of the LULC class
            'exclude' (bool): 0 for keeping, 1 for excluding the LULC class

    """
    LOGGER.info("Test inputs")
    effect_size_df = MODEL_SPEC.get_input(
        'effect_size_csv').get_validated_dataframe(
        args['effect_size_csv'])
    
    if args['health_cost_rate_csv']:
        health_cost_rate_df = MODEL_SPEC.get_input(
            'health_cost_rate_csv').get_validated_dataframe(
            args['health_cost_rate_csv'])
        
    search_radius = float(args['search_radius'])

    LOGGER.info("Make directories")
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate_outputs')
    output_dir = args['workspace_dir']
    utils.make_directories([intermediate_output_dir, output_dir])

    LOGGER.info('Building file registry')
    file_registry = utils.build_file_registry(
        [(_OUTPUT_BASE_FILES, output_dir),
         (_INTERMEDIATE_BASE_FILES, intermediate_output_dir)], file_suffix)
    
    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    LOGGER.debug('n_workers: %s', n_workers)
    task_graph = taskgraph.TaskGraph(
        os.path.join(args['workspace_dir'], 'taskgraph_cache'),
        n_workers, reporting_interval=5)

    # preprocessing
    LOGGER.info("Start preprocessing")
    if args['scenario'] == 'ndvi':
        LOGGER.info("Using scenario option 3: NDVI")
        base_ndvi_raster_info = pygeoprocessing.get_raster_info(
            args['ndvi_base'])
        pixel_size = base_ndvi_raster_info['pixel_size']
        target_projection = base_ndvi_raster_info['projection_wkt']
        # bounding boxes.
        pop_raster_info = pygeoprocessing.get_raster_info(
            args['population_raster'])
        aoi_info = pygeoprocessing.get_vector_info(args['aoi_vector_path'])
        target_bounding_box = pygeoprocessing.merge_bounding_box_list(
            [aoi_info['bounding_box'], base_ndvi_raster_info['bounding_box'],
             pop_raster_info['bounding_box']], 'intersection')
        
        input_align_list = [args['ndvi_base'], args['ndvi_alt']]
        output_align_list = [file_registry['ndvi_base_aligned'],
                             file_registry['ndvi_alt_aligned']]

        ndvi_align_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            args=(input_align_list, output_align_list,
                ['cubicspline', 'cubicspline'],
                pixel_size,
                target_bounding_box),
            kwargs={
                'raster_align_index': 0,  # align to base_ndvi
                'target_projection_wkt': target_projection},
            target_path_list=output_align_list,
            task_name='align NDVI rasters')
        
        population_align_task = task_graph.add_task(
            func=_resample_population_raster,
            kwargs={
                'source_population_raster_path': args['population_raster'],
                'target_population_raster_path': file_registry[
                    'population_aligned'],
                'target_pixel_size': pixel_size,
                'target_bb': target_bounding_box,
                'target_projection_wkt': target_projection,
                'working_dir': intermediate_output_dir,
            },
            target_path_list=[file_registry['population_aligned']],
            task_name='Resample population to NDVI resolution')

        pixel_radius = int(round(search_radius/pixel_size[0]))
        LOGGER.info(f"Search radius {search_radius} results in buffer of {pixel_radius} pixels")
        kernel_task = task_graph.add_task(
            func=pygeoprocessing.kernels.dichotomous_kernel,
            kwargs={
                'target_kernel_path': file_registry['kernel'],
                'max_distance': pixel_radius,
                'normalize': True},
            target_path_list=[file_registry['kernel']],
            task_name=f'create kernel raster',)

        mean_buffer_base_ndvi_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (file_registry['ndvi_base_aligned'], 1), (file_registry['kernel'], 1),
                file_registry['ndvi_base_buffer_mean']),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': True,
                'target_datatype': gdal.GDT_Float32,
                'target_nodata': FLOAT32_NODATA},
            dependent_task_list=[ndvi_align_task, kernel_task],
            target_path_list=[file_registry['ndvi_base_buffer_mean']],
            task_name="calculate mean baseline NDVI within buffer")
        
        mean_buffer_alt_ndvi_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (file_registry['ndvi_alt_aligned'], 1), (file_registry['kernel'], 1),
                file_registry['ndvi_alt_buffer_mean']),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': True,
                'target_datatype':gdal.GDT_Float32,
                'target_nodata': FLOAT32_NODATA},
            dependent_task_list=[ndvi_align_task, kernel_task], #TODO: should this task depend on mean_buffer_base_ndvi_task in case both try to open kernel raster simultaneously?
            target_path_list=[file_registry['ndvi_alt_buffer_mean']],
            task_name="calculate mean alternate NDVI within buffer")
        
        # TODO mask out water
        delta_ndvi_task = task_graph.add_task(
            func=calc_delta_ndvi,
            args=(file_registry['ndvi_base_buffer_mean'],
                  file_registry['ndvi_alt_buffer_mean'],
                  file_registry['delta_ndvi']),
            target_path_list=[file_registry['delta_ndvi']],
            dependent_task_list=[mean_buffer_base_ndvi_task, mean_buffer_alt_ndvi_task],
            task_name="calculate delta ndvi"
        )

        baseline_cases_task = task_graph.add_task(
            func=calc_baseline_cases,
            args=(file_registry['population_aligned'],
                  args['baseline_prevalence_vector'],
                  file_registry['baseline_prevalence_raster'],
                  file_registry['baseline_cases']),
            target_path_list=[file_registry['baseline_cases']],
            dependent_task_list=[population_align_task],
            task_name="calculate baseline cases"
        )

        preventable_cases_task = task_graph.add_task(
            func=calc_preventable_cases,
            args=(file_registry['delta_ndvi'],
                  file_registry['baseline_cases'],
                  args['effect_size_csv'],
                  file_registry['preventable_cases_path']),
            target_path_list=[file_registry['preventable_cases_path']],
            dependent_task_list=[delta_ndvi_task, baseline_cases_task],
            task_name="calculate preventable cases"
        )


    elif args['scenario'] == 'tc_ndvi':
        tc_target = float(args['tc_target'])

    return True

def _resample_population_raster(
        source_population_raster_path, target_population_raster_path,
        target_pixel_size, target_bb, target_projection_wkt, working_dir):
    """Resample a population raster without losing or gaining people.

    Population rasters are an interesting special case where the data are
    neither continuous nor categorical, and the total population count
    typically matters.  Common resampling methods for continuous
    (interpolation) and categorical (nearest-neighbor) datasets leave room for
    the total population of a resampled raster to significantly change.  This
    function resamples a population raster with the following steps:

        1. Convert a population count raster to population density per pixel
        2. Warp the population density raster to the target spatial reference
           and pixel size using bilinear interpolation.
        3. Convert the warped density raster back to population counts.

    This function is pulled from urban_nature_access.

    Args:
        source_population_raster_path (string): The source population raster.
            Pixel values represent the number of people occupying the pixel.
            Must be linearly projected in meters.
        target_population_raster_path (string): The path to where the target,
            warped population raster will live on disk.
        target_pixel_size (tuple): A tuple of the pixel size for the target
            raster.  Passed directly to ``pygeoprocessing.warp_raster``.
        target_bb (tuple): A tuple of the bounding box for the target raster.
            Passed directly to ``pygeoprocessing.warp_raster``.
        target_projection_wkt (string): The Well-Known Text of the target
            spatial reference fro the target raster.  Passed directly to
            ``pygeoprocessing.warp_raster``.  Assumed to be a linear projection
            in meters.
        working_dir (string): The path to a directory on disk.  A new directory
            is created within this directory for the storage of temporary files
            and then deleted upon successful completion of the function.

    Returns:
        ``None``
    """
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    tmp_working_dir = tempfile.mkdtemp(dir=working_dir)
    population_raster_info = pygeoprocessing.get_raster_info(
        source_population_raster_path)
    pixel_area = numpy.multiply(*population_raster_info['pixel_size'])
    population_srs = osr.SpatialReference()
    population_srs.ImportFromWkt(population_raster_info['projection_wkt'])

    # Convert population pixel area to square km
    population_pixel_area = (
        pixel_area * population_srs.GetLinearUnits()) / 1e6

    def _convert_population_to_density(population):
        """Convert population counts to population per square km.

        Args:
            population (numpy.array): A numpy array where pixel values
                represent the number of people who reside in a pixel.

        Returns:
            """
        return population / population_pixel_area

    # Step 1: convert the population raster to population density per sq. km
    density_raster_path = os.path.join(tmp_working_dir, 'pop_density.tif')
    pygeoprocessing.raster_map(
        rasters=[source_population_raster_path],
        op=_convert_population_to_density,
        target_path=density_raster_path,
        target_dtype=numpy.float32)

    # Step 2: align to the LULC
    warped_density_path = os.path.join(tmp_working_dir, 'warped_density.tif')
    pygeoprocessing.warp_raster(
        density_raster_path,
        target_pixel_size=target_pixel_size,
        target_raster_path=warped_density_path,
        resample_method='bilinear',
        target_bb=target_bb,
        target_projection_wkt=target_projection_wkt)

    # Step 3: convert the warped population raster back from density to the
    # population per pixel
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(target_projection_wkt)
    # Calculate target pixel area in km to match above
    target_pixel_area = (
        numpy.multiply(*target_pixel_size) * target_srs.GetLinearUnits()) / 1e6

    def _convert_density_to_population(density):
        """Convert a population density raster back to population counts.

        Args:
            density (numpy.array): An array of the population density per
                square kilometer.

        Returns:
            A ``numpy.array`` of the population counts given the target pixel
            size of the output raster."""
        # We're using a float32 array here because doing these unit
        # conversions is likely to end up with partial people spread out
        # between multiple pixels.  So it's preserving an unrealistic degree of
        # precision, but that's probably OK because pixels are imprecise
        # measures anyways.
        return density * target_pixel_area

    pygeoprocessing.raster_map(
        op=_convert_density_to_population,
        rasters=[warped_density_path],
        target_path=target_population_raster_path)

    shutil.rmtree(tmp_working_dir, ignore_errors=True)



def calc_delta_ndvi(base_ndvi, alt_ndvi, target_path):
    """Calculate the change in nature exposure (NE)
    
    Args:
        base_ndvi (str): path to baseline NDVI raster 
        alt_ndvi (str): path to future or counterfactual NDVI raster
        target_path (str): path to output delta NDVI raster

    Returns:
        None.
    
    """
    def _subtract_op(base_ndvi, alt_ndvi, base_nodata, alt_nodata):
        """operation to subtract alt ndvi from base ndvi and mask nodata"""
        mask = pygeoprocessing.array_equals_nodata(base_ndvi, base_nodata) | pygeoprocessing.array_equals_nodata(alt_ndvi, alt_nodata)
        delta_ndvi = alt_ndvi - base_ndvi
        delta_ndvi[mask] = FLOAT32_NODATA
        
        return delta_ndvi

    base_nodata = pygeoprocessing.get_raster_info(base_ndvi)['nodata']
    alt_nodata = pygeoprocessing.get_raster_info(alt_ndvi)['nodata']
    base_raster_path_band_const_list = [(base_ndvi, 1), (alt_ndvi, 1),
                                        (base_nodata, "raw"), (alt_nodata, "raw")]
   
    pygeoprocessing.raster_calculator(
        base_raster_path_band_const_list, _subtract_op, target_path,
        gdal.GDT_Float32, nodata_target=FLOAT32_NODATA)


def calc_baseline_cases(population_raster, base_prevalence_vector,
                        target_base_prevalence_raster, target_base_cases):
    """Calculate baseline cases via incidence_rate * population

    Args:
        population_raster (str): path to aligned population raster
            representing the number of inhabitants per pixel across
            the study area. Pixels with no population should be
            assigned a value of 0. 
        base_prevalence_vector (str): path to vector with field # TODO: check units/intended range of rate?
            `risk_rate` that provides the baseline prevalence
            (or incidence) rate of a mental health outcome by
            spatial unit (e.g., census tract).
        target_base_prevalence_raster (str): target output path for
            rasterized baseline prevalence
        target_base_cases (str): target output path for baseline
            cases raster
        
    Returns:
        None.

    """
    def _multiply_op(prevalence, pop, prevalence_nodata, pop_nodata):
        """Mulitply baseline prevalence raster (float) by population raster (int)"""
        valid_mask = (~pygeoprocessing.array_equals_nodata(prevalence, prevalence_nodata) &
                      ~pygeoprocessing.array_equals_nodata(pop, pop_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = FLOAT32_NODATA
        result[valid_mask] = prevalence[valid_mask] * pop[valid_mask]
        return result

    pygeoprocessing.new_raster_from_base(
        population_raster, target_base_prevalence_raster,
        gdal.GDT_Float32, [FLOAT32_NODATA])
    
    pygeoprocessing.rasterize(base_prevalence_vector,
                              target_base_prevalence_raster,
                              option_list=[
                                  "ATTRIBUTE=risk_rate", "ALL_TOUCHED=TRUE",
                                  "MERGE_ALG=REPLACE"])
    
    # don't need to dynamically get target_base_prevalence_raster's nodata as we defined it above
    population_nodata = pygeoprocessing.get_raster_info(population_raster)['nodata']
    base_raster_path_band_const_list = [(target_base_prevalence_raster, 1),
                                        (population_raster, 1),
                                        (FLOAT32_NODATA, "raw"),
                                        (population_nodata, "raw")]
    pygeoprocessing.raster_calculator(
        base_raster_path_band_const_list, _multiply_op,
        target_base_cases, gdal.GDT_Float32, nodata_target=FLOAT32_NODATA)


def calc_preventable_cases(delta_ndvi, baseline_cases, health_effect_table,
                           target_preventable_cases):
    """Calculate preventable cases

    PC = PF * BC
    PF = 1 - RR
    RR=exp(ln(RR_0.1NE) * 10 * delta_NE)

    PC = (1 - exp(ln(RR_0.1NE) * 10 * delta_NE))*BC

    PC: preventable cases
    PF: preventable fraction
    BC: baseline cases
    RR: relative risk or risk ratio
    NE: nature exposure, as approimated by NDVI
    RR0.1NE: RR per 0.1 NE increase.
        By default, the model uses the relative risk associated with a 
        0.1 increase in NDVI-based nature exposure. This value must be 
        provided by users in the effect size table, based on empirical 
        studies or meta-analyses relevant to the selected health outcome. 

    Args:
        delta_ndvi (str): path to raster representing change in NDVI from
            baseline to alternate/counterfactural scenario
        baseline_cases (str): path to raster of number of baseline cases 
        health_effect_table (str): path to health effect table containing
            'risk_ratio' column
        target_preventable_cases (str): path to output preventable cases raster
    
    Returns:
        None.
        
    """
    def _preventable_cases_op(delta_ndvi, baseline_cases, effect_size_val):
        valid_mask = (~pygeoprocessing.array_equals_nodata(delta_ndvi, FLOAT32_NODATA) &
                      ~pygeoprocessing.array_equals_nodata(baseline_cases, FLOAT32_NODATA))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = FLOAT32_NODATA
        result[valid_mask] = 1 - numpy.exp(numpy.log(effect_size_val) * 10 * delta_ndvi[valid_mask])
        result[valid_mask] *= baseline_cases[valid_mask] #preventable cases
        return result
    
    # TODO: determine if risk_ratio can just be entered as single value?
    health_effect_df = pandas.read_csv(health_effect_table)
    effect_size_val = health_effect_df["effect_size"][0] #previously called 'rr0'?
    LOGGER.info(f"effect_size_val {effect_size_val}")

    base_raster_path_band_const_list = [(delta_ndvi, 1),
                                        (baseline_cases, 1),
                                        (effect_size_val, "raw")]

    pygeoprocessing.raster_calculator(
        base_raster_path_band_const_list, _preventable_cases_op,
        target_preventable_cases, gdal.GDT_Float32, nodata_target=FLOAT32_NODATA)

def calc_preventable_cost(preventable_cases, health_cost_rate):
    """Calculate preventable cost"""

    preventable_cost = preventable_cases*health_cost_rate
    return preventable_cost

@validation.invest_validator
def validate(args, limit_to=None):
    """Validate args to ensure they conform to `execute`'s contract.

    Args:
        args (dict): dictionary of key(str)/value pairs where keys and
            values are specified in `execute` docstring.
        limit_to (str): (optional) if not None indicates that validation
            should only occur on the args[limit_to] value. The intent that
            individual key validation could be significantly less expensive
            than validating the entire `args` dictionary.

    Returns:
        list of ([invalid key_a, invalid_keyb, ...], 'warning/error message')
            tuples. Where an entry indicates that the invalid keys caused
            the error message in the second part of the tuple. This should
            be an empty list if validation succeeds.
    """

    validation_warnings = validation.validate(args, MODEL_SPEC)

    error_msg = spec.OptionStringInput(
        id='', options=SCENARIO_OPTIONS).validate(args['scenario'])
    if error_msg:
        validation_warnings.append((['scenario'], "Must select a scenario."))

    return validation_warnings
