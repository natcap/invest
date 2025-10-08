"""InVEST Urban Mental Health model."""
import logging
import os
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
from .file_registry import FileRegistry

LOGGER = logging.getLogger(__name__)

SCENARIO_OPTIONS = [
    spec.Option(key="tcc_ndvi", display_name="TCC & Baseline NDVI"),
    spec.Option(key="lulc", display_name="Baseline & Alternate LULC"),
    spec.Option(key="ndvi", display_name="Baseline & Alternate NDVI")
    ]

MODEL_SPEC = spec.ModelSpec(
    model_id="urban_mental_health",
    model_title=gettext("Urban Mental Health"),
    userguide="",  # TODO - add this model to UG
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["aoi_vector_path", "population_raster", "search_radius"],
        ["effect_size", "baseline_prevalence_vector",
         "health_cost_rate"],
        ["scenario"],
        ["ndvi_base", "ndvi_alt"],
        ["lulc_base", "lulc_alt", "lulc_attr_csv"],
        ["tc_raster", "tc_target"]
    ],
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("umh",),
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.AOI.model_copy(update=dict(  #  TODO: potentially want to have this req to be census tract pop. shp? or only req that if opt 1 but optional for opts 2-3?
            id="aoi_vector_path",
            about=gettext(
                "Map of the area over which to run the model. The AOI must be "
                "smaller than the raster inputs by at least the search radius "
                "to ensure correct edge pixel calculation, as InVEST will "
                "buffer your AOI by the `search_radius` to determine the "
                "processing area. Final outputs will be clipped to this AOI."),
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
            units=u.people,
            projected=True,
            projection_units=u.meter
        ),
        spec.NumberInput(
            id="search_radius",
            name=gettext("search radius"),
            about=gettext(
                "Distance used to define the surrounding area of a person's "
                "residence that best represents daily exposure to nearby "
                "nature."),
            units=u.meter,
            expression="value > 0"
        ),
        spec.NumberInput(
            id="effect_size",
            name=gettext("health effect size"),
            about=gettext(
                "Health indicator-specific effect size representing the "
                "relationship between nature exposure and the mental health "
                "outcome, given as relative risk associated with a 0.1 "
                "increase in NDVI. If the user has an effect size value "
                "as an odds ratio, see User's Guide."
            ),
            units=None, #todo: check
            expression="value > 0"
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
        spec.NumberInput(
            id="health_cost_rate",
            name="health cost rate",
            about=gettext(
                "Societal cost (e.g., in USD PPP) per case for the mental "
                "health outcome. This data enables the model to estimate "
                "the economic value of preventable cases under different "
                "urban nature scenarios. Cost can be specified at national, "
                "regional, or local levels depending on data availability."
            ),
            units=u.currency,
            expression="value > 0",
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
                spec.Option(key="",
                            display_name=gettext("--Select a scenario--"))
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
                "Map of the percentage of pixel area covered by trees. "
                "This raster should extend beyond the AOI by at "
                "least the search radius distance"
            ),
            data_type=float,
            units=None,
            projected=None,
            required="scenario=='tcc_ndvi'",
            allowed="scenario=='tcc_ndvi'"
        ),
        spec.SingleBandRasterInput(
            id="ndvi_base",
            name=gettext("baseline NDVI"),
            about=gettext(
                "Map of NDVI under current or baseline conditions. "
                "This raster should extend beyond the AOI by at "
                "least the search radius distance"
            ),
            data_type=float,
            units=None,
            projected=True,
            # require ndvi_base unless scenario is `lulc`` and user enters
            # an attribute table
            required="scenario!='lulc' or not lulc_attr_csv",
        ),
        spec.SingleBandRasterInput(
            id="ndvi_alt",
            name=gettext("alternate NDVI"),
            about=gettext(
                "Map of NDVI under future or counterfactual conditions. "
                "This raster should extend beyond the AOI by at "
                "least the search radius distance"
            ),
            data_type=float,
            units=None,
            projected=True,
            required="scenario=='ndvi'",
            allowed="scenario=='ndvi'"
        ),
        spec.SingleBandRasterInput(
            id="lulc_base",
            name=gettext("baseline land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes under current or baseline "
                "conditions. If scenario is not 'lulc', this is used only for "
                "water masking. If an LULC attribute table is used, "
                "all values in this raster must have corresponding entries. "
                "This raster should extend beyond the AOI by at least the "
                "search radius distance."
            ),
            data_type=int,
            units=None,
            projected=True,
            required="scenario=='lulc' or lulc_attr_csv",
        ),
        spec.SingleBandRasterInput(
            id="lulc_alt",
            name=gettext("alternate land use/land cover"),
            about=gettext(
                "Map of land use/land cover codes under future or "
                "counterfactual conditions. Each land use/land cover type "
                "must be assigned a unique integer code. If an LULC attribute "
                "table is used, all values in this raster must have "
                "corresponding entries. This raster should extend beyond the "
                "AOI by at least the search radius distance"
            ),
            data_type=int,
            units=None,
            projected=True,
            required="scenario=='lulc'",
            allowed="scenario=='lulc' or lulc_base"
        ),
        spec.CSVInput(
            id="lulc_attr_csv",
            name=gettext("LULC Attribute Table"),
            about=gettext(
                "A table mapping unique LULC codes to NDVI values and stating "
                "whether the LULC class should be excluded (0 for keeping, "
                "and 1 for excluding). Typically, water is excluded and other "
                "land cover types are kept. NDVI values are only required if "
                "scenario is 'lulc' and a base NDVI raster is not provided."
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
                    required="scenario=='lulc' and not ndvi_base"
                )
            ],
            # Attr table required if scenario is lulc and ndvi baseline raster
            # is not provided or scenario isnt lulc and baseline raster is
            # provided so attribute table is needed for water masking
            required="(scenario=='lulc' and not ndvi_base) or "
                     "(scenario!='lulc' and lulc_base)",
            allowed="scenario=='lulc or 'lulc_base'"
        )
        ],
    outputs=[
            spec.SingleBandRasterOutput(
                id="preventable_cases",
                path="output/preventable_cases.tif",
                about=gettext("Preventable cases at pixel level"),
                data_type=float,
                units=u.count,
            ),
            spec.SingleBandRasterOutput(
                id="preventable_cost",
                path="output/preventable_cost.tif",
                about=gettext(
                    "Preventable cost at pixel level. The currency unit will "
                    "be the same as that in the health cost rate input."),
                data_type=float,
                units=u.currency,
                created_if="health_cost_rate"
            ),
            spec.VectorOutput(
                id="preventable_cases_cost_sum_vector",
                path="output/preventable_cases_cost_sum.gpkg",
                about=gettext(
                    "Aggregated total preventable cases, and total "
                    "preventable costs by sub-region (e.g., census tract or "
                    "zip code) within the area of interest."
                ),
                fields=[
                    spec.IntegerOutput(
                        id="sum_cases",
                        about=gettext(
                            "Aggregated total preventable cases by polygon"),
                        units=u.count
                    ),
                    spec.IntegerOutput(
                        id="sum_cost",
                        about=gettext(
                            "Total preventable costs by subregion/polygon in "
                            "same currency as input health cost rate"),
                        units=u.currency,
                        created_if="health_cost_rate"
                    )
                ]
            ),
            spec.CSVOutput(
                id="preventable_cases_cost_sum_table",
                path="output/preventable_cases_cost_sum.csv",
                about=gettext(
                    "Aggregated total preventable cases and total preventable "
                    "costs by sub-region (e.g., census tract or zip code) "
                    "within the area of interest, with an additional row "
                    "showing the total cases for the entire area (e.g., city). "
                    "Cost units are same as input health_cost_rate."
                ),
                columns=[
                    spec.NumberOutput(
                        id="sum_cases",
                        about=gettext("Aggregated total preventable cases"),
                        units=u.count
                    ),
                    spec.NumberOutput(
                        id="sum_cost",
                        about=gettext("Total preventable costs by subregion"),
                        units=u.currency,
                        created_if="health_cost_rate"
                    ),
                    spec.NumberOutput(
                        id="total_cases",
                        about=gettext(
                            "Total cases for the entire AOI"),
                        units=u.count
                    ),
                    spec.NumberOutput(
                        id="total_cost",
                        about=gettext(
                            "Total cost for the entire AOI"),
                        units=u.count,
                        created_if="health_cost_rate"
                    )
                ]
            ),
            spec.SingleBandRasterOutput(
                id="baseline_cases",
                path="intermediate/baseline_cases.tif",
                about=gettext("Baseline cases raster"),
                data_type=float,
                units=u.count
            ),
            spec.SingleBandRasterOutput(
                id="baseline_prevalence_raster",
                path="intermediate/baseline_prevalence.tif",
                about=gettext("Baseline prevalence raster"),
                data_type=float,
                units=None
            ),
            spec.SingleBandRasterOutput(
                id="delta_ndvi",
                path="intermediate/delta_ndvi.tif",
                about=gettext(
                    "Difference between baseline and alternate NDVI raster"),
                data_type=float,
                units=None
            ),
            spec.SingleBandRasterOutput(
                id="kernel",
                path="intermediate/kernel.tif",
                about=gettext(
                    "Normalized dichotomous kernel raster with radius "
                    "search_radius"),
                data_type=float,
                units=None
            ),
            spec.SingleBandRasterOutput(
                id="lulc_base_aligned",
                path="intermediate/lulc_base_aligned.tif",
                about=gettext("Aligned and resampled baseline LULC raster"),
                data_type=int,
                units=None,
                created_if="lulc_base"
            ),
            spec.SingleBandRasterOutput(
                id="lulc_alt_aligned",
                path="intermediate/lulc_alt_aligned.tif",
                about=gettext("Aligned and resampled alternate LULC raster"),
                data_type=int,
                units=None,
                created_if="lulc_alt"
            ),
            spec.SingleBandRasterOutput(
                id="lulc_mask",
                path="intermediate/lulc_mask.tif",
                about=gettext(
                    "Binary mask based on LULC raster where 1 indicates "
                    "pixels to be masked out based on `exclude` field in the "
                    "LULC attribute table. This is used to mask the input "
                    "rasters, e.g., baseline and alternate NDVI if "
                    "scenario is `ndvi`"),
                data_type=int,
                units=None,
                created_if="lulc_base and lulc_attr_csv"
            ),
            spec.SingleBandRasterOutput(
                id="ndvi_alt_aligned",
                path="intermediate/ndvi_alt_aligned.tif",
                about=gettext("Aligned and resampled alternate NDVI raster"),
                data_type=float,
                units=None,
                created_if="ndvi_alt"
            ),
            spec.SingleBandRasterOutput(
                id="ndvi_alt_aligned_masked",
                path="intermediate/ndvi_alt_aligned_masked.tif",
                about=gettext(
                    "Masked aligned and resampled alternate NDVI raster"),
                data_type=float,
                units=None,
                created_if="ndvi_alt"
            ),
            spec.SingleBandRasterOutput(
                id="ndvi_base_aligned",
                path="intermediate/ndvi_base_aligned.tif",
                about=gettext("Aligned and resampled baseline NDVI raster"),
                data_type=float,
                units=None,
                created_if="ndvi_base"
            ),
            spec.SingleBandRasterOutput(
                id="ndvi_base_aligned_masked",
                path="intermediate/ndvi_base_aligned_masked.tif",
                about=gettext(
                    "Masked aligned and resampled baseline NDVI raster"),
                data_type=float,
                units=None,
                created_if="ndvi_base"
            ),
            spec.SingleBandRasterOutput(
                id="ndvi_alt_buffer_mean",
                path="intermediate/ndvi_alt_buffer_mean.tif",
                about=gettext(
                    "Alternate NDVI raster convolved with a mean circular "
                    "kernel of radius search_radius"),
                data_type=float,
                units=None,
                created_if="ndvi_alt"
            ),
            spec.SingleBandRasterOutput(
                id="ndvi_base_buffer_mean",
                path="intermediate/ndvi_base_buffer_mean.tif",
                about=gettext(
                    "Baseline NDVI raster convolved with a mean circular "
                    "kernel of radius search_radius"),
                data_type=float,
                units=None,
                created_if="ndvi_base"
            ),
            spec.SingleBandRasterOutput(
                id="population_aligned",
                path="intermediate/population_aligned.tif",
                about=gettext("Aligned and resampled population raster"),
                data_type=float,
                units=u.people
            ),
            spec.TASKGRAPH_CACHE
        ]
)


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
            will write output, intermediate, and other temporary files during
            calculation.
        args['results_suffix'] (str): (optional) appended to any output
            filename.
        args['aoi_vector_path'] (str): (required) path to a polygon vector
            that is projected in a coordinate system with units of meters.
            The polygon should intersect the baseline prevalence vector and
            the population raster.
        args['population_raster] (str): (required) a path to a raster of
            gridded population data representing the number of people per
            pixel.
        args['search_radius'] (float): (required) the distance used to define
            the surrounding area of a person's residence that best represents
            daily exposure to nearby nature.
        args['effect_size'] (float): (required) Health indicator-specific
            effect size representing the relationship between nature
            exposure and the mental health outcome, given as relative
            risk associated with a 0.1 increase in NDVI.
        args['baseline_prevalence_vector'] (str): (required) a path to a
            vector providing the baseline prevalence (or incidence) rate of
            a specific mental health outcome (e.g., depression or anxiety)
            across administrative units within the study area. This data allows
            the model to estimate preventable cases by comparing current rates
            with those projected under improved nature exposure scenarios. The
            vector must contain field `risk_rate`.
        args['health_cost_rate'] (float): (optional) the societal cost per case
            (e.g., in USD PPP) for the mental health outcome described by
            the `baseline_prevalence_vector`. This data enables the model
            to estimate the economic value of preventable cases under
            different urban nature scenarios. Costs can be specified at
            national, regional, or local levels depending on data availability.
        args['scenario'] (str): (required) which of the three land use
            scenarios to model.
        args['tc_raster'] (str): required if args['scenario'] == 'tc_ndvi',
            a path to a raster providing tree canopy cover under current or
            baseline conditions.
        args['tc_target'] (float): required if args['scenario'] == 'tc_ndvi',
            user-defined target for tree canopy cover within area of interest.
            This value represents a desired scenario and will be used to
            compare against the baseline tree cover to estimate potential
            health benefits.
        args['ndvi_base'] (str): required if args['scenario'] != 'lulc' or
            not args['lulc_attr_csv'], a path to a Normalized Difference
            Vegetation Index raster representing current or baseline
            conditions, which gives the greenness of vegetation in a given
            cell.
        args['ndvi_alt'] (str): required if args['scenario'] == 'ndvi', a path
            to an NDVI raster under future or counterfactual conditions.
        args['lulc_base'] (str): required if args['scenario'] == 'lulc', a path
            to a Land Use/Land Cover raster showing current or baseline
            conditions
        args['lulc_alt'] (str): required if args['scenario'] == 'lulc', a path
            to a Land Use/Land Cover raster showing a future or counterfactual
            scenario.
        args['lulc_attr_csv'] (str): required if args['scenario'] == 'lulc' and
            not args['ndvi_base'], a path to a CSV table that maps LULC codes
            to corresponding NDVI values and specifies whether to exclude the
            LULC class from analysis. The following fields are required:
                'lucode' (int): unique LULC class identifier
                'ndvi' (float): NDVI value of the LULC class
                'exclude' (bool): 0 for keeping, 1 for excluding the LULC class

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths

    """
    LOGGER.info("Starting Urban Mental Health Model")
    search_radius = float(args['search_radius'])

    LOGGER.info("Making directories")
    file_suffix = utils.make_suffix_string(args, 'results_suffix')
    intermediate_output_dir = os.path.join(
        args['workspace_dir'], 'intermediate')
    output_dir = args['workspace_dir']
    utils.make_directories([intermediate_output_dir, output_dir])
    file_registry = FileRegistry(MODEL_SPEC.outputs, args['workspace_dir'],
                                 file_suffix)

    try:
        n_workers = int(args['n_workers'])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    LOGGER.debug('n_workers: %s', n_workers)
    task_graph = taskgraph.TaskGraph(
        file_registry['taskgraph_cache'], n_workers, reporting_interval=5)

    # preprocessing
    LOGGER.info("Start preprocessing")
    if args['scenario'] == 'ndvi':
        # TODO rearrage whats in if/else block when implementing scenarios 1-2
        LOGGER.info("Using scenario option 3: NDVI")
        base_ndvi_raster_info = pygeoprocessing.get_raster_info(
            args['ndvi_base'])
        pixel_size = base_ndvi_raster_info['pixel_size']
        target_projection = base_ndvi_raster_info['projection_wkt']
        ndvi_bbox = base_ndvi_raster_info['bounding_box']
        # alternative option is to have mode='intersection' and use AOI vector to clip rasters during align/resize task
        # however, this would mean that edge pixels would be less accurate/lack spatial context

        # Users should input the AOI to which they want outputs clipped
        # InVEST will take care of buffering the processing AOI to ensure
        # correct edge pixel calculation
        aoi_info = pygeoprocessing.get_vector_info(args['aoi_vector_path'])
        if aoi_info["projection_wkt"] != target_projection:
            aoi_bbox = pygeoprocessing.transform_bounding_box(
                aoi_info["bounding_box"], aoi_info["projection_wkt"],
                target_projection)
        else:
            aoi_bbox = aoi_info["bounding_box"]

        # Expand target bounding box to ensure correct edge pixel calculation
        aoi_bbox += numpy.array([-search_radius, -search_radius,
                                 search_radius, search_radius])
        aoi_bbox = list(aoi_bbox)

        # Check if buffered AOI bbox is larger than input base and alt NDVI
        check_raster_bounds_against_aoi(aoi_bbox, ndvi_bbox, "NDVI_base")
        check_raster_bounds_against_aoi(
            aoi_bbox,
            pygeoprocessing.get_raster_info(args['ndvi_alt'])['bounding_box'],
            "NDVI_alt")

        input_align_list = [args['ndvi_base'], args['ndvi_alt']]
        output_align_list = [file_registry['ndvi_base_aligned'],
                             file_registry['ndvi_alt_aligned']]
        resample_method_list = ['cubicspline', 'cubicspline']

        if 'lulc_base' in args and args['lulc_base'] not in ["", None]:
            input_align_list.append(args['lulc_base'])
            output_align_list.append(file_registry['lulc_base_aligned'])
            resample_method_list.append('near')
            check_raster_bounds_against_aoi(
                aoi_bbox, pygeoprocessing.get_raster_info(
                    args['lulc_base'])['bounding_box'], "LULC_base")
            #TODO: check if aoi_bbox still works?

        ndvi_align_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            args=(input_align_list, output_align_list,
                  resample_method_list,
                  pixel_size,
                  aoi_bbox),
            kwargs={
                'raster_align_index': 0,  # align to base_ndvi
                'target_projection_wkt': target_projection},
            target_path_list=output_align_list,
            task_name='align NDVI rasters')

        mask_base_inputs = [file_registry['ndvi_base_aligned'],
                            file_registry['ndvi_base_aligned_masked']]
        mask_alt_inputs = [file_registry['ndvi_alt_aligned'],
                           file_registry['ndvi_alt_aligned_masked']]
        mask_base_outputs = [file_registry['ndvi_base_aligned_masked']]
        mask_alt_outputs = [file_registry['ndvi_alt_aligned_masked']]
        if 'lulc_base' in args and args['lulc_base'] not in ["", None]:
            LOGGER.info("Masking NDVI using LULC")
            mask_base_inputs += [file_registry['lulc_base_aligned'],
                                 args['lulc_attr_csv'],
                                 file_registry['lulc_mask']]
            mask_base_outputs.append(file_registry['lulc_mask'])

            # Use lulc_alt to mask if provided. Otherwise, use lulc_base
            if 'lulc_alt' in args and args['lulc_alt'] not in ["", None]:
                mask_alt_inputs += [file_registry['lulc_alt_aligned'],
                                    args['lulc_attr_csv'],
                                    file_registry['lulc_mask']]
            else:
                mask_alt_inputs += [file_registry['lulc_base_aligned'],
                                    args['lulc_attr_csv'],
                                    file_registry['lulc_mask']]

            mask_alt_outputs.append(file_registry['lulc_mask'])
        else:
            LOGGER.info("Masking NDVI using threshold NDVI<0")

        mask_base_ndvi_task = task_graph.add_task(
            func=mask_ndvi,
            args=mask_base_inputs,
            target_path_list=mask_base_outputs,
            dependent_task_list=[ndvi_align_task],
            task_name="Mask baseline NDVI"
        )

        mask_alt_ndvi_task = task_graph.add_task(
            func=mask_ndvi,
            args=mask_alt_inputs,
            target_path_list=mask_alt_outputs,
            dependent_task_list=[ndvi_align_task],
            task_name="Mask alternate NDVI"
        )

        pixel_radius = int(round(search_radius/pixel_size[0]))
        LOGGER.info(f"Search radius {search_radius} results in "
                    f"buffer of {pixel_radius} pixels")
        if pixel_radius == 0:
            raise ValueError(
                f"Search radius {search_radius} yielded pixel_radius of zero. "
                "Please increase search radius.")
        kernel_task = task_graph.add_task(
            func=pygeoprocessing.kernels.dichotomous_kernel,
            kwargs={
                'target_kernel_path': file_registry['kernel'],
                'max_distance': pixel_radius,
                'normalize': True}, #TODO do i need to normalize both this and in convolve2d op?
            target_path_list=[file_registry['kernel']],
            task_name='create kernel raster')

        mean_buffer_base_ndvi_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (file_registry['ndvi_base_aligned_masked'], 1),
                (file_registry['kernel'], 1),
                file_registry['ndvi_base_buffer_mean']),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': True,
                'target_datatype': pygeoprocessing.get_raster_info(
                    file_registry['ndvi_base_aligned_masked'])["datatype"],
                'target_nodata': pygeoprocessing.get_raster_info(
                    file_registry['ndvi_base_aligned_masked'])["nodata"][0]},
            dependent_task_list=[mask_base_ndvi_task, kernel_task],
            target_path_list=[file_registry['ndvi_base_buffer_mean']],
            task_name="calculate mean baseline NDVI within buffer")

        mean_buffer_alt_ndvi_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (file_registry['ndvi_alt_aligned_masked'], 1),
                (file_registry['kernel'], 1),
                file_registry['ndvi_alt_buffer_mean']),
            kwargs={
                'ignore_nodata_and_edges': True,
                'mask_nodata': True,
                'normalize_kernel': True,
                'target_datatype': pygeoprocessing.get_raster_info(
                    file_registry['ndvi_alt_aligned_masked'])["datatype"],
                'target_nodata': pygeoprocessing.get_raster_info(
                    file_registry['ndvi_alt_aligned_masked'])["nodata"][0]},
            dependent_task_list=[mask_alt_ndvi_task, kernel_task],
            target_path_list=[file_registry['ndvi_alt_buffer_mean']],
            task_name="calculate mean alternate NDVI within buffer")

        # NOTE: this is the first step where the nodata value of the output raster
        # is set based on pygeoprocessing default for the raster's datatype (rather
        # than using the raster's native nodata)
        delta_ndvi_task = task_graph.add_task(
            func=pygeoprocessing.raster_map,
            args=(lambda base_ndvi, alt_ndvi: alt_ndvi - base_ndvi,
                  [file_registry['ndvi_base_buffer_mean'],
                   file_registry['ndvi_alt_buffer_mean']],
                  file_registry['delta_ndvi']),
            target_path_list=[file_registry['delta_ndvi']],
            dependent_task_list=[mean_buffer_base_ndvi_task,
                                 mean_buffer_alt_ndvi_task],
            task_name="calculate delta ndvi"  # change in nature exposure
        )

        pop_raster_info = pygeoprocessing.get_raster_info(
            args['population_raster'])
        if pop_raster_info['projection_wkt'] != target_projection:
            transformed_bounding_box = pygeoprocessing.transform_bounding_box(
                pop_raster_info['bounding_box'],
                pop_raster_info['projection_wkt'],
                target_projection, edge_samples=11)
            pop_raster_info['bounding_box'] = transformed_bounding_box

        delta_ndvi_bbox = pygeoprocessing.get_raster_info(
                file_registry['delta_ndvi'])['bounding_box']
        target_bounding_box = pygeoprocessing.merge_bounding_box_list(
            [delta_ndvi_bbox,
             pop_raster_info['bounding_box']], 'intersection')
        LOGGER.info("Target bounding box (intersection of population raster"
                    f"and delta NDVI): {target_bounding_box}")

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

        target_bounding_box = pygeoprocessing.get_raster_info(
            file_registry['population_aligned'])["bounding_box"]
        if target_bounding_box != delta_ndvi_bbox:
            raise ValueError(
                "The bounding boxes of the preprocessed population raster and "
                "delta_ndvi raster are not equal. This occurs when the "
                "population raster is too small. The bounds of the population "
                "raster should at least extend across the entire AOI.")

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
                  args['effect_size'],
                  file_registry['preventable_cases'],
                  args["aoi_vector_path"],
                  intermediate_output_dir),
            target_path_list=[file_registry['preventable_cases']],
            dependent_task_list=[delta_ndvi_task, baseline_cases_task],
            task_name="calculate preventable cases"
        )

        health_cost = args.get("health_cost_rate")
        health_cost = float(health_cost) if health_cost else None

        if health_cost:
            LOGGER.info("Calculating preventable cost")
            preventable_cost_task = task_graph.add_task(
                func=calc_preventable_cost,
                args=(file_registry['preventable_cases'], health_cost,
                      file_registry['preventable_cost']),
                target_path_list=[file_registry['preventable_cost']],
                dependent_task_list=[preventable_cases_task],
                task_name="calculate preventable cost"
            )

        task_graph.join()
        # TODO ^ is this best way to require prev cost task done?
        # Can't add as dependent task below as not done if not health_cost_rate

        zonal_stats_inputs = [
            args['aoi_vector_path'],
            file_registry['preventable_cases_cost_sum_table'],
            file_registry['preventable_cases_cost_sum_vector'],
            file_registry['preventable_cases']]

        zonal_stats_dependent_tasks = [preventable_cases_task]

        if health_cost:
            LOGGER.info("Calculating sum preventable cases and cost by polygon")
            zonal_stats_inputs.append(file_registry['preventable_cost'])
            zonal_stats_dependent_tasks.append(preventable_cost_task)

        zonal_stats_task = task_graph.add_task(
            func=zonal_stats_preventable_cases_cost,
            args=zonal_stats_inputs,
            target_path_list=[
                file_registry['preventable_cases_cost_sum_table'],
                file_registry['preventable_cases_cost_sum_vector']],
            dependent_task_list=zonal_stats_dependent_tasks,
            task_name='calculate zonal statistics'
        )

    elif args['scenario'] == 'tcc_ndvi':
        tc_target = float(args['tc_target'])
        raise NotImplementedError

    elif args['scenario'] == 'lulc':
        raise NotImplementedError

    task_graph.close()
    task_graph.join()
    LOGGER.info('Finished Urban Mental Health Model')
    return file_registry.registry


def check_raster_bounds_against_aoi(aoi_bbox, raster_bbox, raster_name):
    """Check if raster bounds are >= bounds of AOI + search_radius.

    Check if the bounds of the raster extend at least search_radius
    meters beyond the bounds of the AOI vector.

    Args:
        aoi_bbox (list): aoi bounds in format [xmin, ymin, xmax, ymax],
            calculated by extending the input vector AOI bounds by
            `search_radius` meters (in the case of TCC, NDVI or LULC rasters)
        raster_bbox (list): raster bounds in format [xmin, ymin, xmax, ymax]

    Returns:
        None

    """

    errors_dict = {"xmin": aoi_bbox[0] < raster_bbox[0],
                   "ymin": aoi_bbox[1] < raster_bbox[1],
                   "xmax": aoi_bbox[2] > raster_bbox[2],
                   "ymax": aoi_bbox[3] > raster_bbox[3]}

    # format numbers nicely
    aoi_bbox = [float(i) for i in aoi_bbox]
    raster_bbox = [float(i) for i in raster_bbox]

    if any(errors_dict.values()):
        errors = [k for k, v in errors_dict.items() if v]
        raise UserWarning(
            "The extent of bounding box of the AOI buffered by the search "
            f"radius exceeds that of the {raster_name} raster input. The "
            f"issue is with the following coordinates: {errors}. For "
            f"reference, the buffered AOI bounding box is: {aoi_bbox}, and "
            f"the raster bbox is: {raster_bbox}")


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


def mask_ndvi(input_ndvi, target_masked_ndvi, input_lulc=None,
              lulc_attr_table=None, target_lulc_mask=None):
    """Mask NDVI using either threshold of NDVI<0 or LULC exclude codes

    Args:
        input_ndvi (str): path to NDVI raster
        target_masked_ndvi (str): path to output masked NDVI raster
        input_lulc (str): (optional) path to LULC raster
        lulc_attr_table (str): (required if input_lulc) path to lulc attribute
            table csv
        target_lulc_mask (str): (required if input_lulc) path to output
            binary mask

    Returns:
        None

    """

    ndvi_info = pygeoprocessing.get_raster_info(input_ndvi)
    ndvi_nodata = ndvi_info["nodata"][0]
    ndvi_dtype = ndvi_info["datatype"]

    def _mask_with_lulc(ndvi, ndvi_nodata, lulc_mask):
        """Mask NDVI using places where LULC's exclude code = 1"""

        ndvi[lulc_mask == 1] = ndvi_nodata
        return ndvi

    def _mask_with_ndvi(ndvi, ndvi_nodata):
        """Mask NDVI using threshold 0"""
        ndvi[ndvi < 0] = ndvi_nodata
        return ndvi

    raster_list = [(input_ndvi, 1), (ndvi_nodata, "raw")]

    if input_lulc:
        # create lulc_mask where any lulc code with corresponding 'exclude'
        # value = 1 get assigned value of 1 and all others are 0
        lulc_df = pandas.read_csv(lulc_attr_table)
        codes = list(lulc_df['lucode'])
        excludes = list(lulc_df['exclude'])

        value_map = {c: e for c, e in zip(codes, excludes) if
                     numpy.isfinite(c)}

        pygeoprocessing.reclassify_raster(
            (input_lulc, 1), value_map, target_lulc_mask, gdal.GDT_Byte,
            255, values_required=True)

        mask_op = _mask_with_lulc
        raster_list.append((target_lulc_mask, 1))
    else:
        mask_op = _mask_with_ndvi

    # has to be float64 or error in raster_map bc numpy cannot cast FLOAT32_NODATA to float32:
    # numpy.can_cast(numpy.min_scalar_type(target_nodata), target_dtype) is False
    pygeoprocessing.raster_calculator(
        raster_list, mask_op, target_masked_ndvi,
        datatype_target=ndvi_dtype, nodata_target=ndvi_nodata)


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
    def _multiply_op(prevalence, pop):
        """Multiply baseline prevalence raster by population raster"""
        return prevalence * pop

    pygeoprocessing.new_raster_from_base(
        population_raster, target_base_prevalence_raster,
        gdal.GDT_Float32, [float(numpy.finfo(numpy.float32).max)])

    pygeoprocessing.rasterize(base_prevalence_vector,
                              target_base_prevalence_raster,
                              option_list=[
                                  "ATTRIBUTE=risk_rate", "ALL_TOUCHED=TRUE",
                                  "MERGE_ALG=REPLACE"])

    raster_list = [target_base_prevalence_raster, population_raster]
    pygeoprocessing.raster_map(_multiply_op, raster_list, target_base_cases)


def calc_preventable_cases(delta_ndvi, baseline_cases, effect_size,
                           target_preventable_cases, aoi, work_dir):
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
            baseline to alternate/counterfactual scenario
        baseline_cases (str): path to raster of number of baseline cases
        effect_size (float): health indicator-specific effect size, given
            as a risk ratio, representing the relationship between nature
            exposure and mental health outcomes.
        target_preventable_cases (str): path to output preventable cases raster
        aoi (str): path to area of interest
        work_dir (str): path to create a temp folder for saving files.

    Returns:
        None.

    """
    def _preventable_cases_op(delta_ndvi, baseline_cases, effect_size_val,
                              ndvi_nodata, bc_nodata):
        valid_mask = (~pygeoprocessing.array_equals_nodata(delta_ndvi,
                                                           ndvi_nodata) &
                      ~pygeoprocessing.array_equals_nodata(baseline_cases,
                                                           bc_nodata))
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = bc_nodata
        result[valid_mask] = 1 - numpy.exp(numpy.log(
            effect_size_val) * 10 * delta_ndvi[valid_mask])
        result[valid_mask] *= baseline_cases[valid_mask]  # preventable cases
        return result

    # make temporary directory to save unclipped file
    temp_dir = tempfile.mkdtemp(dir=work_dir, prefix='unclipped')

    effect_size_val = float(effect_size)  # previously called 'rr0'

    ndvi_nodata = pygeoprocessing.get_raster_info(delta_ndvi)["nodata"][0]
    bc_info = pygeoprocessing.get_raster_info(baseline_cases)
    bc_nodata = bc_info["nodata"][0]
    target_dtype = bc_info["datatype"]

    base_raster_path_band_const_list = [(delta_ndvi, 1),
                                        (baseline_cases, 1),
                                        (effect_size_val, "raw"),
                                        (ndvi_nodata, "raw"),
                                        (bc_nodata, "raw")]

    intermediate_raster = os.path.join(temp_dir, 'prev_cases_unclipped.tif')
    pygeoprocessing.raster_calculator( # error here if population raster smaller than NDVI
        base_raster_path_band_const_list, _preventable_cases_op,
        intermediate_raster, target_dtype, nodata_target=bc_nodata)

    pygeoprocessing.mask_raster(
        (intermediate_raster, 1), aoi, target_preventable_cases)

    shutil.rmtree(temp_dir, ignore_errors=True)


def calc_preventable_cost(preventable_cases, health_cost_rate,
                          target_preventable_cost):
    """Calculate preventable cost

    Args:
        preventable_cases (str): path to preventable cases raster
        health_cost_rate (float): health cost
        target_preventable_cost (str): path to output preventable cost raster

    Returns:
        None

    """
    def _preventable_cost_op(preventable_cases, cost, nodata):
        valid_mask = ~pygeoprocessing.array_equals_nodata(
            preventable_cases, nodata)
        result = numpy.empty(valid_mask.shape, dtype=numpy.float32)
        result[:] = nodata
        result[valid_mask] = preventable_cases[valid_mask]*cost
        return result

    raster_info = pygeoprocessing.get_raster_info(preventable_cases)
    nodata = raster_info["nodata"][0]
    target_dtype = raster_info["datatype"]

    pygeoprocessing.raster_calculator(
        [(preventable_cases, 1), (health_cost_rate, "raw"), (nodata, "raw")],
        _preventable_cost_op, target_preventable_cost, target_dtype,
        nodata_target=nodata)


def zonal_stats_preventable_cases_cost(
        base_vector_path, target_stats_csv, target_aggregate_vector_path,
        preventable_cases_raster, preventable_cost_raster=None):
    """Calculate zonal statistics for each polygon in the AOI
    and write results to a csv and vector file.

    Args:
        base_vector_path (string): path to the AOI vector.
        target_stats_csv (string): path to csv file to store dictionary
            returned by zonal stats.
        target_aggregate_vector_path (string): path to vector to store zonal
            stats
        preventable_cases_raster (string): path to preventable cases raster,
            which gets aggregated by AOI polygon(s).
        preventable_cost_raster (string): (optional) path to preventable cost
            raster, which gets aggregated by AOI polygon(s).

    Returns:
        None

    """

    # write zonal stats to new vector
    aoi_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    driver = gdal.GetDriverByName('GPKG')

    if os.path.exists(target_aggregate_vector_path):
        driver.Delete(target_aggregate_vector_path)
    driver.CreateCopy(target_aggregate_vector_path, aoi_vector)
    aoi_vector = None

    cases_sum_field = ogr.FieldDefn('sum_cases', ogr.OFTReal)
    cases_sum_field.SetWidth(24)
    cases_sum_field.SetPrecision(11)

    # calculate zonal stats for cases and cost
    cases_stats_dict = pygeoprocessing.zonal_statistics(
        (preventable_cases_raster, 1), base_vector_path, ignore_nodata=True)

    output_dict = {k: {"sum_cases": v['sum']}
                   for k, v in cases_stats_dict.items()}

    target_aggregate_vector = gdal.OpenEx(
        target_aggregate_vector_path, gdal.OF_UPDATE)
    target_aggregate_layer = target_aggregate_vector.GetLayer()
    target_aggregate_layer.CreateField(cases_sum_field)

    if preventable_cost_raster:
        cost_stats_dict = pygeoprocessing.zonal_statistics(
            (preventable_cost_raster, 1), base_vector_path, ignore_nodata=True)

        cost_stats_dict = {k: {"sum_cost": v['sum']}
                           for k, v in cost_stats_dict.items()}

        # merge the dicts - TODO check this always works
        output_dict = {fid: output_dict[fid] | cost_stats_dict.get(fid, None)
                       for fid in output_dict.keys()}

        cost_sum_field = ogr.FieldDefn('sum_cost', ogr.OFTReal)
        cost_sum_field.SetWidth(24)
        cost_sum_field.SetPrecision(11)
        target_aggregate_layer.CreateField(cost_sum_field)

    target_aggregate_layer.ResetReading()
    target_aggregate_layer.StartTransaction()

    for poly_feat in target_aggregate_layer:
        poly_fid = poly_feat.GetFID()
        poly_feat.SetField('sum_cases',
                           float(output_dict[poly_fid]['sum_cases']))
        if preventable_cost_raster:
            poly_feat.SetField('sum_cost',
                               float(output_dict[poly_fid]['sum_cost']))
        target_aggregate_layer.SetFeature(poly_feat)

    target_aggregate_layer.CommitTransaction()
    target_aggregate_layer, target_aggregate_vector = None, None

    # Calculate total cases and cost for all polygons in AOI
    tot_sum_prev_cases = numpy.sum([v['sum_cases']
                                    for v in output_dict.values()])
    output_dict["ALL"] = {}
    output_dict["ALL"]["total_cases"] = tot_sum_prev_cases

    if preventable_cost_raster:
        tot_sum_cost = numpy.sum([v['sum_cost'] for k, v in output_dict.items()
                                  if k != "ALL"])
        output_dict["ALL"]["total_cost"] = tot_sum_cost

    output_df = pandas.DataFrame(output_dict).T
    output_df.index.name = "FID"
    output_df.to_csv(target_stats_csv)


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
