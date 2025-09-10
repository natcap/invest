"""InVEST Urban Mental Health model."""
import logging
import math
import os
import pickle

import numpy
import pygeoprocessing
import taskgraph
from osgeo import gdal
from osgeo import ogr

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
    different_projections_ok=False,  # TODO
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
            projected=True
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


def execute(args):
    """Urban Mental Health

    The model estimates the impacts of nature exposure, and more specifically
    residential greenness, on mental health. Residential nature exposure is
    defined as the average NDVI within a distance of a residence that benefits
    human mental health. The mental health model calculates the preventable
    mental disorder cases at the pixel level, based on the selected urban
    greening scenario.

    Args:
        args['workspace_dir'] (string): (required) a path to the directory that
            will write output and other temporary files during calculation.
        args['results_suffix'] (string): (optional) appended to any output
            filename.
        args['aoi_vector_path'] (string): (required) path to a polygon vector
            that is projected in a coordinate system with units of meters.
            The polygon should intersect the baseline prevalence vector and
            the population raster.
        args['population_raster] (string): (required) a path to a raster of
            gridded population data representing the number of people per pixel,
            ideally at the pixel level.
        args['search_radius'] (float): (required) the distance used to define
            the surrounding area of a person's residence that best represents
            daily exposure to nearby nature.
        args['effect_size_csv'] (string): (required) a path to a CSV table
            containing health indicator-specific effect sizes, such as risk
            ratios or odds ratios, representing the relationship between nature
            exposure and mental health outcomes.
        args['baseline_prevalence_vector'] (string): (required) a path to a
            vector providing the baseline prevalence (or incidence) rate of
            a specific mental health outcome (e.g., depression or anxiety)
            across administrative units within the study area. This data allows
            the model to estimate preventable cases by comparing current rates
            with those projected under improved nature exposure scenarios. The
            vector must contain field `risk_rate`.
        args['health_cost_rate_csv'] (string): (optional) a path to a CSV table
            providing the societal cost per case (e.g., in USD PPP) for the
            mental health outcome described by the `baseline_prevalence_vector`.
            This data enables the model to estimate the economic value of
            preventable cases under different urban nature scenarios. Costs can
            be specified at national, regional, or local levels depending on
            data availability.
        args['scenario'] (string): (required) which of the three land use scenarios
            to model.
        args['tc_raster'] (string): required if args['scenario'] == 'tc_ndvi',
            a path to a raster providing tree canopy cover under current or
            baseline conditions.
        args['tc_target'] (float): required if args['scenario'] == 'tc_ndvi',
            user-defined target for tree canopy cover within area of interest.
            This value represents a desired scenario and will be used to
            compare against the baseline tree cover to estimate potential
            health benefits.
        args['ndvi_base'] (string): required if args['scenario'] != 'lulc' or
            not args['lulc_attr_csv'], a path to a Normalized Difference Vegetation
            Index raster representing current or baseline conditions, which gives
            the greenness of vegetation in a given cell.
        args['ndvi_alt'] (string): required if args['scenario'] == 'ndvi', a path
            to an NDVI raster under future or counterfactual conditions.
        args['lulc_base'] (string): required if args['scenario'] == 'lulc', a path
            to a Land Use/Land Cover raster showing current or baseline conditions
        args['lulc_alt'] (string): required if args['scenario'] == 'lulc', a path
            to a Land Use/Land Cover raster showing a future or counterfactural
            scenario.
        args['lulc_attr_csv'] (string): required if args['scenario'] == 'lulc' and
            not args['ndvi_base'], a path to a CSV table that maps LULC codes to
            corresponding NDVI values and specifies whether to exclude the LULC
            class from analysis. The following fields are required: 
            'lucode' (int): unique LULC class identifier
            'ndvi' (float): NDVI value of the LULC class
            'exclude' (bool): 0 for keeping, 1 for excluding the LULC class

    """
    return True


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
