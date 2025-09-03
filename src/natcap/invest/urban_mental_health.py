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

MODEL_SPEC = spec.ModelSpec(
    model_id="urban_mental_health",
    model_title=gettext("Urban Mental Health"),
    userguide="urban_mental_health.html",  # TODO - add this to UG
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["aoi_vector_path", "population_raster"],
        ["search_radius"],
        ["effect_size_csv"],
        ["baseline_prevalence_vector", "health_cost_rate_csv"],
        ["scenario", "tc_raster", "tc_target", "ndvi_base", "ndvi_alt",
         "lulc_base", "lulc_alt", "lulc_attr_csv"]
        #  ^ could either be a dropdown menu of scenario options (i.e., "TC + baseline NDVI", "baseline and alternate LULC", "baseline and alternate NDVI")
    ],
    validate_spatial_overlap=True,  # TODO
    different_projections_ok=False,  # TODO
    aliases=("umh"),
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.AOI.model_copy(update=dict( #  TODO: potentially want to have this req to be census tract pop. shp? or only req that if opt 1 but optional for opts 2-3?
            id="aoi_vector_path",
            about=gettext("Map of the area over which to run the model."),
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
            columns=[
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
                    id="health_risk_rate",
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
                spec.Option(key="tcc_ndvi",
                            display_name="TCC & Baseline NDVI"),
                spec.Option(key="lulc",
                            display_name="Baseline & Alternate LULC"),
                spec.Option(key="ndvi",
                            display_name="Baseline & Alternate NDVI")
            ]
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
            required="scenario=tcc_ndvi" #  TODO: check this is correct
        ),
        spec.SingleBandRasterInput(
            id="ndvi_base",
            name=gettext("baseline ndvi"),
            about=gettext(
                "Map of NDVI under current or baseline conditions."
            ),
            data_type=float,
            units=None,
            projected=True,
            required="scenario=ndvi or scenario=tcc_ndvi" #TODO check this
        ),
        spec.SingleBandRasterInput(
            id="ndvi_alt",
            name=gettext("alternate ndvi"),
            about=gettext(
                "Map of NDVI under future or counterfactual conditions."
            ),
            data_type=float,
            units=None,
            projected=True,
            required="scenario=ndvi"
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
            required="scenario=lulc"
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
            required="scenario=lulc"
        ),
        spec.CSVInput(
            id="lulc_attr_csv",
            name=gettext("LULC Attribute Table"),
            about=gettext(
                "A table mapping LULC codes to containing LULC codes, corresponding LULC class "
                "names, excluded LULC (0 for keeping, and 1 for excluding), "
                "and NDVI value."
            ),
            fields=[
                spec.IntegerInput(
                    id="lulc_code",
                    about=gettext("LULC code")
                ),
                spec.BooleanInput(
                    id="exclude",
                    about=gettext(
                        "Whether to exclude the lulc class (e.g., if water) "
                        "or keep it")
                ),
                spec.NumberInput(
                    id="ndvi_value",
                    about=gettext("NDVI value"),
                    data_type=float
                )
            ],
        )
        ]
)
