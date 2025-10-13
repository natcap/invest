import collections
import logging
import math
import os
import re
import shutil
import tempfile

import numpy
import numpy.testing
import pygeoprocessing
import pygeoprocessing.kernels
import pygeoprocessing.symbolic
import shapely.ops
import shapely.wkb
import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from . import gettext
from . import spec
from . import utils
from . import validation
from .spec import u

LOGGER = logging.getLogger(__name__)
UINT32_NODATA = int(numpy.iinfo(numpy.uint32).max)
FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255
KERNEL_LABEL_DICHOTOMY = 'dichotomy'
KERNEL_LABEL_EXPONENTIAL = 'exponential'
KERNEL_LABEL_GAUSSIAN = 'gaussian'
KERNEL_LABEL_DENSITY = 'density'
# KERNEL_LABEL_POWER = 'power'
RADIUS_OPT_UNIFORM = 'uniform radius'
RADIUS_OPT_URBAN_NATURE = 'radius per urban nature class'
RADIUS_OPT_POP_GROUP = 'radius per population group'
POP_FIELD_REGEX = '^pop_'
ID_FIELDNAME = 'adm_unit_id'
MODEL_SPEC = spec.ModelSpec(
    model_id="urban_nature_access",
    model_title=gettext("Urban Nature Access"),
    userguide="urban_nature_access.html",
    validate_spatial_overlap=True,
    different_projections_ok=True,
    aliases=("una",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_raster_path", "lulc_attribute_table"],
        [
            "population_raster_path",
            "admin_boundaries_vector_path",
            "population_group_radii_table",
            "urban_nature_demand",
            "aggregate_by_pop_group"
        ],
        ["search_radius_mode", "decay_function", "search_radius"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="lulc_raster_path",
            name=gettext("land use/land cover"),
            about=gettext(
                "A map of LULC codes. Each land use/land cover type must be assigned a"
                " unique integer code. All values in this raster must have corresponding"
                " entries in the LULC attribute table. For this model in particular, the"
                " urban nature types are of importance.  Non-nature types are not"
                " required to be uniquely identified. All outputs will be produced at the"
                " resolution of this raster."
            ),
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.CSVInput(
            id="lulc_attribute_table",
            name=gettext("LULC attribute table"),
            about=gettext(
                "A table identifying which LULC codes represent urban nature. All LULC"
                " classes in the Land Use Land Cover raster MUST have corresponding"
                " values in this table.  Each row is a land use land cover class."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.RatioInput(
                    id="urban_nature",
                    about=gettext(
                        "The proportion (0-1) indicating the naturalness of the land"
                        " types. 0 indicates the naturalness level of this LULC type is"
                        " lowest (0% nature), while 1 indicates that of this LULC type is"
                        " the highest (100% nature)"
                    ),
                    units=None
                ),
                spec.NumberInput(
                    id="search_radius_m",
                    about=gettext(
                        "The distance within which a LULC type is relevant to the"
                        " population group of interest. This is the search distance that"
                        " the model will apply for this LULC type. Values must be >= 0"
                        " and defined in meters. Required when running the model with"
                        " search radii defined per urban nature class."
                    ),
                    required='search_radius_mode == "radius per urban nature class"',
                    units=u.meter,
                    expression="value >= 0"
                )
            ],
            index_col="lucode"
        ),
        spec.SingleBandRasterInput(
            id="population_raster_path",
            name=gettext("population raster"),
            about=gettext("A raster representing the number of inhabitants per pixel."),
            data_type=float,
            units=u.count,
            projected=True,
            projection_units=u.meter
        ),
        spec.VectorInput(
            id="admin_boundaries_vector_path",
            name=gettext("administrative boundaries"),
            about=gettext(
                "A vector representing administrative units. Polygons representing"
                " administrative units should not overlap. Overlapping administrative"
                " geometries may cause unexpected results and for this reason should not"
                " overlap."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.RatioInput(
                    id="pop_[POP_GROUP]",
                    about=(
                        "The proportion of the population within each administrative unit"
                        " belonging to the identified population group (POP_GROUP). At"
                        " least one column with the prefix 'pop_' is required when"
                        " aggregating output by population groups."
                    ),
                    required=(
                        "(search_radius_mode == 'radius per population group') or"
                        " aggregate_by_pop_group"
                    ),
                    units=None
                )
            ],
            projected=None
        ),
        spec.NumberInput(
            id="urban_nature_demand",
            name=gettext("urban nature demand per capita"),
            about=gettext(
                "The amount of urban nature that each resident should have access to."
                " This is often defined by local urban planning documents."
            ),
            units=u.meter**2,
            expression="value > 0"
        ),
        spec.OptionStringInput(
            id="decay_function",
            name=gettext("decay function"),
            about=gettext(
                "Pixels within the search radius of an urban nature pixel have a"
                " distance-weighted contribution to an urban nature pixel according to"
                " the selected distance-weighting function."
            ),
            options=[
                spec.Option(
                    key="density",
                    about=(
                        "Contributions to an urban nature pixel decrease faster as"
                        " distances approach the search radius. Weights are calculated by"
                        ' "weight = 0.75 * (1-(pixel_dist / search_radius)^2)"')),
                spec.Option(
                    key="dichotomy",
                    about=(
                        "All pixels within the search radius contribute equally to an"
                        " urban nature pixel.")),
                spec.Option(
                    key="exponential",
                    about=(
                        "Contributions to an urban nature pixel decrease exponentially,"
                        ' where "weight = e^(-pixel_dist / search_radius)"')),
                spec.Option(
                    key="gaussian",
                    about=(
                        "Contributions to an urban nature pixel decrease according to a"
                        ' normal ("gaussian") distribution with a sigma of 3.'),
                    display_name="Gaussian")
            ]
        ),
        spec.OptionStringInput(
            id="search_radius_mode",
            name=gettext("search radius mode"),
            about=gettext("The type of search radius to use."),
            options=[
                spec.Option(
                    key="uniform radius",
                    display_name="Uniform radius",
                    about="The search radius is the same for all types of urban nature."),
                spec.Option(
                    key="radius per urban nature class",
                    display_name="Radius defined per urban nature class",
                    about=(
                        "The search radius is defined for each distinct urban nature LULC"
                        " classification.")),
                spec.Option(
                    key="radius per population group",
                    display_name="Radius defined per population group",
                    about="The search radius is defined for each distinct population group."),
            ]
        ),
        spec.BooleanInput(
            id="aggregate_by_pop_group",
            name=gettext("Aggregate by population groups"),
            about=gettext(
                "Whether to aggregate statistics by population group within each"
                " administrative unit. If selected, population groups will be read from"
                " the fields of the user-defined administrative boundaries vector. This"
                " option is implied if the search radii are defined by population groups."
            ),
            required=False
        ),
        spec.NumberInput(
            id="search_radius",
            name=gettext("uniform search radius"),
            about=gettext(
                "The search radius to use when running the model under a uniform search"
                " radius. Required when running the model with a uniform search radius."
                " Units are in meters."
            ),
            required='search_radius_mode == "uniform radius"',
            allowed='search_radius_mode == "uniform radius"',
            units=u.meter,
            expression="value > 0"
        ),
        spec.CSVInput(
            id="population_group_radii_table",
            name=gettext("population group radii table"),
            about=gettext(
                "A table associating population groups with the distance in meters that"
                " members of the population group will, on average, travel to find urban"
                " nature.  Required when running the model with search radii defined per"
                " population group."
            ),
            required='search_radius_mode == "radius per population group"',
            allowed='search_radius_mode == "radius per population group"',
            columns=[
                spec.StringInput(
                    id="pop_group",
                    about=gettext(
                        "The name of the population group. Names must match the names"
                        " defined in the administrative boundaries vector."
                    ),
                    required=False,
                    regexp=None
                ),
                spec.NumberInput(
                    id="search_radius_m",
                    about=gettext(
                        "The search radius in meters to use for this population group. "
                        " Values must be >= 0."
                    ),
                    units=u.meter,
                    expression="value >= 0"
                )
            ],
            index_col="pop_group"
        )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id="urban_nature_supply_percapita",
            path="output/urban_nature_supply_percapita.tif",
            about=gettext("The calculated supply per capita of urban nature."),
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_demand",
            path="output/urban_nature_demand.tif",
            about=gettext(
                "The required area of urban nature needed by the population"
                " residing in each pixel in order to fully satisfy their urban"
                " nature needs. Higher values indicate a greater demand for"
                " accessible urban nature from the surrounding area."
            ),
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_balance_totalpop",
            path="output/urban_nature_balance_totalpop.tif",
            about=gettext(
                "The urban nature balance for the total population in a pixel."
                " Positive values indicate an oversupply of urban nature relative"
                " to the stated urban nature demand. Negative values indicate an"
                " undersupply of urban nature relative to the stated urban nature"
                " demand. This output is of particular relevance to understand"
                " the total amount of nature deficit for the population in a"
                " particular pixel."
            ),
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_balance_percapita",
            path="output/urban_nature_balance_percapita.tif",
            about=gettext("The urban nature balance per capita in a pixel."),
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.VectorOutput(
            id="admin_boundaries",
            path="output/admin_boundaries.gpkg",
            about=(
                "A copy of the user's administrative boundaries vector with a"
                " single layer."
            ),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[
                spec.NumberOutput(
                    id="SUP_DEMadm_cap",
                    about=gettext(
                        "The average urban nature supply/demand balance available"
                        " per person within this administrative unit. If no"
                        " people reside within this administrative unit, this"
                        " field will have no value (NaN, NULL or None, depending"
                        " on your GIS software)."
                    ),
                    units=u.meter**2 / u.person
                ),
                spec.NumberOutput(
                    id="Pund_adm",
                    about=gettext(
                        "The total population within the administrative unit that"
                        " is undersupplied with urban nature. If aggregating by"
                        " population groups, this will be the sum of"
                        " undersupplied populations across all population groups"
                        " within this administrative unit."
                    ),
                    units=u.people
                ),
                spec.NumberOutput(
                    id="Povr_adm",
                    about=gettext(
                        "The total population within the administrative unit that"
                        " is oversupplied with urban nature. If aggregating by"
                        " population groups, this will be the sum of oversupplied"
                        " populations across all population groups within this"
                        " administrative unit."
                    ),
                    units=u.people
                ),
                spec.NumberOutput(
                    id="SUP_DEMadm_cap_[POP_GROUP]",
                    about=gettext(
                        "The mean urban nature supply/demand balance available"
                        " per person in population group POP_GROUP within this"
                        " administrative unit."
                    ),
                    created_if=(
                        "(search_radius_mode == 'radius per population group') or"
                        " aggregate_by_pop_group"
                    ),
                    units=u.meter**2 / u.person
                ),
                spec.NumberOutput(
                    id="Pund_adm_[POP_GROUP]",
                    about=gettext(
                        "The total population belonging to the population group"
                        " POP_GROUP within this administrative unit that are"
                        " undersupplied with urban nature."
                    ),
                    created_if=(
                        "(search_radius_mode == 'radius per population group') or"
                        " aggregate_by_pop_group"
                    ),
                    units=u.people
                ),
                spec.NumberOutput(
                    id="Povr_adm_[POP_GROUP]",
                    about=gettext(
                        "The total population belonging to the population group"
                        " POP_GROUP within this administrative unit that is"
                        " oversupplied with urban nature."
                    ),
                    created_if=(
                        "(search_radius_mode == 'radius per population group') or"
                        " aggregate_by_pop_group"
                    ),
                    units=u.people
                )
            ]
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_balance_[POP_GROUP]",
            path="output/urban_nature_balance_[POP_GROUP].tif",
            about=gettext(
                "Positive pixel values indicate an oversupply of urban nature for"
                " the population group POP_GROUP relative to the stated urban"
                " nature demand. Negative values indicate an undersupply of urban"
                " nature for the population group POP_GROUP relative to the"
                " stated urban nature demand."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_balance_percapita_[POP_GROUP]",
            path="output/urban_nature_balance_percapita_[POP_GROUP].tif",
            about=gettext(
                "Per-capita urban nature balance for each population group."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.SingleBandRasterOutput(
            id="accessible_urban_nature",
            path="output/accessible_urban_nature.tif",
            about=gettext(
                "The area of greenspace available within the defined radius,"
                " weighted by the selected decay function."
            ),
            created_if="search_radius_mode == 'radius per urban nature class'",
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="accessible_urban_nature_lucode_[LUCODE]",
            path="output/accessible_urban_nature_lucode_[LUCODE].tif",
            about=gettext(
                "The area of greenspace available within the radius associated"
                " with urban nature class LUCODE, weighted by the selected decay"
                " function."
            ),
            created_if="search_radius_mode == 'radius per urban nature class'",
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="accessible_urban_nature_to_[POP_GROUP]",
            path="output/accessible_urban_nature_to_[POP_GROUP].tif",
            about=gettext(
                "The area of greenspace available within the radius associated"
                " with group POP_GROUP, weighted by the selected decay function."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="aligned_lulc",
            path="intermediate/aligned_lulc.tif",
            about=(
                "A copy of the user's land use land cover raster. If the"
                " user-supplied LULC has non-square pixels, they will be"
                " resampled to square pixels in this raster."
            ),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="aligned_population",
            path="intermediate/aligned_population.tif",
            about=(
                "The user's population raster, aligned to the same resolution and"
                " dimensions as the aligned LULC."
            ),
            data_type=float,
            units=u.count
        ),
        spec.SingleBandRasterOutput(
            id="undersupplied_population",
            path="intermediate/undersupplied_population.tif",
            about=gettext("The population experiencing an urban nature deficit."),
            data_type=float,
            units=u.count
        ),
        spec.SingleBandRasterOutput(
            id="oversupplied_population",
            path="intermediate/oversupplied_population.tif",
            about=gettext("The population experiencing an urban nature surplus."),
            data_type=float,
            units=u.count
        ),
        spec.SingleBandRasterOutput(
            id="distance_weighted_population_within_[SEARCH_RADIUS]",
            path="intermediate/distance_weighted_population_within_[SEARCH_RADIUS].tif",
            about=(
                "A sum of the population within the given search radius"
                " SEARCH_RADIUS, weighted by the user's decay function."
            ),
            created_if=(
                "search_radius_mode == 'uniform radius' or search_radius_mode =="
                " 'radius per urban nature class'"
            ),
            data_type=float,
            units=u.count
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_area",
            path="intermediate/urban_nature_area.tif",
            about=gettext(
                "The area of urban nature (in square meters) represented in each"
                " pixel."
            ),
            created_if=(
                "search_radius_mode == 'uniform radius' or search_radius_mode =="
                " 'radius per population group'"
            ),
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_population_ratio",
            path="intermediate/urban_nature_population_ratio.tif",
            about=gettext("The calculated urban nature/population ratio."),
            created_if="search_radius_mode == 'uniform radius'",
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_area_[LUCODE]",
            path="intermediate/urban_nature_area_[LUCODE].tif",
            about=gettext(
                "Pixel values represent the ares of urban nature (in square"
                " meters) represented in each pixel for the urban nature class"
                " represented by the land use land cover code LUCODE."
            ),
            created_if="search_radius_mode == 'radius per urban nature class'",
            data_type=float,
            units=u.meter**2
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_supply_percapita_lucode_[LUCODE]",
            path="intermediate/urban_nature_supply_percapita_lucode_[LUCODE].tif",
            about=gettext(
                "The urban nature supplied to populations due to the land use"
                " land cover code LUCODE"
            ),
            created_if="search_radius_mode == 'radius per urban nature class'",
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_population_ratio_lucode_[LUCODE]",
            path="intermediate/urban_nature_population_ratio_lucode_[LUCODE].tif",
            about=gettext(
                "The calculated urban nature/population ratio for the urban"
                " nature class represented by the land use land cover code"
                " LUCODE."
            ),
            created_if="search_radius_mode == 'radius per urban nature class'",
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.SingleBandRasterOutput(
            id="population_in_[POP_GROUP]",
            path="intermediate/population_in_[POP_GROUP].tif",
            about=gettext(
                "Each pixel represents the population of a pixel belonging to the"
                " population in the population group POP_GROUP."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.count
        ),
        spec.SingleBandRasterOutput(
            id="proportion_of_population_in_[POP_GROUP]",
            path="intermediate/proportion_of_population_in_[POP_GROUP].tif",
            about=gettext(
                "Each pixel represents the proportion of the total population"
                " that belongs to the population group POP_GROUP."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.none
        ),
        spec.SingleBandRasterOutput(
            id="distance_weighted_population_in_[POP_GROUP]",
            path="intermediate/distance_weighted_population_in_[POP_GROUP].tif",
            about=(
                "Each pixel represents the total number of people within the"
                " search radius for the population group POP_GROUP, weighted by"
                " the user's selection of decay function."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.people
        ),
        spec.SingleBandRasterOutput(
            id="distance_weighted_population_all_groups",
            path="intermediate/distance_weighted_population_all_groups.tif",
            about=gettext(
                "The total population, weighted by the appropriate decay"
                " function."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.people
        ),
        spec.SingleBandRasterOutput(
            id="urban_nature_supply_percapita_to_[POP_GROUP]",
            path="intermediate/urban_nature_supply_percapita_to_[POP_GROUP].tif",
            about=gettext(
                "The urban nature supply per capita to population group"
                " POP_GROUP."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.meter**2 / u.person
        ),
        spec.SingleBandRasterOutput(
            id="undersupplied_population_[POP_GROUP]",
            path="intermediate/undersupplied_population_[POP_GROUP].tif",
            about=gettext(
                "The population in population group POP_GROUP that are"
                " experiencing an urban nature deficit."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.people
        ),
        spec.SingleBandRasterOutput(
            id="oversupplied_population_[POP_GROUP]",
            path="intermediate/oversupplied_population_[POP_GROUP].tif",
            about=gettext(
                "The population in population group POP_GROUP that are"
                " experiencing an urban nature surplus."
            ),
            created_if="search_radius_mode == 'radius per population group'",
            data_type=float,
            units=u.people
        ),
        spec.SingleBandRasterOutput(
            id="masked_population",
            path="intermediate/masked_population.tif",
            about=gettext("Masked population raster."),
            data_type=float,
            units=u.people
        ),
        spec.SingleBandRasterOutput(
            id="masked_lulc",
            path="intermediate/masked_lulc.tif",
            about=gettext("Masked LULC raster."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="aligned_mask",
            path="intermediate/aligned_valid_pixels_mask.tif.tif",
            about=gettext("Aligned mask raster indicating valid pixels."),
            data_type=int,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="kernel_[SEARCH_RADIUS]",
            path="intermediate/kernel_[SEARCH_RADIUS].tif",
            about=gettext(
                "The distance decay kernel raster for the given search radius."),
            data_type=float,
            units=None
        ),
        spec.SingleBandRasterOutput(
            id="admin_boundaries_ids",
            path="intermediate/admin_boundaries_ids.tif",
            about=gettext("Raster of admin boundary IDs."),
            data_type=int,
            units=None
        ),
        spec.VectorOutput(
            id="reprojected_admin_boundaries",
            path="intermediate/reprojected_admin_boundaries.gpkg",
            about=gettext("Reprojected administrative area boundaries vector"),
            geometry_types={"POLYGON", "MULTIPOLYGON"},
            fields=[],
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Urban Nature Access.

    Args:
        args['workspace_dir'] (string): (required) Output directory for
            intermediate, temporary and final files.
        args['results_suffix'] (string): (optional) String to append to any
            output file.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for executing the tasks of this model.  If omitted, computation
            will take place in the current process.
        args['lulc_raster_path'] (string): (required) A string path to a
            GDAL-compatible land-use/land-cover raster containing integer
            landcover codes.  Must be linearly projected in meters.
        args['lulc_attribute_table'] (string): (required) A string path to a
            CSV with the following columns:

            * ``lucode``: (required) the integer landcover code represented.
            * ``urban_nature``: (required) a proportion (0-1) representing
              how much of this landcover type is urban nature.  ``0``
              indicates none of this type's area is urban nature, ``1``
              indicates all of this type's area is urban nature.
            * ``search_radius_m``: (conditionally required) the search radius
              for this urban nature LULC class in meters. Required for all
              urban nature LULC codes if ``args['search_radius_mode'] ==
              RADIUS_OPT_URBAN_NATURE``

        args['population_raster_path'] (string): (required) A string path to a
            GDAL-compatible raster where pixels represent the population of
            that pixel.  Must be linearly projected in meters.
        args['admin_boundaries_vector_path'] (string): (required) A string path to a
            GDAL-compatible vector containing polygon areas of interest,
            typically administrative boundaries.  If this vector has any fields
            with fieldnames beginning with ``"pop_"``, these will be treated
            as representing the proportion of the population within an admin
            unit belonging to the given population group.  The name of the
            population group (everything other than a leading ``"pop_"``) must
            uniquely identify the group.
        args['urban_nature_demand'] (number): (required) A positive, nonzero
            number indicating the required urban_nature, in m² per capita.
        args['decay_function'] (string): (required) The selected kernel type.
            Must be one of the keys in ``KERNEL_TYPES``.
        args['search_radius_mode'] (string): (required).  The selected search
            radius mode.  Must be one of ``RADIUS_OPT_UNIFORM``,
            ``RADIUS_OPT_URBAN_NATURE``, or ``RADIUS_OPT_POP_GROUP``.
        args['search_radius'] (number): Required if
            ``args['search_radius_mode'] == RADIUS_OPT_UNIFORM``.  The search
            radius in meters to use in the analysis.
        args['population_group_radii_table'] (string): (optional) A table
            associating population groups with a search radius for that
            population group.  Population group fieldnames must match
            population group fieldnames in the aoi vector.
        args['aggregate_by_pop_group'] (bool): Whether to aggregate statistics
            by population groups in the target vector.  This is implied when
            running the model with ``args['search_radius_mode'] ==
            RADIUS_OPT_POP_GROUP``

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    #    args['decay_function_power_beta'] (number): The beta parameter used
    #        during creation of a power kernel. Required when the selected
    #        kernel is KERNEL_LABEL_POWER.

    LOGGER.info('Starting Urban Nature Access Model')
    args, file_registry, graph = MODEL_SPEC.setup(args)

    kernel_creation_functions = {
        KERNEL_LABEL_GAUSSIAN: _kernel_gaussian,
        KERNEL_LABEL_DENSITY: _kernel_density,
        # Use the user-provided beta args parameter if the user has provided
        # it.  Helpful to have a consistent kernel creation API.
        # KERNEL_LABEL_POWER: functools.partial(
        #     _kernel_power, beta=args.get('decay_function_power_beta', None)),
    }
    # Taskgraph needs a __name__ attribute, so adding one here.
    # kernel_creation_functions[KERNEL_LABEL_POWER].__name__ = (
    #     'functools_partial_decay_power')

    LOGGER.info(f'Using decay function {args["decay_function"]}')

    # Align the population and LULC rasters to the intersection of their
    # bounding boxes.
    lulc_raster_info = pygeoprocessing.get_raster_info(
        args['lulc_raster_path'])
    pop_raster_info = pygeoprocessing.get_raster_info(
        args['population_raster_path'])
    target_bounding_box = pygeoprocessing.merge_bounding_box_list(
        [lulc_raster_info['bounding_box'], pop_raster_info['bounding_box']],
        'intersection')

    squared_lulc_pixel_size = _square_off_pixels(args['lulc_raster_path'])

    lulc_alignment_task = graph.add_task(
        _warp_lulc,
        kwargs={
            "source_lulc_path": args['lulc_raster_path'],
            "target_lulc_path": file_registry['aligned_lulc'],
            "target_pixel_size": squared_lulc_pixel_size,
            "target_bounding_box": target_bounding_box,
        },
        target_path_list=[file_registry['aligned_lulc']],
        task_name='Resample LULC to have square pixels'
    )

    population_alignment_task = graph.add_task(
        _resample_population_raster,
        kwargs={
            'source_population_raster_path': args['population_raster_path'],
            'target_population_raster_path': file_registry['aligned_population'],
            'lulc_pixel_size': squared_lulc_pixel_size,
            'lulc_bb': target_bounding_box,
            'lulc_projection_wkt': lulc_raster_info['projection_wkt'],
            'working_dir': args['workspace_dir'],
        },
        target_path_list=[file_registry['aligned_population']],
        task_name='Resample population to LULC resolution')

    valid_pixels_mask_task = graph.add_task(
        _create_valid_pixels_nodata_mask,
        kwargs={
            'raster_list': [
                file_registry['aligned_lulc'],
                file_registry['aligned_population']
            ],
            'target_mask_path': file_registry['aligned_mask'],
        },
        task_name='Create a valid pixels mask from lulc and population',
        target_path_list=[file_registry['aligned_mask']],
        dependent_task_list=[
            lulc_alignment_task, population_alignment_task]
    )

    population_mask_task = graph.add_task(
        _mask_raster,
        kwargs={
            'source_raster_path': file_registry['aligned_population'],
            'mask_raster_path': file_registry['aligned_mask'],
            'target_raster_path': file_registry['masked_population'],
        },
        task_name='Mask population to the known valid pixels',
        target_path_list=[file_registry['masked_population']],
        dependent_task_list=[
            population_alignment_task, valid_pixels_mask_task]
    )

    lulc_mask_task = graph.add_task(
        _mask_raster,
        kwargs={
            'source_raster_path': file_registry['aligned_lulc'],
            'mask_raster_path': file_registry['aligned_mask'],
            'target_raster_path': file_registry['masked_lulc'],
        },
        task_name='Mask lulc to the known valid pixels',
        target_path_list=[file_registry['masked_lulc']],
        dependent_task_list=[
            lulc_alignment_task, valid_pixels_mask_task]
    )

    aoi_reprojection_task = graph.add_task(
        _reproject_and_identify,
        kwargs={
            'base_vector_path': args['admin_boundaries_vector_path'],
            'target_projection_wkt': lulc_raster_info['projection_wkt'],
            'target_path': file_registry['reprojected_admin_boundaries'],
            'driver_name': 'GPKG',
            # Making the layer name be what we want the final output to be
            # called.  Preemptively removing dashes - Arc doesn't like it.
            'target_layer_name': os.path.splitext(
                os.path.basename(
                    file_registry['admin_boundaries']))[0].replace('-', '_'),
            'id_fieldname': ID_FIELDNAME,
        },
        task_name='Reproject admin units',
        target_path_list=[file_registry['reprojected_admin_boundaries']],
        dependent_task_list=[]
    )

    # This _could_ be a raster_calculator operation, but the math is so simple
    # that it seems like this could suffice.
    _ = graph.add_task(
        pygeoprocessing.symbolic.evaluate_raster_calculator_expression,
        kwargs={
            'expression': f"population * {args['urban_nature_demand']}",
            'symbol_to_path_band_map': {
                'population': (file_registry['masked_population'], 1),
            },
            'target_nodata': FLOAT32_NODATA,
            'target_raster_path': file_registry['urban_nature_demand'],
        },
        task_name='Calculate urban nature demand',
        target_path_list=[file_registry['urban_nature_demand']],
        dependent_task_list=[population_mask_task]
    )

    # If we're doing anything with population groups, rasterize the AOIs and
    # create the proportional population rasters.
    proportional_population_tasks = {}
    pop_group_proportion_tasks = {}
    if (args['search_radius_mode'] == RADIUS_OPT_POP_GROUP
            or args['aggregate_by_pop_group']):
        split_population_fields = list(
            filter(lambda x: re.match(POP_FIELD_REGEX, x),
                   validation.load_fields_from_vector(
                       args['admin_boundaries_vector_path'])))

        if _geometries_overlap(args['admin_boundaries_vector_path']):
            LOGGER.warning(
                "Some administrative boundaries overlap, which will affect "
                "the accuracy of supply rasters per population group. ")

        aois_rasterization_task = graph.add_task(
            _rasterize_aois,
            kwargs={
                'base_raster_path': file_registry['masked_lulc'],
                'aois_vector_path':
                    file_registry['reprojected_admin_boundaries'],
                'target_raster_path': file_registry['admin_boundaries_ids'],
                'id_fieldname': ID_FIELDNAME,
            },
            task_name='Rasterize the admin units vector',
            target_path_list=[file_registry['admin_boundaries_ids']],
            dependent_task_list=[
                aoi_reprojection_task, lulc_mask_task]
        )

        for pop_group in split_population_fields:
            aoi_reprojection_task.join()
            field_value_map = _read_field_from_vector(
                file_registry['reprojected_admin_boundaries'], ID_FIELDNAME,
                pop_group)
            proportional_population_tasks[pop_group] = graph.add_task(
                _reclassify_and_multiply,
                kwargs={
                    'aois_raster_path': file_registry['admin_boundaries_ids'],
                    'reclassification_map': field_value_map,
                    'supply_raster_path': file_registry['masked_population'],
                    'target_raster_path': file_registry['population_in_[POP_GROUP]', pop_group],
                },
                task_name=f"Population proportion in pop group {pop_group}",
                target_path_list=[file_registry['population_in_[POP_GROUP]', pop_group]],
                dependent_task_list=[
                    aois_rasterization_task, population_mask_task]
            )

            pop_group_proportion_tasks[pop_group] = graph.add_task(
                _rasterize_aois,
                kwargs={
                    'base_raster_path': file_registry['masked_lulc'],
                    'aois_vector_path':
                        file_registry['reprojected_admin_boundaries'],
                    'target_raster_path':
                        file_registry['proportion_of_population_in_[POP_GROUP]', pop_group],
                    'id_fieldname': pop_group,
                },
                task_name=f'Rasterize proportion of admin units as {pop_group}',
                target_path_list=[file_registry['proportion_of_population_in_[POP_GROUP]', pop_group]],
                dependent_task_list=[
                    aoi_reprojection_task, lulc_mask_task]
            )

    attr_table = MODEL_SPEC.get_input(
        'lulc_attribute_table').get_validated_dataframe(args['lulc_attribute_table'])
    kernel_tasks = {}  # search_radius, kernel task

    if args['search_radius_mode'] == RADIUS_OPT_UNIFORM:
        search_radii = set([args['search_radius']])
    elif args['search_radius_mode'] == RADIUS_OPT_URBAN_NATURE:
        urban_nature_attrs = attr_table[attr_table['urban_nature'] > 0]
        try:
            search_radii = set(urban_nature_attrs['search_radius_m'].unique())
        except KeyError as missing_key:
            raise ValueError(
                f"The column {str(missing_key)} is missing from the LULC "
                f"attribute table {args['lulc_attribute_table']}")
        # Build an iterable of plain tuples: (lucode, search_radius_m)
        lucode_to_search_radii = list(
            urban_nature_attrs[['search_radius_m']].itertuples(name=None))
    elif args['search_radius_mode'] == RADIUS_OPT_POP_GROUP:
        pop_group_table = MODEL_SPEC.get_input(
            'population_group_radii_table').get_validated_dataframe(
            args['population_group_radii_table'])
        search_radii = set(pop_group_table['search_radius_m'].unique())
        # Build a dict of {pop_group: search_radius_m}
        search_radii_by_pop_group = pop_group_table['search_radius_m'].to_dict()
    else:
        valid_options = ', '.join(
            MODEL_SPEC.get_input('search_radius_mode').options.keys())
        raise ValueError(
            "Invalid search radius mode provided: "
            f"{args['search_radius_mode']}; must be one of {valid_options}")

    for search_radius_m in search_radii:
        search_radius_in_pixels = abs(
            search_radius_m / squared_lulc_pixel_size[0])

        if args['decay_function'] == KERNEL_LABEL_DICHOTOMY:
            kernel_func = pygeoprocessing.kernels.dichotomous_kernel
            kernel_kwargs = dict(
                target_kernel_path=file_registry[
                    'kernel_[SEARCH_RADIUS]', search_radius_m],
                max_distance=search_radius_in_pixels,
                normalize=False)
        elif args['decay_function'] == KERNEL_LABEL_EXPONENTIAL:
            kernel_func = pygeoprocessing.kernels.exponential_decay_kernel
            kernel_kwargs = dict(
                target_kernel_path=file_registry[
                    'kernel_[SEARCH_RADIUS]', search_radius_m],
                max_distance=math.ceil(search_radius_in_pixels) * 2 + 1,
                expected_distance=search_radius_in_pixels,
                normalize=False)
        elif args['decay_function'] in [KERNEL_LABEL_GAUSSIAN, KERNEL_LABEL_DENSITY]:
            kernel_func = pygeoprocessing.kernels.create_distance_decay_kernel

            def decay_func(dist_array):
                return kernel_creation_functions[args['decay_function']](
                    dist_array, max_distance=search_radius_in_pixels)

            kernel_kwargs = dict(
                target_kernel_path=file_registry[
                    'kernel_[SEARCH_RADIUS]', search_radius_m],
                distance_decay_function=decay_func,
                max_distance=search_radius_in_pixels,
                normalize=False)
        else:
            raise ValueError('Invalid kernel creation option selected')

        kernel_tasks[search_radius_m] = graph.add_task(
            kernel_func,
            kwargs=kernel_kwargs,
            task_name=(
                f'Create {args["decay_function"]} kernel - {search_radius_m}m'),
            target_path_list=[file_registry['kernel_[SEARCH_RADIUS]', search_radius_m]])

    # Search radius mode 1: the same search radius applies to everything
    if args['search_radius_mode'] == RADIUS_OPT_UNIFORM:
        search_radius_m = list(search_radii)[0]
        LOGGER.info("Running model with search radius mode "
                    f"{RADIUS_OPT_UNIFORM}, radius {search_radius_m}")

        decayed_population_task = graph.add_task(
            _convolve_and_set_lower_bound,
            kwargs={
                'signal_path_band': (file_registry['masked_population'], 1),
                'kernel_path_band': (file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                'target_path': file_registry[
                    'distance_weighted_population_within_[SEARCH_RADIUS]', search_radius_m],
                'working_dir': args['workspace_dir'],
            },
            task_name=f'Convolve population - {search_radius_m}m',
            target_path_list=[
                file_registry['distance_weighted_population_within_[SEARCH_RADIUS]',
                search_radius_m]],
            dependent_task_list=[
                kernel_tasks[search_radius_m], population_mask_task])

        urban_nature_reclassification_task = graph.add_task(
            _reclassify_urban_nature_area,
            kwargs={
                'lulc_raster_path': file_registry['masked_lulc'],
                'lulc_attribute_table': args['lulc_attribute_table'],
                'target_raster_path': file_registry['urban_nature_area'],
            },
            target_path_list=[file_registry['urban_nature_area']],
            task_name='Identify urban nature areas',
            dependent_task_list=[lulc_mask_task]
        )

        _ = graph.add_task(
            _convolve_and_set_lower_bound,
            kwargs={
                "signal_path_band": (file_registry['urban_nature_area'], 1),
                "kernel_path_band": (file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                "target_path": file_registry['accessible_urban_nature'],
                "working_dir": args['workspace_dir'],
            },
            task_name='Accessible urban nature',
            target_path_list=[file_registry['accessible_urban_nature']],
            dependent_task_list=[urban_nature_reclassification_task]
        )

        urban_nature_population_ratio_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_urban_nature_population_ratio,
                rasters=[
                    file_registry['urban_nature_area'],
                    file_registry['distance_weighted_population_within_[SEARCH_RADIUS]',
                                  search_radius_m]],
                target_path=file_registry['urban_nature_population_ratio']),
            task_name=(
                '2SFCA: Calculate R_j urban nature/population ratio - '
                f'{search_radius_m}'),
            target_path_list=[file_registry['urban_nature_population_ratio']],
            dependent_task_list=[
                urban_nature_reclassification_task, decayed_population_task
            ])

        urban_nature_supply_percapita_task = graph.add_task(
            _convolve_and_set_lower_bound,
            kwargs={
                'signal_path_band': (
                    file_registry['urban_nature_population_ratio'], 1),
                'kernel_path_band': (file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                'target_path': file_registry['urban_nature_supply_percapita'],
                'working_dir': args['workspace_dir'],
            },
            task_name='2SFCA - urban nature supply',
            target_path_list=[file_registry['urban_nature_supply_percapita']],
            dependent_task_list=[
                kernel_tasks[search_radius_m],
                urban_nature_population_ratio_task])

    # Search radius mode 2: Search radii are defined per greenspace lulc class.
    elif args['search_radius_mode'] == RADIUS_OPT_URBAN_NATURE:
        LOGGER.info("Running model with search radius mode "
                    f"{RADIUS_OPT_URBAN_NATURE}")
        decayed_population_tasks = {}
        for search_radius_m in search_radii:
            decayed_population_tasks[search_radius_m] = graph.add_task(
                _convolve_and_set_lower_bound,
                kwargs={
                    'signal_path_band': (
                        file_registry['masked_population'], 1),
                    'kernel_path_band': (file_registry[
                        'kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                    'target_path': file_registry[
                        'distance_weighted_population_within_[SEARCH_RADIUS]', search_radius_m],
                    'working_dir': args['workspace_dir'],
                },
                task_name=f'Convolve population - {search_radius_m}m',
                target_path_list=[file_registry[
                    'distance_weighted_population_within_[SEARCH_RADIUS]', search_radius_m]],
                dependent_task_list=[
                    kernel_tasks[search_radius_m], population_mask_task])

        partial_urban_nature_supply_percapita_tasks = []
        for lucode, search_radius_m in lucode_to_search_radii:
            urban_nature_reclassification_task = graph.add_task(
                _reclassify_urban_nature_area,
                kwargs={
                    'lulc_raster_path': file_registry['masked_lulc'],
                    'lulc_attribute_table': args['lulc_attribute_table'],
                    'target_raster_path': file_registry['urban_nature_area_[LUCODE]', lucode],
                    'only_these_urban_nature_codes': set([lucode]),
                },
                target_path_list=[file_registry['urban_nature_area_[LUCODE]', lucode]],
                task_name=f'Identify urban nature areas with lucode {lucode}',
                dependent_task_list=[lulc_mask_task]
            )

            _ = graph.add_task(
                _convolve_and_set_lower_bound,
                kwargs={
                    "signal_path_band": (file_registry['urban_nature_area_[LUCODE]', lucode], 1),
                    "kernel_path_band": (file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                    "target_path": file_registry['accessible_urban_nature_lucode_[LUCODE]', lucode],
                    "working_dir": args['workspace_dir'],
                },
                task_name='Accessible urban nature',
                target_path_list=[file_registry['accessible_urban_nature_lucode_[LUCODE]', lucode]],
                dependent_task_list=[urban_nature_reclassification_task]
            )

            urban_nature_population_ratio_task = graph.add_task(
                func=pygeoprocessing.raster_map,
                kwargs=dict(
                    op=_urban_nature_population_ratio,
                    rasters=[
                        file_registry['urban_nature_area_[LUCODE]', lucode],
                        file_registry['distance_weighted_population_within_[SEARCH_RADIUS]',
                            search_radius_m]],
                    target_path=file_registry[
                        'urban_nature_population_ratio_lucode_[LUCODE]', lucode]),
                task_name=(
                    '2SFCA: Calculate R_j urban nature/population ratio - '
                    f'{search_radius_m}'),
                target_path_list=[file_registry[
                    'urban_nature_population_ratio_lucode_[LUCODE]', lucode]],
                dependent_task_list=[
                    urban_nature_reclassification_task,
                    decayed_population_tasks[search_radius_m]
                ])

            partial_urban_nature_supply_percapita_tasks.append(graph.add_task(
                pygeoprocessing.convolve_2d,
                kwargs={
                    'signal_path_band': (file_registry[
                        'urban_nature_population_ratio_lucode_[LUCODE]', lucode], 1),
                    'kernel_path_band': (file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                    'target_path': file_registry['urban_nature_supply_percapita_lucode_[LUCODE]', lucode],
                    'working_dir': args['workspace_dir'],
                },
                task_name=f'2SFCA - urban_nature supply for lucode {lucode}',
                target_path_list=[file_registry['urban_nature_supply_percapita_lucode_[LUCODE]', lucode]],
                dependent_task_list=[
                    kernel_tasks[search_radius_m],
                    urban_nature_population_ratio_task]))

        urban_nature_supply_percapita_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_sum_op,
                rasters=[file_registry[
                    'urban_nature_supply_percapita_lucode_[LUCODE]', lucode
                ] for lucode, _ in lucode_to_search_radii],
                target_path=file_registry['urban_nature_supply_percapita']),
            task_name='2SFCA - urban nature supply total',
            target_path_list=[file_registry['urban_nature_supply_percapita']],
            dependent_task_list=partial_urban_nature_supply_percapita_tasks
        )

    # Search radius mode 3: search radii are defined per population group.
    elif args['search_radius_mode'] == RADIUS_OPT_POP_GROUP:
        LOGGER.info("Running model with search radius mode "
                    f"{RADIUS_OPT_POP_GROUP}")
        urban_nature_reclassification_task = graph.add_task(
            _reclassify_urban_nature_area,
            kwargs={
                'lulc_raster_path': file_registry['masked_lulc'],
                'lulc_attribute_table': args['lulc_attribute_table'],
                'target_raster_path': file_registry['urban_nature_area'],
            },
            target_path_list=[file_registry['urban_nature_area']],
            task_name='Identify urban nature areas',
            dependent_task_list=[lulc_mask_task]
        )

        decayed_population_in_group_tasks = []
        for pop_group in split_population_fields:
            search_radius_m = search_radii_by_pop_group[pop_group]

            _ = graph.add_task(
                _convolve_and_set_lower_bound,
                kwargs={
                    "signal_path_band": (file_registry['urban_nature_area'], 1),
                    "kernel_path_band": (file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                    "target_path": file_registry['accessible_urban_nature_to_[POP_GROUP]', pop_group],
                    "working_dir": args['workspace_dir'],
                },
                task_name='Accessible urban nature',
                target_path_list=[file_registry['accessible_urban_nature_to_[POP_GROUP]', pop_group]],
                dependent_task_list=[urban_nature_reclassification_task]
            )

            decayed_population_in_group_tasks.append(graph.add_task(
                _convolve_and_set_lower_bound,
                kwargs={
                    'signal_path_band': (
                        file_registry['population_in_[POP_GROUP]', pop_group], 1),
                    'kernel_path_band': (
                        file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                    'target_path': file_registry[
                        'distance_weighted_population_in_[POP_GROUP]', pop_group],
                    'working_dir': args['workspace_dir'],
                },
                task_name=f'Convolve population - {search_radius_m}m',
                target_path_list=[file_registry[
                    'distance_weighted_population_in_[POP_GROUP]', pop_group]],
                dependent_task_list=[
                    kernel_tasks[search_radius_m],
                    proportional_population_tasks[pop_group]]
            ))

        sum_of_decayed_population_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_sum_op,
                rasters=[file_registry[
                    'distance_weighted_population_in_[POP_GROUP]', pop_group
                ] for pop_group in split_population_fields],
                target_path=file_registry['distance_weighted_population_all_groups']),
            task_name='2SFCA - urban nature supply total',
            target_path_list=[file_registry['distance_weighted_population_all_groups']],
            dependent_task_list=decayed_population_in_group_tasks
        )

        urban_nature_population_ratio_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_urban_nature_population_ratio,
                rasters=[
                    file_registry['urban_nature_area'],
                    file_registry['distance_weighted_population_all_groups']],
                target_path=file_registry['urban_nature_population_ratio']),
            task_name=(
                '2SFCA: Calculate R_j urban nature/population ratio - '
                f'{search_radius_m}'),
            target_path_list=[
                file_registry['urban_nature_population_ratio']],
            dependent_task_list=[
                urban_nature_reclassification_task,
                sum_of_decayed_population_task
            ])

        urban_nature_supply_percapita_by_group_tasks = []
        urban_nature_balance_totalpop_by_group_tasks = []
        supply_population_tasks = {'over': {}, 'under': {}}
        for pop_group in split_population_fields:
            search_radius_m = search_radii_by_pop_group[pop_group]
            urban_nature_supply_percapita_by_group_task = graph.add_task(
                _convolve_and_set_lower_bound,
                kwargs={
                    'signal_path_band': (
                        file_registry['urban_nature_population_ratio'], 1),
                    'kernel_path_band': (
                        file_registry['kernel_[SEARCH_RADIUS]', search_radius_m], 1),
                    'target_path': file_registry[
                        'urban_nature_supply_percapita_to_[POP_GROUP]', pop_group],
                    'working_dir': args['workspace_dir'],
                },
                task_name=f'2SFCA - urban nature supply for {pop_group}',
                target_path_list=[file_registry[
                    'urban_nature_supply_percapita_to_[POP_GROUP]', pop_group]],
                dependent_task_list=[
                    kernel_tasks[search_radius_m],
                    urban_nature_population_ratio_task])
            urban_nature_supply_percapita_by_group_tasks.append(
                urban_nature_supply_percapita_by_group_task)

            per_cap_urban_nature_balance_pop_group_task = graph.add_task(
                _calculate_urban_nature_balance_percapita,
                kwargs={
                    'urban_nature_supply_path': file_registry[
                        'urban_nature_supply_percapita_to_[POP_GROUP]', pop_group],
                    'urban_nature_demand': args['urban_nature_demand'],
                    'target_path': file_registry[
                        'urban_nature_balance_percapita_[POP_GROUP]', pop_group]},
                task_name=(
                    f'Calculate per-capita urban nature balance-{pop_group}'),
                target_path_list=[file_registry[
                    'urban_nature_balance_percapita_[POP_GROUP]', pop_group]],
                dependent_task_list=[
                    urban_nature_supply_percapita_by_group_task
                ])

            urban_nature_balance_totalpop_by_group_tasks.append(graph.add_task(
                pygeoprocessing.raster_map,
                kwargs=dict(
                    op=_urban_nature_balance_totalpop_op,
                    rasters=[
                        file_registry['urban_nature_balance_percapita_[POP_GROUP]',
                                      pop_group],
                        file_registry['population_in_[POP_GROUP]', pop_group]
                    ],
                    target_path=file_registry[
                        'urban_nature_balance_[POP_GROUP]', pop_group]),
                task_name='Calculate per-capita urban nature supply-demand',
                target_path_list=[file_registry[
                    'urban_nature_balance_[POP_GROUP]', pop_group]],
                dependent_task_list=[
                    per_cap_urban_nature_balance_pop_group_task,
                    proportional_population_tasks[pop_group]
                ]))

            for supply_type, op in [('under', numpy.less),
                                    ('over', numpy.greater)]:
                supply_population_tasks[
                    supply_type][pop_group] = graph.add_task(
                    pygeoprocessing.raster_calculator,
                    kwargs={
                        'base_raster_path_band_const_list': [
                            (file_registry['population_in_[POP_GROUP]', pop_group], 1),
                            (file_registry[
                                'urban_nature_balance_percapita_[POP_GROUP]', pop_group], 1),
                            (op, 'raw'),  # numpy element-wise comparator
                        ],
                        'local_op': _filter_population,
                        'target_raster_path': file_registry[
                            f'{supply_type}supplied_population_[POP_GROUP]', pop_group],
                        'datatype_target': gdal.GDT_Float32,
                        'nodata_target': FLOAT32_NODATA,
                    },
                    task_name=(
                        f'Determine {supply_type}supplied populations to '
                        f'{pop_group}'),
                    target_path_list=[file_registry[
                        f'{supply_type}supplied_population_[POP_GROUP]', pop_group]],
                    dependent_task_list=[
                        per_cap_urban_nature_balance_pop_group_task,
                        proportional_population_tasks[pop_group]
                    ])

        urban_nature_supply_percapita_task = graph.add_task(
            _weighted_sum,
            kwargs={
                'raster_path_list': [file_registry[
                    'urban_nature_supply_percapita_to_[POP_GROUP]', group
                ] for group in sorted(split_population_fields)],
                'weight_raster_list': [file_registry[
                    'proportion_of_population_in_[POP_GROUP]', group
                ] for group in sorted(split_population_fields)],
                'target_path': file_registry['urban_nature_supply_percapita'],
            },
            task_name='2SFCA - urban nature supply total',
            target_path_list=[file_registry['urban_nature_supply_percapita']],
            dependent_task_list=[
                *urban_nature_supply_percapita_by_group_tasks,
                *pop_group_proportion_tasks.values()
            ])

        per_capita_urban_nature_balance_task = graph.add_task(
            _calculate_urban_nature_balance_percapita,
            kwargs={
                'urban_nature_supply_path':
                    file_registry['urban_nature_supply_percapita'],
                'urban_nature_demand': args['urban_nature_demand'],
                'target_path':
                    file_registry['urban_nature_balance_percapita']},
            task_name=(
                'Calculate per-capita urban nature balance}'),
            target_path_list=[
                file_registry['urban_nature_balance_percapita']],
            dependent_task_list=[
                urban_nature_supply_percapita_task
            ])

        urban_nature_balance_totalpop_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=_sum_op,
                rasters=[file_registry['urban_nature_balance_[POP_GROUP]', pop_group]
                    for pop_group in split_population_fields],
                target_path=file_registry['urban_nature_balance_totalpop']),
            task_name='2SFCA - urban nature - total population',
            target_path_list=[
                file_registry['urban_nature_balance_totalpop']],
            dependent_task_list=urban_nature_balance_totalpop_by_group_tasks
        )

        # Summary stats for RADIUS_OPT_POP_GROUP
        _ = graph.add_task(
            _supply_demand_vector_for_pop_groups,
            kwargs={
                'source_aoi_vector_path': file_registry['reprojected_admin_boundaries'],
                'target_aoi_vector_path': file_registry['admin_boundaries'],
                'file_registry': file_registry
            },
            task_name=(
                'Aggregate supply-demand to admin units (by pop groups)'),
            target_path_list=[file_registry['admin_boundaries']],
            dependent_task_list=[
                aoi_reprojection_task,
                *urban_nature_balance_totalpop_by_group_tasks,
                *proportional_population_tasks.values(),
                *supply_population_tasks['under'].values(),
                *supply_population_tasks['over'].values()
            ])

    # Greenspace budget, supply/demand and over/undersupply rasters are the
    # same for uniform radius and for split urban_nature modes.
    if args['search_radius_mode'] in (RADIUS_OPT_UNIFORM,
                                      RADIUS_OPT_URBAN_NATURE):
        # This is "SUP_DEMi_cap" from the user's guide
        per_capita_urban_nature_balance_task = graph.add_task(
            _calculate_urban_nature_balance_percapita,
            kwargs={
                'urban_nature_supply_path':
                    file_registry['urban_nature_supply_percapita'],
                'urban_nature_demand': args['urban_nature_demand'],
                'target_path':
                    file_registry['urban_nature_balance_percapita']},
            task_name=(
                'Calculate per-capita urban nature balance'),
            target_path_list=[
                file_registry['urban_nature_balance_percapita']],
            dependent_task_list=[
                urban_nature_supply_percapita_task
            ])

        # This is "SUP_DEMi" from the user's guide
        urban_nature_balance_totalpop_task = graph.add_task(
            pygeoprocessing.raster_map,
            kwargs=dict(
                op=_urban_nature_balance_totalpop_op,
                rasters=[
                    file_registry['urban_nature_balance_percapita'],
                    file_registry['masked_population']
                ],
                target_path=file_registry['urban_nature_balance_totalpop']
            ),
            task_name='Calculate urban nature balance for the total population',
            target_path_list=[
                file_registry['urban_nature_balance_totalpop']],
            dependent_task_list=[
                 per_capita_urban_nature_balance_task,
                 population_mask_task
            ])

        supply_population_tasks = []
        pop_paths = [(None, file_registry['masked_population'])]
        if args['aggregate_by_pop_group']:
            pop_paths.extend([(pop_group, file_registry['population_in_[POP_GROUP]',
                pop_group]) for pop_group in split_population_fields])

        for pop_group, proportional_pop_path in pop_paths:
            if pop_group is not None:
                pop_group = pop_group[4:]  # trim leading 'pop_'
            for supply_type, op in [('under', numpy.less),
                                    ('over', numpy.greater)]:
                if pop_group is None:
                    supply_population_path = file_registry[f'{supply_type}supplied_population']
                else:
                    supply_population_path = file_registry[
                        f'{supply_type}supplied_population_[POP_GROUP]', pop_group]

                supply_population_tasks.append(graph.add_task(
                    pygeoprocessing.raster_calculator,
                    kwargs={
                        'base_raster_path_band_const_list': [
                            (proportional_pop_path, 1),
                            (file_registry['urban_nature_balance_percapita'], 1),
                            (op, 'raw'),  # numpy element-wise comparator
                        ],
                        'local_op': _filter_population,
                        'target_raster_path': supply_population_path,
                        'datatype_target': gdal.GDT_Float32,
                        'nodata_target': FLOAT32_NODATA,
                    },
                    task_name=f'Determine {supply_type}supplied populations',
                    target_path_list=[supply_population_path],
                    dependent_task_list=[
                        per_capita_urban_nature_balance_task,
                        population_mask_task,
                        *list(proportional_population_tasks.values())
                    ]))

        _ = graph.add_task(
            _supply_demand_vector_for_single_raster_modes,
            kwargs={
                'source_aoi_vector_path': file_registry['reprojected_admin_boundaries'],
                'target_aoi_vector_path': file_registry['admin_boundaries'],
                'urban_nature_budget_path': file_registry[
                    'urban_nature_balance_totalpop'],
                'population_path': file_registry['masked_population'],
                'undersupplied_populations_path': file_registry[
                    'undersupplied_population'],
                'oversupplied_populations_path': file_registry[
                    'oversupplied_population'],
                'include_pop_groups': args['aggregate_by_pop_group'],
            },
            task_name=(
                'Aggregate supply-demand to admin units (single rasters)'),
            target_path_list=[file_registry['admin_boundaries']],
            dependent_task_list=[
                population_mask_task,
                aoi_reprojection_task,
                urban_nature_balance_totalpop_task,
                *supply_population_tasks
            ])

    graph.close()
    graph.join()
    LOGGER.info('Finished Urban Nature Access Model')
    return file_registry.registry


# Sum a list of arrays element-wise
def _sum_op(*array_list): return numpy.sum(array_list, axis=0)


def _geometries_overlap(vector_path):
    """Check if the geometries of the vector's first layer overlap.

    Args:
        vector_path (string): The path to a GDAL vector.

    Returns:
        bool: Whether there's numerically significant overlap between polygons
            in the first layer.

    """
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    area_sum = 0
    geometries = []
    for feature in layer:
        ogr_geom = feature.GetGeometryRef()
        area_sum += ogr_geom.Area()
        shapely_geom = shapely.wkb.loads(bytes(ogr_geom.ExportToWkb()))
        geometries.append(shapely_geom)

    layer = None
    vector = None

    union_area = shapely.ops.unary_union(geometries).area
    LOGGER.debug(
        f"Vector has a union area of {union_area} and area sum of "
        f"{area_sum},so about {round((1-(union_area/area_sum))*100, 2)}% of "
        f"the area overlaps in vector {vector_path}")
    if math.isclose(union_area, area_sum):
        return False
    return True


def _reproject_and_identify(base_vector_path, target_projection_wkt,
                            target_path, driver_name, id_fieldname,
                            target_layer_name):
    """Reproject a vector and add an ID field.

    Args:
        base_vector_path (string): The string path to the source vector.
        target_projection_wkt (string): The WKT of the target projection.
        target_path (string): The string path to where the new vector should be
            saved.
        driver_name (string): The GDAL driver name of the target vector.
        id_fieldname (string): The name of the ID field.  A new field with this
            name and an integer type will be created in the target vector.
            Each feature in the target vector will be assigned a unique integer
            ID.
        target_layer_name (string): The layer name of the target vector.

    Returns:
        ``None``
    """
    pygeoprocessing.reproject_vector(
        base_vector_path, target_projection_wkt, target_path,
        target_layer_name=target_layer_name,
        driver_name=driver_name)

    vector = gdal.OpenEx(target_path, gdal.GA_Update)
    layer = vector.GetLayer()
    field = ogr.FieldDefn(id_fieldname, ogr.OFTInteger)
    layer.CreateField(field)

    layer.StartTransaction()
    for field_id, feature in enumerate(layer):
        feature.SetField(id_fieldname, field_id)
        layer.SetFeature(feature)
    layer.CommitTransaction()
    layer = None
    vector = None


def _weighted_sum(raster_path_list, weight_raster_list, target_path):
    """Create a spatially-weighted sum.

    Args:
        raster_path_list (list): A list of raster paths containing values to
            weight and sum.
        weight_raster_list (list): A list of raster paths containing weights.
        target_path (str): The path to where the output raster should be
            stored.

    Returns
        ``None``
    """
    assert len(raster_path_list) == len(weight_raster_list)

    raster_nodata_list = [pygeoprocessing.get_raster_info(path)['nodata'][0]
                          for path in raster_path_list]
    weight_nodata_list = [pygeoprocessing.get_raster_info(path)['nodata'][0]
                          for path in weight_raster_list]

    def _weight_and_sum(*args):
        pixel_arrays = args[:int(len(args)/2)]
        weight_arrays = args[int(len(args)/2):]

        target_array = numpy.zeros(pixel_arrays[0].shape, dtype=numpy.float32)
        touched_pixels = numpy.zeros(target_array.shape, dtype=bool)
        for source_array, weight_array, source_nodata, weight_nodata in zip(
                pixel_arrays, weight_arrays, raster_nodata_list, weight_nodata_list):
            valid_pixels = (
                ~pygeoprocessing.array_equals_nodata(source_array, source_nodata) &
                ~pygeoprocessing.array_equals_nodata(weight_array, weight_nodata))
            touched_pixels |= valid_pixels
            target_array[valid_pixels] += (
                source_array[valid_pixels] * weight_array[valid_pixels])

        # Any pixels that were not touched, set them to nodata.
        target_array[~touched_pixels] = FLOAT32_NODATA
        return target_array

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list + weight_raster_list],
        _weight_and_sum, target_path, gdal.GDT_Float32, FLOAT32_NODATA)


def _reclassify_and_multiply(
        aois_raster_path, reclassification_map, supply_raster_path,
        target_raster_path):
    """Create a raster of urban nature supply given areas of interest.

    This is done by:

        1. Reclassifying AOI IDs to population group ratios and then
        2. Multiplying the population group ratios by the urban nature supply.

    Args:
        aois_raster_path (string): The path to a raster of integers
            identifying which admin unit a pixel belongs to.
        reclassification_map (dict): A dict mapping integer admin unit IDs to
            float population proportions (values 0-1) for a given population
            group.
        supply_raster_path (string): A string path to a raster of urban nature
            supply values for the total population.
        target_raster_path (string): The string path to where the resulting
            supply-to-group raster should be written.

    Returns:
        ``None``
    """
    pygeoprocessing.reclassify_raster(
        (aois_raster_path, 1), reclassification_map, target_raster_path,
        gdal.GDT_Float32, FLOAT32_NODATA)

    pop_group_raster = gdal.OpenEx(target_raster_path,
                                   gdal.GA_Update | gdal.OF_RASTER)
    pop_group_band = pop_group_raster.GetRasterBand(1)
    pop_group_nodata = pop_group_band.GetNoDataValue()
    supply_raster = gdal.OpenEx(supply_raster_path,
                                gdal.GA_ReadOnly | gdal.OF_RASTER)
    supply_band = supply_raster.GetRasterBand(1)
    supply_nodata = supply_band.GetNoDataValue()
    for block_info in pygeoprocessing.iterblocks((target_raster_path, 1),
                                                 offset_only=True):
        pop_group_proportion_block = pop_group_band.ReadAsArray(**block_info)
        supply_block = supply_band.ReadAsArray(**block_info)

        valid_mask = (
            ~pygeoprocessing.array_equals_nodata(
                pop_group_proportion_block, pop_group_nodata) &
            ~pygeoprocessing.array_equals_nodata(supply_block, supply_nodata))
        target_block = numpy.full(supply_block.shape, FLOAT32_NODATA,
                                  dtype=numpy.float32)
        target_block[valid_mask] = (
            pop_group_proportion_block[valid_mask] * supply_block[valid_mask])
        pop_group_band.WriteArray(
            target_block, xoff=block_info['xoff'], yoff=block_info['yoff'])

    pop_group_band = None
    pop_group_raster = None
    supply_band = None
    supply_raster = None


def _read_field_from_vector(vector_path, key_field, value_field):
    """Read a field from a vector's first layer.

    Args:
        vector_path (string): The string path to a vector.
        key_field (string): The string key field within the vector.
            ``key_field`` must exist within the vector at ``vector_path``.
            ``key_field`` is case-sensitive.
        value_field (string): The string value field within the vector.
            ``value_field`` must exist within the vector at ``vector_path``.
            ``value_field`` is case-sensitive.

    Returns:
        attribute_map (dict): A dict mapping each ``key_field`` key to
            the corresponding ``value_field`` value.
    """
    vector = gdal.OpenEx(vector_path)
    layer = vector.GetLayer()
    attribute_map = {}
    for feature in layer:
        if key_field == 'FID':
            key = feature.GetFID()
        else:
            key = feature.GetField(key_field)
        attribute_map[key] = feature.GetField(value_field)
    return attribute_map


def _rasterize_aois(base_raster_path, aois_vector_path,
                    target_raster_path, id_fieldname):
    """Rasterize the admin units vector onto a new raster.

    Args:
        base_raster_path (string): The string path to a raster on disk to be
            used as a template raster.
        aois_vector_path (string): The path to a vector on disk of areas of
            interest, typically administrative units.  The ``id_fieldname``
            feature of the features in this vector will be rasterized onto a
            new raster.
        target_raster_path (string): The path to a new UInt32 raster created on
            disk with new values burned into it.
        id_fieldname (string): The fieldname of the ID field to rasterize.

    Returns:
        ``None``
    """
    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_raster_path, gdal.GDT_UInt32,
        [UINT32_NODATA], [UINT32_NODATA])

    pygeoprocessing.rasterize(
        aois_vector_path, target_raster_path,
        option_list=[f"ATTRIBUTE={id_fieldname}"])


def _reclassify_urban_nature_area(
        lulc_raster_path, lulc_attribute_table, target_raster_path,
        only_these_urban_nature_codes=None):
    """Reclassify LULC pixels into the urban nature area they represent.

    After execution, urban nature pixels will have values representing the
    pixel's area of urban nature (pixel area * proportion of urban nature),
    while pixels that are not urban nature will have a pixel value of 0.
    Nodata values will propagate to the output raster.

    Args:
        lulc_raster_path (string): The path to a land-use/land-cover raster.
        lulc_attribute_table (string): The path to a CSV table representing
            LULC attributes.  Must have "lucode" and "urban_nature" columns.
            The "urban_nature" column represents a proportion 0-1 of how much
            of the pixel's area represents urban nature.
        target_raster_path (string): Where the reclassified urban nature raster
            should be written.
        only_these_urban_nature_codes=None (iterable or None): If ``None``, all
            lucodes with a ``urban_nature`` value of 1 will be reclassified to
            1.  If an iterable, must be an iterable of landuse codes matching
            codes in the lulc attribute table.  Only these landcover codes will
            have urban nature area classified in the target raster path.

    Returns:
        ``None``
    """
    lulc_attribute_df = MODEL_SPEC.get_input(
        'lulc_attribute_table').get_validated_dataframe(lulc_attribute_table)

    squared_pixel_area = abs(
        numpy.multiply(*_square_off_pixels(lulc_raster_path)))

    if only_these_urban_nature_codes:
        valid_urban_nature_codes = set(only_these_urban_nature_codes)
    else:
        valid_urban_nature_codes = set(
            lulc_attribute_df[lulc_attribute_df['urban_nature'] > 0].index)

    urban_nature_area_map = {}
    for row in lulc_attribute_df[['urban_nature']].itertuples():
        lucode = row.Index
        urban_nature_proportion = row.urban_nature
        urban_nature_area = 0
        if lucode in valid_urban_nature_codes:
            urban_nature_area = squared_pixel_area * urban_nature_proportion
        urban_nature_area_map[lucode] = urban_nature_area

    lulc_raster_nodata = pygeoprocessing.get_raster_info(
        lulc_raster_path)['nodata'][0]
    if lulc_raster_nodata is not None:
        urban_nature_area_map[lulc_raster_nodata] = FLOAT32_NODATA

    utils.reclassify_raster(
        raster_path_band=(lulc_raster_path, 1),
        value_map=urban_nature_area_map,
        target_raster_path=target_raster_path,
        target_datatype=gdal.GDT_Float32,
        target_nodata=FLOAT32_NODATA,
        error_details={
            'raster_name': MODEL_SPEC.get_input('lulc_raster_path').name,
            'column_name': 'urban_nature',
            'table_name': MODEL_SPEC.get_input('lulc_attribute_table').name
        }
    )


def _filter_population(population, urban_nature_budget, numpy_filter_op):
    """Filter the population by a defined op and the urban nature budget.

    Note:
        The ``population`` and ``urban_nature_budget`` inputs must have the same
        shape and must both use ``FLOAT32_NODATA`` as their nodata value.

    Args:
        population (numpy.array): A numpy array with population counts.
        urban_nature_budget (numpy.array): A numpy array with the urban nature
            budget values.
        numpy_filter_op (callable): A function that takes a numpy array as
            parameter 1 and a scalar value as parameter 2.  This function must
            return a boolean numpy array of the same shape as parameter 1.

    Returns:
        A ``numpy.array`` with the population values where the
        ``urban_nature_budget`` pixels match the ``numpy_filter_op``.
    """
    population_matching_filter = numpy.full(
        population.shape, FLOAT32_NODATA, dtype=numpy.float32)
    valid_pixels = (
        ~numpy.isclose(urban_nature_budget, FLOAT32_NODATA) &
        ~numpy.isclose(population, FLOAT32_NODATA))

    population_matching_filter[valid_pixels] = numpy.where(
        numpy_filter_op(urban_nature_budget[valid_pixels], 0),
        population[valid_pixels],  # If condition is true, use population
        0  # If condition is false, use 0
    )
    return population_matching_filter


def _supply_demand_vector_for_pop_groups(
        source_aoi_vector_path, target_aoi_vector_path, file_registry):
    """Write a supply-demand vector when rasters are by population group.

    Args:
        source_aoi_vector_path (str): The source AOI vector path.
        target_aoi_vector_path (str): The target AOI vector path.
        file_registry (FileRegistry): used to look up raster paths to aggregate

    Returns:
        ``None``
    """
    def _get_zonal_stats(raster_path):
        return pygeoprocessing.zonal_statistics(
            (raster_path, 1), source_aoi_vector_path)

    pop_group_fields = []
    feature_ids = set()
    vector = gdal.OpenEx(source_aoi_vector_path)
    layer = vector.GetLayer()
    for feature in layer:
        feature_ids.add(feature.GetFID())
    pop_group_fields = []
    for field_defn in layer.schema:
        fieldname = field_defn.GetName()
        if re.match(POP_FIELD_REGEX, fieldname):
            pop_group_fields.append(fieldname)
    layer = None
    vector = None

    sums = {
        'supply-demand': collections.defaultdict(float),
        'population': collections.defaultdict(float),
        'oversupply': collections.defaultdict(float),
        'undersupply': collections.defaultdict(float),
    }
    stats_by_feature = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for pop_group_field in pop_group_fields:
        # trim the leading 'pop_'
        groupname = re.sub(POP_FIELD_REGEX, '', pop_group_field)

        urban_nature_sup_dem_stats = _get_zonal_stats(
            file_registry['urban_nature_balance_[POP_GROUP]', pop_group_field])
        proportional_pop_stats = _get_zonal_stats(
            file_registry['population_in_[POP_GROUP]', pop_group_field])
        undersupply_stats = _get_zonal_stats(
            file_registry['undersupplied_population_[POP_GROUP]', pop_group_field])
        oversupply_stats = _get_zonal_stats(
            file_registry['oversupplied_population_[POP_GROUP]', pop_group_field])

        for feature_id in feature_ids:
            group_population_in_region = proportional_pop_stats[
                feature_id]['sum']
            group_sup_dem_in_region = urban_nature_sup_dem_stats[
                feature_id]['sum']
            group_oversupply_in_region = oversupply_stats[feature_id]['sum']
            group_undersupply_in_region = undersupply_stats[feature_id]['sum']
            stats_by_feature[feature_id][f'SUP_DEMadm_cap_{groupname}'] = (
                group_sup_dem_in_region / group_population_in_region)
            stats_by_feature[feature_id][f'Pund_adm_{groupname}'] = (
                group_undersupply_in_region)
            stats_by_feature[feature_id][f'Povr_adm_{groupname}'] = (
                group_oversupply_in_region)
            sums['supply-demand'][feature_id] += group_sup_dem_in_region
            sums['population'][feature_id] += group_population_in_region
            sums['oversupply'][feature_id] += group_oversupply_in_region
            sums['undersupply'][feature_id] += group_undersupply_in_region

    for feature_id in feature_ids:
        stats_by_feature[feature_id]['SUP_DEMadm_cap'] = (
            sums['supply-demand'][feature_id] / sums['population'][feature_id])
        stats_by_feature[feature_id]['Pund_adm'] = (
            sums['undersupply'][feature_id])
        stats_by_feature[feature_id]['Povr_adm'] = (
            sums['oversupply'][feature_id])

    _write_supply_demand_vector(
        source_aoi_vector_path, stats_by_feature, target_aoi_vector_path)


def _supply_demand_vector_for_single_raster_modes(
        source_aoi_vector_path,
        target_aoi_vector_path,
        urban_nature_budget_path,
        population_path,
        undersupplied_populations_path,
        oversupplied_populations_path,
        include_pop_groups=False):
    """Create summary vector for modes with single-raster summary stats.

    Args:
        source_aoi_vector_path (str): Path to the source aois vector.
        target_aoi_vector_path (str): Path to where the target aois vector
            should be written.
        urban_nature_budget_path (str): Path to a raster of urban nature
            supply/demand budget.
        population_path (str): Path to a population raster.
        undersupplied_populations_path (str): Path to a raster of oversupplied
            population per pixel.
        oversupplied_populations_path (str): Path to a raster of undersupplied
            population per pixel.
        include_pop_groups=False (bool): Whether to include population groups
            if they are present in the source AOI vector.

    Returns:
        ``None``
    """
    def _get_zonal_stats(raster_path):
        return pygeoprocessing.zonal_statistics(
            (raster_path, 1), source_aoi_vector_path)

    urban_nature_budget_stats = _get_zonal_stats(urban_nature_budget_path)
    population_stats = _get_zonal_stats(population_path)
    undersupplied_stats = _get_zonal_stats(undersupplied_populations_path)
    oversupplied_stats = _get_zonal_stats(oversupplied_populations_path)

    pop_group_fields = []
    group_names = {}  # {fieldname: groupname}
    pop_proportions_by_fid = collections.defaultdict(dict)
    if include_pop_groups:
        pop_group_fields = list(
            filter(lambda x: re.match(POP_FIELD_REGEX, x),
                   validation.load_fields_from_vector(source_aoi_vector_path)))
        for pop_group_field in pop_group_fields:
            for id_field, value in _read_field_from_vector(
                    source_aoi_vector_path, 'FID',
                    pop_group_field).items():
                group = pop_group_field[4:]  # trim leading 'pop_'
                group_names[pop_group_field] = group
                pop_proportions_by_fid[id_field][group] = value

    stats_by_feature = {}
    for fid in urban_nature_budget_stats.keys():
        stats = {
            'Pund_adm': undersupplied_stats[fid]['sum'],
            'Povr_adm': oversupplied_stats[fid]['sum'],
        }

        # Handle the case where an administrative unit might overlap no people
        if population_stats[fid]['sum'] == 0:
            per_capita_supply = float('nan')
        else:
            per_capita_supply = (
                urban_nature_budget_stats[fid]['sum'] /
                population_stats[fid]['sum'])
        stats['SUP_DEMadm_cap'] = per_capita_supply

        for pop_group_field in pop_group_fields:
            group = group_names[pop_group_field]
            group_proportion = pop_proportions_by_fid[fid][group]
            for prefix, supply_stats in [('Pund', undersupplied_stats),
                                         ('Povr', oversupplied_stats)]:
                stats[f'{prefix}_adm_{group}'] = (
                    supply_stats[fid]['sum'] * group_proportion)
        stats_by_feature[fid] = stats

    _write_supply_demand_vector(
        source_aoi_vector_path, stats_by_feature, target_aoi_vector_path)


def _write_supply_demand_vector(source_aoi_vector_path, feature_attrs,
                                target_aoi_vector_path):
    """Write data to a copy of an existing AOI vector.

    Args:
        source_aoi_vector_path (str): The source AOI vector path.
        feature_attrs (dict): A dict mapping int feature IDs (GDAL FIDs) to
            dicts mapping fieldnames to field values.
        target_aoi_vector_path (str): The path to where the target vector
            should be written.

    Returns:
        ``None``
    """
    gdal.VectorTranslate(
        target_aoi_vector_path, source_aoi_vector_path,
        format='GPKG',
        preserveFID=True)

    target_vector = gdal.OpenEx(target_aoi_vector_path, gdal.GA_Update)
    target_layer = target_vector.GetLayer()

    for fieldname in next(iter(feature_attrs.values())).keys():
        field = ogr.FieldDefn(fieldname, ogr.OFTReal)
        field.SetWidth(24)
        field.SetPrecision(11)
        target_layer.CreateField(field)

    target_layer.StartTransaction()
    for feature in target_layer:
        feature_id = feature.GetFID()
        for attr_name, attr_value in feature_attrs[feature_id].items():
            # It is possible that attr_value may be a numpy.float32 object,
            # which will raise a cryptic error.  Numpy.float64 will not raise
            # this error.  Casting to float avoids the issue.
            feature.SetField(attr_name, float(attr_value))

        target_layer.SetFeature(feature)
    target_layer.CommitTransaction()

    target_layer = None
    target_vector = None


def _calculate_urban_nature_balance_percapita(
        urban_nature_supply_path, urban_nature_demand, target_path):
    supply_nodata = pygeoprocessing.get_raster_info(
        urban_nature_supply_path)['nodata'][0]

    def _urban_nature_balance_percapita_op(urban_nature_supply,
                                           urban_nature_demand):
        """Calculate the per-capita urban nature balance.

        This is the amount of urban nature that each pixel has above (positive
        values) or below (negative values) the user-defined
        ``urban_nature_demand`` value.

        Args:
            urban_nature_supply (numpy.array): The supply of urban nature
                available to each person in the population.  This is ``Ai`` in
                the User's Guide.
            urban_nature_demand (float): The policy-defined urban nature
            requirement, in square meters per person.

        Returns:
            A ``numpy.array`` of the calculated urban nature budget.
        """
        balance = numpy.full(
            urban_nature_supply.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = ~pygeoprocessing.array_equals_nodata(
            urban_nature_supply, supply_nodata)
        balance[valid_pixels] = (
            urban_nature_supply[valid_pixels] - urban_nature_demand)
        return balance

    pygeoprocessing.raster_calculator(
        base_raster_path_band_const_list=[
            (urban_nature_supply_path, 1), (urban_nature_demand, 'raw')],
        local_op=_urban_nature_balance_percapita_op,
        target_raster_path=target_path,
        datatype_target=gdal.GDT_Float32,
        nodata_target=FLOAT32_NODATA)


def _urban_nature_balance_totalpop_op(urban_nature_balance, population):
    """Calculate the total population urban nature balance.

    Args:
        urban_nature_balance (numpy.array): The area of urban nature budgeted to
            each person, relative to a minimum required per-person area of
            urban nature.  This matrix must have ``FLOAT32_NODATA`` as its nodata
            value.  This matrix must be the same size and shape as
            ``population``.
        population (numpy.array): Pixel values represent the population count
            of the pixel.  This matrix must be the same size and shape as
            ``urban_nature_budget``, and must have ``FLOAT32_NODATA`` as its
            nodata value.

    Returns:
        A ``numpy.array`` of the area (in square meters) of urban nature
        supplied to each individual in each pixel.
    """
    return urban_nature_balance * population


def _urban_nature_population_ratio(urban_nature_area, convolved_population):
    """Calculate the urban nature-population ratio R_j.

    Args:
        urban_nature_area (numpy.array): A numpy array representing the area
            of urban nature in the pixel.  Pixel values will be ``0`` if
            there is no urban nature.  Pixel values may also match
            ``urban_nature_nodata``.
        convolved_population (numpy.array): A numpy array where each pixel
            represents the total number of people within a search radius of
            each pixel, perhaps weighted by a search kernel.

    Returns:
        A numpy array with the ratio ``R_j`` representing the
        urban nature-population ratio with the following constraints:

            * ``convolved_population`` pixels that are numerically close to
              ``0`` are snapped to ``0`` to avoid unrealistically small
              denominators in the final ratio.
            * Any non-urban nature pixels will have a value of ``0`` in the
              output matrix.
    """
    # ASSUMPTION: population nodata value is not close to 0.
    #  Shouldn't be if we're coming from convolution.
    out_array = numpy.full(
        urban_nature_area.shape, FLOAT32_NODATA, dtype=numpy.float32)

    # Small negative values should already have been filtered out in
    # another function after the convolution.
    # This avoids divide-by-zero errors when taking the ratio.
    valid_pixels = (convolved_population > 0)

    # R_j is a ratio only calculated for the urban nature pixels.
    urban_nature_pixels = ~numpy.isclose(urban_nature_area, 0)
    valid_pixels &= urban_nature_pixels

    # The user's guide specifies that if the population in the search
    # radius is numerically 0, the urban nature/population ratio should be
    # set to the urban nature area.
    # A consequence of this is that as the population approaches 0 from the
    # positive side, the ratio will approach infinity.
    # After checking with the science team, we decided that where the
    # population is less than or equal to 1, the calculated
    # urban nature/population ratio would be set to the available urban
    # nature on that pixel.
    population_close_to_zero = (convolved_population <= 1.0)
    out_array[population_close_to_zero] = (
        urban_nature_area[population_close_to_zero])
    out_array[~urban_nature_pixels] = 0

    valid_pixels_with_population = (
        valid_pixels & (~population_close_to_zero))
    out_array[valid_pixels_with_population] = (
        urban_nature_area[valid_pixels_with_population] /
        convolved_population[valid_pixels_with_population])

    # eliminate pixel values < 0
    out_array[valid_pixels & (out_array < 0)] = 0

    return out_array


def _convolve_and_set_lower_bound(
        signal_path_band, kernel_path_band, target_path, working_dir):
    """Convolve a raster and set all values below 0 to 0.

    Args:
        signal_path_band (tuple): A 2-tuple of (signal_raster_path, band_index)
            to use as the signal raster in the convolution.
        kernel_path_band (tuple): A 2-tuple of (kernel_raster_path, band_index)
            to use as the kernel raster in the convolution.  This kernel should
            be non-normalized.
        target_path (string): Where the target raster should be written.
        working_dir (string): The working directory that
            ``pygeoprocessing.convolve_2d`` may use for its intermediate files.

    Returns:
        ``None``
    """
    pygeoprocessing.convolve_2d(
        signal_path_band=signal_path_band,
        kernel_path_band=kernel_path_band,
        target_path=target_path,
        working_dir=working_dir)

    # Sometimes there are negative values that should have been clamped to 0 in
    # the convolution but weren't, so let's clamp them to avoid support issues
    # later on.
    target_raster = gdal.OpenEx(target_path, gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    target_nodata = target_band.GetNoDataValue()
    for block_data, block in pygeoprocessing.iterblocks(
            (target_path, 1)):
        valid_pixels = ~pygeoprocessing.array_equals_nodata(block, target_nodata)
        block[(block < 0) & valid_pixels] = 0
        target_band.WriteArray(
            block, xoff=block_data['xoff'], yoff=block_data['yoff'])

    target_band = None
    target_raster = None


def _square_off_pixels(raster_path):
    """Create square pixels from the provided raster.

    The pixel dimensions produced will respect the sign of the original pixel
    dimensions and will be the mean of the absolute source pixel dimensions.

    Args:
        raster_path (string): The path to a raster on disk.

    Returns:
        A 2-tuple of ``(pixel_width, pixel_height)``, in projected units.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_width, pixel_height = raster_info['pixel_size']

    if abs(pixel_width) == abs(pixel_height):
        return (pixel_width, pixel_height)

    pixel_tuple = ()
    average_absolute_size = (abs(pixel_width) + abs(pixel_height)) / 2
    for pixel_dimension_size in (pixel_width, pixel_height):
        # This loop allows either or both pixel dimension(s) to be negative
        sign_factor = 1
        if pixel_dimension_size < 0:
            sign_factor = -1

        pixel_tuple += (average_absolute_size * sign_factor,)

    return pixel_tuple


def _resample_population_raster(
        source_population_raster_path, target_population_raster_path,
        lulc_pixel_size, lulc_bb, lulc_projection_wkt, working_dir):
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

    Args:
        source_population_raster_path (string): The source population raster.
            Pixel values represent the number of people occupying the pixel.
            Must be linearly projected in meters.
        target_population_raster_path (string): The path to where the target,
            warped population raster will live on disk.
        lulc_pixel_size (tuple): A tuple of the pixel size for the target
            raster.  Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_bb (tuple): A tuple of the bounding box for the target raster.
            Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_projection_wkt (string): The Well-Known Text of the target
            spatial reference fro the target raster.  Passed directly to
            ``pygeoprocessing.warp_raster``.  Assumed to be a linear projection
            in meters.
        working_dir (string): The path to a directory on disk.  A new directory
            is created within this directory for the storage of temporary files
            and then deleted upon successful completion of the function.

    Returns:
        ``None``
    """
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
        target_pixel_size=lulc_pixel_size,
        target_raster_path=warped_density_path,
        resample_method='bilinear',
        target_bb=lulc_bb,
        target_projection_wkt=lulc_projection_wkt)

    # Step 3: convert the warped population raster back from density to the
    # population per pixel
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(lulc_projection_wkt)
    # Calculate target pixel area in km to match above
    target_pixel_area = (
        numpy.multiply(*lulc_pixel_size) * target_srs.GetLinearUnits()) / 1e6

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


def _kernel_power(distance, max_distance, beta):
    """Create a power kernel with user-defined beta.

    Args:
        distance (numpy.array): An array of euclidean distances (in pixels)
            from the center of the kernel.
        max_distance (float): The maximum distance of the kernel.  Pixels that
            are more than this number of pixels will have a value of 0.

    Returns:
        ``numpy.array`` with dtype of numpy.float32 and same shape as
        ``distance.
    """
    kernel = numpy.zeros(distance.shape, dtype=numpy.float32)

    # NOTE: The UG expects beta to be negative, but we cannot raise a distance
    # of 0 to a negative exponent.  So, assume that the kernel value at
    # distance == 0 is 1.
    pixels_in_radius = (distance <= max_distance) & (distance > 0)
    kernel[pixels_in_radius] = distance[pixels_in_radius] ** float(beta)
    kernel[distance == 0] = 1
    return kernel


def _kernel_gaussian(distance, max_distance):
    """Create a gaussian kernel.

    Args:
        distance (numpy.array): An array of euclidean distances (in pixels)
            from the center of the kernel.
        max_distance (float): The maximum distance of the kernel.  Pixels that
            are more than this number of pixels will have a value of 0.

    Returns:
        ``numpy.array`` with dtype of numpy.float32 and same shape as
        ``distance.
    """
    kernel = numpy.zeros(distance.shape, dtype=numpy.float32)
    pixels_in_radius = (distance <= max_distance)
    kernel[pixels_in_radius] = (
        (numpy.e ** (-0.5 * ((distance[pixels_in_radius] / max_distance) ** 2))
         - numpy.e ** (-0.5)) / (1 - numpy.e ** (-0.5)))
    return kernel


def _kernel_density(distance, max_distance):
    """Create a kernel based on density.

    Args:
        distance (numpy.array): An array of euclidean distances (in pixels)
            from the center of the kernel.
        max_distance (float): The maximum distance of the kernel.  Pixels that
            are more than this number of pixels will have a value of 0.

    Returns:
        ``numpy.array`` with dtype of numpy.float32 and same shape as
        ``distance.
    """
    kernel = numpy.zeros(distance.shape, dtype=numpy.float32)
    pixels_in_radius = (distance <= max_distance)
    kernel[pixels_in_radius] = (
        0.75 * (1 - (distance[pixels_in_radius] / max_distance) ** 2))
    return kernel


def _create_valid_pixels_nodata_mask(raster_list, target_mask_path):
    """Create a valid pixels mask across a stack of aligned rasters.

    The target raster will have pixel values of 0 where nodata was found
    somewhere in the pixel stack, 1 where no nodata was found.

    Args:
        raster_list (list): A list of string paths to single-band rasters.
        target_mask_path (str): A string path to where the new mask raster
            should be written.

    Returns:
        ``None``
    """
    nodatas = [
        pygeoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_list]

    def _create_mask(*raster_arrays):
        valid_pixels_mask = numpy.ones(raster_arrays[0].shape, dtype=bool)
        for nodata, array in zip(nodatas, raster_arrays):
            valid_pixels_mask &= ~pygeoprocessing.array_equals_nodata(array, nodata)

        return valid_pixels_mask

    pygeoprocessing.raster_calculator(
        [(path, 1) for path in raster_list],
        _create_mask, target_mask_path, gdal.GDT_Byte, nodata_target=255)


def _warp_lulc(source_lulc_path, target_lulc_path, target_pixel_size,
               target_bounding_box):
    """Warp a LULC raster and set a nodata if needed.

    Args:
        source_lulc_path (str): The path to a source LULC raster.
        target_lulc_path (str): The path to the new LULC raster.
        target_pixel_size (tuple): A 2-tuple of the target pixel size.
        target_bounding_box (tuple): A 4-tuple of the target bounding box.

    Returns:
        ``None``.
    """
    source_raster_info = pygeoprocessing.get_raster_info(source_lulc_path)
    target_nodata = source_raster_info['nodata'][0]

    pygeoprocessing.warp_raster(
        source_lulc_path, target_pixel_size, target_lulc_path,
        'near', target_bb=target_bounding_box,
        target_projection_wkt=source_raster_info['projection_wkt'])

    # if there is no defined nodata, set a default value
    if target_nodata is None:
        # Guarantee that our nodata cannot be represented by the datatype -
        # select a nodata value that's out of range.
        target_nodata = pygeoprocessing.choose_nodata(
            source_raster_info['numpy_type']) + 1
        raster = gdal.OpenEx(target_lulc_path, gdal.GA_Update)
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(target_nodata)
        band = None
        raster = None


def _mask_raster(source_raster_path, mask_raster_path, target_raster_path):
    """Convert pixels to nodata given an existing mask raster.

    Args:
        source_raster_path (str): The path to a source raster.
        mask_raster_path (str): The path to a mask raster.  Pixel values must
            be either 0 (invalid) or 1 (valid).
        target_raster_path (str): The path to a new raster on disk.  Pixels
            marked as 0 in the mask raster will be written out as nodata.

    Returns:
        ``None``
    """
    source_raster_info = pygeoprocessing.get_raster_info(source_raster_path)
    source_raster_nodata = source_raster_info['nodata'][0]

    def _mask(array, valid_mask):
        array = array.copy()
        array[valid_mask == 0] = source_raster_nodata
        return array

    pygeoprocessing.raster_calculator(
        [(source_raster_path, 1), (mask_raster_path, 1)], _mask,
        target_raster_path,
        datatype_target=source_raster_info['datatype'],
        nodata_target=source_raster_nodata)


def validate(args, limit_to=None):
    return validation.validate(args, MODEL_SPEC)
