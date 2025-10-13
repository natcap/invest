# -*- coding: utf-8 -*-
"""Coastal Blue Carbon Preprocessor."""
import logging
import os
import time

import pygeoprocessing
import taskgraph
from osgeo import gdal

from .. import gettext
from .. import spec
from .. import utils
from .. import validation
from ..unit_registry import u
from . import coastal_blue_carbon
from ..file_registry import FileRegistry

LOGGER = logging.getLogger(__name__)

MODEL_SPEC = spec.ModelSpec(
    model_id="coastal_blue_carbon_preprocessor",
    model_title=gettext("Coastal Blue Carbon Preprocessor"),
    userguide="coastal_blue_carbon.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=("cbc_pre",),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_lookup_table_path", "landcover_snapshot_csv"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.CSVInput(
            id="lulc_lookup_table_path",
            name=gettext("LULC lookup table"),
            about=gettext(
                "A table mapping LULC codes from the snapshot rasters to the"
                " corresponding LULC class names, and whether or not the class is a"
                " coastal blue carbon habitat."
            ),
            columns=[
                spec.IntegerInput(
                    id="lucode",
                    about=gettext(
                        "LULC code. Every value in the snapshot LULC maps must have a"
                        " corresponding entry in this column."
                    )
                ),
                spec.StringInput(
                    id="lulc-class", about=gettext("Name of the LULC class."), regexp=None
                ),
                spec.BooleanInput(
                    id="is_coastal_blue_carbon_habitat",
                    about=gettext(
                        "Enter TRUE if this LULC class is a coastal blue carbon habitat,"
                        " FALSE if not."
                    )
                )
            ],
            index_col="lucode"
        ),
        spec.CSVInput(
            id="landcover_snapshot_csv",
            name=gettext("LULC snapshots table"),
            about=gettext(
                "A table mapping snapshot years to corresponding LULC maps for each year."
            ),
            columns=[
                spec.IntegerInput(id="snapshot_year", about=gettext("Year to snapshot.")),
                spec.SingleBandRasterInput(
                    id="raster_path",
                    about=gettext(
                        "Map of LULC in the snapshot year. All values in this raster must"
                        " have corresponding entries in the LULC Lookup table."
                    ),
                    data_type=int,
                    units=None,
                    projected=None
                )
            ],
            index_col="snapshot_year"
        )
    ],
    outputs=[
        spec.CSVOutput(
            id="carbon_biophysical_table_template",
            path="outputs_preprocessor/carbon_biophysical_table_template.csv",
            about=gettext(
                "LULC transition matrix. The first column represents the source LULC"
                " class, and the first row represents the destination LULC classes. Cells"
                " are populated with transition states, or left empty if no such"
                " transition occurs."
            ),
            columns=[
                spec.StringOutput(
                    id="lulc-class",
                    about=gettext(
                        "LULC class names matching those in the biophysical table."
                    )
                ),
                spec.OptionStringOutput(
                    id="[LULC]",
                    about=None,
                    options=[
                        spec.Option(key="accum", about="a state of carbon accumulation"),
                        spec.Option(
                            key="disturb",
                            about=(
                                "Carbon disturbance occurred. Replace this with one of"
                                " ‘low-impact-disturb’, ‘med-impact-disturb’, or"
                                " ‘high-impact-disturb’ to indicate the degree of"
                                " disturbance."
                            )
                        ),
                        spec.Option(key="NCC", about="no change in carbon"),
                    ]
                )
            ],
            index_col="lulc-class"
        ),
        spec.CSVOutput(
            id="carbon_pool_transition_template",
            path="outputs_preprocessor/carbon_pool_transition_template.csv",
            about=gettext(
                "Table mapping each LULC type to impact and accumulation information."
                " This is a template that you will fill out to create the biophysical"
                " table input to the main model."
            ),
            columns=[
                spec.IntegerOutput(
                    id="lucode",
                    about=gettext(
                        "The LULC code that represents this LULC class in the LULC"
                        " snapshot rasters."
                    )
                ),
                spec.StringOutput(
                    id="lulc-class",
                    about=gettext(
                        "Name of the LULC class. This label must be unique among the all"
                        " the LULC classes."
                    )
                ),
                spec.NumberOutput(
                    id="biomass-initial",
                    about=gettext(
                        "The initial carbon stocks in the biomass pool for this LULC"
                        " class."
                    ),
                    units=u.megametric_ton / u.hectare
                ),
                spec.NumberOutput(
                    id="soil-initial",
                    about=gettext(
                        "The initial carbon stocks in the soil pool for this LULC class."
                    ),
                    units=u.megametric_ton / u.hectare
                ),
                spec.NumberOutput(
                    id="litter-initial",
                    about=gettext(
                        "The initial carbon stocks in the litter pool for this LULC"
                        " class."
                    ),
                    units=u.megametric_ton / u.hectare
                ),
                spec.NumberOutput(
                    id="biomass-half-life",
                    about=gettext("The half-life of carbon in the biomass pool."),
                    units=u.year
                ),
                spec.RatioOutput(
                    id="biomass-low-impact-disturb",
                    about=gettext(
                        "Proportion of carbon stock in the biomass pool that is disturbed"
                        " when a cell transitions away from this  LULC class in a"
                        " low-impact disturbance."
                    )
                ),
                spec.RatioOutput(
                    id="biomass-med-impact-disturb",
                    about=gettext(
                        "Proportion of carbon stock in the biomass pool that is disturbed"
                        " when a cell transitions away from this LULC class in a"
                        " medium-impact disturbance."
                    )
                ),
                spec.RatioOutput(
                    id="biomass-high-impact-disturb",
                    about=gettext(
                        "Proportion of carbon stock in the biomass pool that is disturbed"
                        " when a cell transitions away from this LULC class in a"
                        " high-impact disturbance."
                    )
                ),
                spec.NumberOutput(
                    id="biomass-yearly-accumulation",
                    about=gettext(
                        "Annual rate of CO2E accumulation in the biomass pool."
                    ),
                    units=u.megametric_ton / u.hectare / u.year
                ),
                spec.NumberOutput(
                    id="soil-half-life",
                    about=gettext("The half-life of carbon in the soil pool."),
                    units=u.year
                ),
                spec.RatioOutput(
                    id="soil-low-impact-disturb",
                    about=gettext(
                        "Proportion of carbon stock in the soil pool that is disturbed"
                        " when a cell transitions away from this LULC class in a"
                        " low-impact disturbance."
                    )
                ),
                spec.RatioOutput(
                    id="soil-med-impact-disturb",
                    about=gettext(
                        "Proportion of carbon stock in the soil pool that is disturbed"
                        " when a cell transitions away from this LULC class in a"
                        " medium-impact disturbance."
                    )
                ),
                spec.RatioOutput(
                    id="soil-high-impact-disturb",
                    about=gettext(
                        "Proportion of carbon stock in the soil pool that is disturbed"
                        " when a cell transitions away from this LULC class in a"
                        " high-impact disturbance."
                    )
                ),
                spec.NumberOutput(
                    id="soil-yearly-accumulation",
                    about=gettext("Annual rate of CO2E accumulation in the soil pool."),
                    units=u.megametric_ton / u.hectare / u.year
                ),
                spec.NumberOutput(
                    id="litter-yearly-accumulation",
                    about=gettext("Annual rate of CO2E accumulation in the litter pool."),
                    units=u.megametric_ton / u.hectare / u.year
                )
            ],
            index_col="lucode"
        ),
        spec.SingleBandRasterOutput(
            id="aligned_lulc_[YEAR]",
            path="outputs_preprocessor/aligned_lulc_[YEAR].tif",
            about=gettext(
                "Copy of LULC map for the given year, aligned and resampled to match all"
                " the other LULC maps."
            ),
            data_type=int,
            units=None
        ),
        spec.TASKGRAPH_CACHE
    ]
)


def execute(args):
    """Coastal Blue Carbon Preprocessor.

    The preprocessor accepts a list of rasters and checks for cell-transitions
    across the rasters.  The preprocessor outputs a CSV file representing a
    matrix of land cover transitions, each cell pre-filled with a string
    indicating whether carbon accumulates or is disturbed as a result of the
    transition, if a transition occurs.

    Args:
        args['workspace_dir'] (string): directory path to workspace
        args['results_suffix'] (string): append to outputs directory name if
            provided
        args['lulc_lookup_table_path'] (string): filepath of lulc lookup table
        args['landcover_csv_path'] (string): filepath to a CSV containing the
            year and filepath to snapshot rasters on disk.  The years may be in
            any order, but must be unique.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    args, file_registry, task_graph = MODEL_SPEC.setup(args)

    snapshots_dict = MODEL_SPEC.get_input(
        'landcover_snapshot_csv').get_validated_dataframe(
            args['landcover_snapshot_csv'])['raster_path'].to_dict()

    # Align the raster stack for analyzing the various transitions.
    min_pixel_size = float('inf')
    source_snapshot_paths = []
    aligned_snapshot_paths = []
    for snapshot_year, raster_path in sorted(
            snapshots_dict.items(), key=lambda x: x[0]):
        source_snapshot_paths.append(raster_path)
        aligned_snapshot_paths.append(file_registry['aligned_lulc_[YEAR]', snapshot_year])
        min_pixel_size = min(
            utils.mean_pixel_size_and_area(
                pygeoprocessing.get_raster_info(raster_path)['pixel_size'])[0],
            min_pixel_size)

    baseline_srs_wkt = pygeoprocessing.get_raster_info(
        snapshots_dict[min(snapshots_dict.keys())])['projection_wkt']
    alignment_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(source_snapshot_paths,
              aligned_snapshot_paths,
              (['near']*len(source_snapshot_paths)),
              (min_pixel_size, -min_pixel_size),
              'intersection'),
        kwargs={'target_projection_wkt': baseline_srs_wkt},
        target_path_list=aligned_snapshot_paths,
        task_name='Align input landcover rasters')

    landcover_df = MODEL_SPEC.get_input(
        'lulc_lookup_table_path').get_validated_dataframe(
            args['lulc_lookup_table_path'])

    _ = task_graph.add_task(
        func=_create_transition_table,
        args=(landcover_df,
              aligned_snapshot_paths,
              file_registry['carbon_pool_transition_template']),
        target_path_list=[file_registry['carbon_pool_transition_template']],
        dependent_task_list=[alignment_task],
        task_name='Determine transitions and write transition table')

    _ = task_graph.add_task(
        func=_create_biophysical_table,
        args=(landcover_df, file_registry['carbon_biophysical_table_template']),
        target_path_list=[file_registry['carbon_biophysical_table_template']],
        task_name='Write biophysical table template')

    task_graph.close()
    task_graph.join()
    return file_registry.registry


def _create_transition_table(landcover_df, lulc_snapshot_list,
                             target_table_path):
    """Create the transition table from a series of landcover snapshots.

    Args:
        landcover_df (pandas.DataFrame: A table mapping integer landcover
            codes to values indicating the landcover class name in the
            ``lulc-class`` column and ``True`` or ``False`` under the
            ``is_coastal_blue_carbon_habitat`` column.
        lulc_snapshot_list (list): A list of string paths to GDAL rasters on
            disk.  All rasters must have the same spatial reference, pixel size
            and dimensions and must also all be integer rasters, where all
            non-nodata pixel values must be represented in the
            ``landcover_df`` dataframe.
        target_table_path (string): A string path to where the target
            transition table should be written.

    Returns:
        ``None``.
    """
    n_rows, n_cols = pygeoprocessing.get_raster_info(
        lulc_snapshot_list[0])['raster_size']
    n_pixels_total = (n_rows * n_cols) * len(lulc_snapshot_list)
    n_pixels_processed = 0

    raster_tuple_list = []
    for raster_path in lulc_snapshot_list:
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        raster_tuple_list.append((raster, band, nodata))

    transition_pairs = set()
    last_log_time = time.time()
    for block_offsets in pygeoprocessing.iterblocks((lulc_snapshot_list[0], 1),
                                                    offset_only=True):
        _, from_band, from_nodata = raster_tuple_list[0]
        from_array = from_band.ReadAsArray(**block_offsets)
        from_band = None

        for (_, to_band, to_nodata) in raster_tuple_list[1:]:
            if time.time() - last_log_time >= 5.0:
                percent_complete = n_pixels_processed / n_pixels_total
                LOGGER.info(
                    "Determining landcover transitions, "
                    f"{percent_complete:.2f}% complete.")

            to_array = to_band.ReadAsArray(**block_offsets)

            # This comparison assumes that our landcover rasters are of an
            # integer type.  When int matrices, we can compare directly to
            # None.
            valid_pixels = (
                ~pygeoprocessing.array_equals_nodata(from_array, from_nodata) &
                ~pygeoprocessing.array_equals_nodata(to_array, to_nodata))
            transition_pairs = transition_pairs.union(
                set(zip(from_array[valid_pixels].flatten(),
                        to_array[valid_pixels].flatten())))

            # Swap the arrays around to use the current 'to_array', 'to_nodata'
            # as the 'from_array', 'from_nodata' in the next iteration.
            from_array, from_nodata = (to_array, to_nodata)
            n_pixels_processed += to_array.size
    raster_tuple_list = None
    to_band = None
    LOGGER.info("Determining landcover transitions, 100.00%% complete.")

    # Mapping of whether the from, to landcover types are coastal blue carbon
    # habitats to the string carbon transition type.
    # The keys are structured as a tuple of two booleans where:
    #  * tuple[0] = whether the FROM transition is CBC habitat
    #  * tuple[1] = whether the TO transition is CBC habitat
    transition_types = {
        (True, True): 'accum',  # veg --> veg
        (False, True): 'accum',  # non-veg --> veg
        (True, False): 'disturb',  # veg --> non-veg
        (False, False): 'NCC',  # non-veg --> non-veg
    }

    sparse_transition_table = {}
    for from_lucode, to_lucode in transition_pairs:
        try:
            from_is_cbc = landcover_df[
                'is_coastal_blue_carbon_habitat'][from_lucode]
            to_is_cbc = landcover_df[
            'is_coastal_blue_carbon_habitat'][to_lucode]
        except KeyError:
            for variable in (from_lucode, to_lucode):
                if variable not in landcover_df.index:
                    raise ValueError(
                        'The landcover table is missing a row with the '
                        f'landuse code {variable}.')

        sparse_transition_table[(from_lucode, to_lucode)] = (
            transition_types[(from_is_cbc, to_is_cbc)])

    code_list = sorted(landcover_df.index)
    lulc_class_list_sorted = [
        landcover_df['lulc-class'][code] for code in code_list]
    with open(target_table_path, 'w') as csv_file:
        fieldnames = ['lulc-class'] + lulc_class_list_sorted
        csv_file.write(f"{','.join(fieldnames)}\n")
        for row_code in code_list:
            class_name = landcover_df['lulc-class'][row_code]
            row = [class_name]
            for col_code in code_list:
                try:
                    column_value = sparse_transition_table[
                        (row_code, col_code)]
                except KeyError:
                    # When there isn't a transition that we know about, just
                    # leave the table blank.
                    column_value = ''
                row.append(column_value)
            csv_file.write(','.join(row) + '\n')

        # Append legend
        csv_file.write("\n,legend")
        csv_file.write(
            "\n,empty cells indicate that no transitions occur of that type")
        csv_file.write("\n,disturb (disturbance): change to low- med- or "
                       "high-impact-disturb")
        csv_file.write("\n,accum (accumulation)")
        csv_file.write("\n,NCC (no-carbon-change)")


def _create_biophysical_table(landcover_df, target_biophysical_table_path):
    """Write the biophysical table template to disk.

    The biophysical table templates contains all of the fields required by the
    main Coastal Blue Carbon model, and any field values that exist in the
    landcover table provided to this model will be carried over to the new
    table.

    Args:
        landcover_df (pandas.DataFrame): A table mapping int landcover codes
            to biophysical data
        target_biophysical_table_path (string): The path to where the
            biophysical table template will be stored on disk.

    Returns:
        ``None``
    """
    target_column_names = [
        spec.id.lower() for spec in
        coastal_blue_carbon.MODEL_SPEC.get_input('biophysical_table_path').columns]

    with open(target_biophysical_table_path, 'w') as bio_table:
        bio_table.write(f"{','.join(target_column_names)}\n")
        for lulc_code, row in landcover_df.sort_index().iterrows():
            # 2 columns are defined below, and we need 1 less comma to only
            # have commas between fields.
            row = []
            for colname in target_column_names:
                if colname == 'lucode':
                    row.append(str(lulc_code))
                else:
                    try:
                        # Use the user's defined value if it exists
                        row.append(str(landcover_df[colname][lulc_code]))
                    except KeyError:
                        row.append('')
            bio_table.write(f"{','.join(row)}\n")


@validation.invest_validator
def validate(args, limit_to=None):
    """Validate an input dictionary for Coastal Blue Carbon: Preprocessor.

    Args:
        args (dict): The args dictionary.
        limit_to=None (str or None): If a string key, only this args parameter
            will be validated.  If ``None``, all args parameters will be
            validated.

    Returns:
        A list of tuples where tuple[0] is an iterable of keys that the error
        message applies to and tuple[1] is the string validation warning.
    """
    return validation.validate(args, MODEL_SPEC)
