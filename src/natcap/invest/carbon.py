# coding=UTF-8
"""Carbon Storage and Sequestration."""
import codecs
import logging
import os
import time

from osgeo import gdal
import numpy
import pygeoprocessing
import taskgraph

from . import validation
from . import utils
from . import spec
from .unit_registry import u
from . import gettext

LOGGER = logging.getLogger(__name__)

MODEL_SPEC = spec.ModelSpec(
    model_id="carbon",
    model_title=gettext("Carbon Storage and Sequestration"),
    userguide="carbonstorage.html",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    aliases=(),
    module_name=__name__,
    input_field_order=[
        ["workspace_dir", "results_suffix"],
        ["lulc_bas_path", "carbon_pools_path"],
        ["calc_sequestration", "lulc_alt_path"],
        ["do_valuation", "lulc_bas_year", "lulc_alt_year", "price_per_metric_ton_of_c",
         "discount_rate", "rate_change"]
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.SingleBandRasterInput(
            id="lulc_bas_path",
            name=gettext("baseline LULC"),
            about=gettext(
                "A map of LULC for the baseline scenario, which must occur prior to the"
                " alternate scenario. All values in this raster must have corresponding"
                " entries in the Carbon Pools table."
            ),
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.BooleanInput(
            id="calc_sequestration",
            name=gettext("calculate sequestration"),
            about=gettext(
                "Run sequestration analysis. This requires inputs of LULC maps for both"
                " baseline and alternate scenarios. Required if run valuation model is"
                " selected."
            ),
            required="do_valuation"
        ),
        spec.SingleBandRasterInput(
            id="lulc_alt_path",
            name=gettext("alternate LULC"),
            about=gettext(
                "A map of LULC for the alternate scenario, which must occur after the"
                " baseline scenario. All values in this raster must have corresponding"
                " entries in the Carbon Pools table. This raster must align with the"
                " Baseline LULC raster. Required if Calculate Sequestration is selected."
            ),
            required="calc_sequestration",
            allowed="calc_sequestration",
            data_type=int,
            units=None,
            projected=True,
            projection_units=u.meter
        ),
        spec.CSVInput(
            id="carbon_pools_path",
            name=gettext("carbon pools"),
            about=gettext(
                "A table that maps each LULC code to carbon pool data for that LULC type."
            ),
            columns=[
                spec.LULC_TABLE_COLUMN,
                spec.NumberInput(
                    id="c_above",
                    about=gettext("Carbon density of aboveground biomass."),
                    units=u.metric_ton / u.hectare
                ),
                spec.NumberInput(
                    id="c_below",
                    about=gettext("Carbon density of belowground biomass."),
                    units=u.metric_ton / u.hectare
                ),
                spec.NumberInput(
                    id="c_soil",
                    about=gettext("Carbon density of soil."),
                    units=u.metric_ton / u.hectare
                ),
                spec.NumberInput(
                    id="c_dead",
                    about=gettext("Carbon density of dead matter."),
                    units=u.metric_ton / u.hectare
                )
            ],
            index_col="lucode"
        ),
        spec.NumberInput(
            id="lulc_bas_year",
            name=gettext("baseline LULC year"),
            about=gettext(
                "The calendar year of the baseline scenario depicted in the baseline LULC"
                " map. Must be < alternate LULC year. Required if Run Valuation model is"
                " selected."
            ),
            required="do_valuation",
            allowed="do_valuation",
            units=u.year_AD,
            expression="float(value).is_integer()"
        ),
        spec.NumberInput(
            id="lulc_alt_year",
            name=gettext("alternate LULC year"),
            about=gettext(
                "The calendar year of the alternate scenario depicted in the alternate"
                " LULC map. Must be > baseline LULC year. Required if Run Valuation model"
                " is selected."
            ),
            required="do_valuation",
            allowed="do_valuation",
            units=u.year_AD,
            expression="float(value).is_integer()"
        ),
        spec.BooleanInput(
            id="do_valuation",
            name=gettext("run valuation model"),
            about=gettext(
                "Calculate net present value for the alternate scenario and report it in"
                " the final HTML document."
            ),
            required=False,
            allowed="calc_sequestration"
        ),
        spec.NumberInput(
            id="price_per_metric_ton_of_c",
            name=gettext("price of carbon"),
            about=gettext(
                "The present value of carbon. Required if Run Valuation model is"
                " selected."
            ),
            required="do_valuation",
            allowed="do_valuation",
            units=u.currency / u.metric_ton
        ),
        spec.PercentInput(
            id="discount_rate",
            name=gettext("annual market discount rate"),
            about=gettext(
                "The annual market discount rate in the price of carbon, which reflects"
                " society's preference for immediate benefits over future benefits."
                " Required if Run Valuation model is selected. This assumes that the"
                " baseline scenario is current and the alternate scenario is in the"
                " future."
            ),
            required="do_valuation",
            allowed="do_valuation",
            units=None
        ),
        spec.PercentInput(
            id="rate_change",
            name=gettext("annual price change"),
            about=gettext(
                "The relative annual change of the price of carbon. Required if Run"
                " Valuation model is selected."
            ),
            required="do_valuation",
            allowed="do_valuation",
            units=None
        )
    ],
    outputs=[
        spec.FileOutput(
            id="html_report",
            path="report.html",
            about=gettext(
                "This file presents a summary of all data computed by the model. It also"
                " includes descriptions of all other output files produced by the model,"
                " so it is a good place to begin exploring and understanding model"
                " results. Because this is an HTML file, it can be opened with any web"
                " browser."
            )
        ),
        spec.SingleBandRasterOutput(
            id="c_storage_bas",
            path="c_storage_bas.tif",
            about=gettext(
                "Raster showing the amount of carbon stored in each pixel for the"
                " baseline scenario. It is a sum of all of the carbon pools provided by"
                " the biophysical table."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_storage_alt",
            path="c_storage_alt.tif",
            about=gettext(
                "Raster showing the amount of carbon stored in each pixel for the"
                " alternate scenario. It is a sum of all of the carbon pools provided by"
                " the biophysical table."
            ),
            created_if="lulc_alt_path",
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_change_bas_alt",
            path="c_change_bas_alt.tif",
            about=gettext(
                "Raster showing the difference in carbon stored between the alternate"
                " landscape and the baseline landscape. In this map some values may be"
                " negative and some positive. Positive values indicate sequestered"
                " carbon, negative values indicate carbon that was lost."
            ),
            created_if="lulc_alt_path",
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="npv_alt",
            path="npv_alt.tif",
            about=gettext(
                "Rasters showing the economic value of carbon sequestered between the"
                " baseline and the alternate landscape dates."
            ),
            created_if="lulc_alt_path",
            data_type=float,
            units=u.currency / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_above_bas",
            path="intermediate_outputs/c_above_bas.tif",
            about=gettext(
                "Raster of aboveground carbon values in the baseline scenario,"
                " mapped from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_above_alt",
            path="intermediate_outputs/c_above_alt.tif",
            about=gettext(
                "Raster of aboveground carbon values in the alternate scenario,"
                " mapped from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_below_bas",
            path="intermediate_outputs/c_below_bas.tif",
            about=gettext(
                "Raster of belowground carbon values in the baseline scenario,"
                " mapped from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_below_alt",
            path="intermediate_outputs/c_below_alt.tif",
            about=gettext(
                "Raster of belowground carbon values in the alternate scenario,"
                " mapped from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_soil_bas",
            path="intermediate_outputs/c_soil_bas.tif",
            about=gettext(
                "Raster of soil carbon values in the baseline scenario, mapped"
                " from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_soil_alt",
            path="intermediate_outputs/c_soil_alt.tif",
            about=gettext(
                "Raster of soil carbon values in the alternate scenario, mapped"
                " from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_dead_bas",
            path="intermediate_outputs/c_dead_bas.tif",
            about=gettext(
                "Raster of dead matter carbon values in the baseline scenario,"
                " mapped from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.SingleBandRasterOutput(
            id="c_dead_alt",
            path="intermediate_outputs/c_dead_alt.tif",
            about=gettext(
                "Raster of dead matter carbon values in the alternate scenario,"
                " mapped from the Carbon Pools table to the LULC."
            ),
            data_type=float,
            units=u.metric_ton / u.hectare
        ),
        spec.TASKGRAPH_CACHE
    ]
)

# -1.0 since carbon stocks are 0 or greater
_CARBON_NODATA = -1.0


def execute(args):
    """Carbon.

    Calculate the amount of carbon stocks given a landscape, or the difference
    due to some change, and calculate economic valuation on those scenarios.

    The model can operate on a single scenario or a combined baseline and
    alternate scenario.

    Args:
        args['workspace_dir'] (string): a path to the directory that will
            write output and other temporary files during calculation.
        args['results_suffix'] (string): appended to any output file name.
        args['lulc_bas_path'] (string): a path to a raster representing the
            baseline carbon stocks.
        args['calc_sequestration'] (bool): if true, sequestration should
            be calculated and 'lulc_alt_path' should be defined.
        args['lulc_alt_path'] (string): a path to a raster representing alternate
            landcover scenario.  Optional, but if present and well defined
            will trigger a sequestration calculation.
        args['carbon_pools_path'] (string): path to CSV or that indexes carbon
            storage density to lulc codes. (required if 'do_uncertainty' is
            false)
        args['lulc_bas_year'] (int/string): an integer representing the year
            of `args['lulc_bas_path']` used if `args['do_valuation']`
            is True.
        args['lulc_alt_year'](int/string): an integer representing the year
            of `args['lulc_alt_path']` used in valuation if it exists.
            Required if  `args['do_valuation']` is True and
            `args['lulc_alt_path']` is present and well defined.
        args['do_valuation'] (bool): if true then run the valuation model on
            available outputs. Calculate NPV for an alternate scenario and
            report in final HTML document.
        args['price_per_metric_ton_of_c'] (float): Is the present value of
            carbon per metric ton. Used if `args['do_valuation']` is present
            and True.
        args['discount_rate'] (float): Discount rate used if NPV calculations
            are required.  Used if `args['do_valuation']` is  present and
            True.
        args['rate_change'] (float): Annual rate of change in price of carbon
            as a percentage.  Used if `args['do_valuation']` is  present and
            True.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    args, file_registry, graph = MODEL_SPEC.setup(args)

    if (args['do_valuation'] and
            args['lulc_bas_year'] >= args['lulc_alt_year']):
        raise ValueError(
            "Invalid input for lulc_bas_year or lulc_alt_year. The Alternate "
            f"LULC Year ({args['lulc_alt_year']}) must be greater "
            f"than the Baseline LULC Year ({args['lulc_bas_year']}). "
            "Ensure that the Baseline LULC Year is earlier than the Alternate LULC Year."
        )

    cell_size_set = set()
    raster_size_set = set()
    valid_lulc_keys = []
    valid_scenarios = []
    tifs_to_summarize = set()  # passed to _generate_report()

    for scenario_type in ['bas', 'alt']:
        lulc_key = "lulc_%s_path" % (scenario_type)
        if args[lulc_key]:
            raster_info = pygeoprocessing.get_raster_info(args[lulc_key])
            cell_size_set.add(raster_info['pixel_size'])
            raster_size_set.add(raster_info['raster_size'])
            valid_lulc_keys.append(lulc_key)
            valid_scenarios.append(scenario_type)
    if len(cell_size_set) > 1:
        raise ValueError(
            "the pixel sizes of %s are not equivalent. Here are the "
            "different sets that were found in processing: %s" % (
                valid_lulc_keys, cell_size_set))
    if len(raster_size_set) > 1:
        raise ValueError(
            "the raster dimensions of %s are not equivalent. Here are the "
            "different sizes that were found in processing: %s" % (
                valid_lulc_keys, raster_size_set))

    # calculate total carbon storage
    LOGGER.info('Map all carbon pools to carbon storage rasters.')
    carbon_map_task_lookup = {}
    sum_rasters_task_lookup = {}
    carbon_pool_df = MODEL_SPEC.get_input(
        'carbon_pools_path').get_validated_dataframe(
        args['carbon_pools_path'])
    for scenario_type in valid_scenarios:
        carbon_map_task_lookup[scenario_type] = []
        storage_path_list = []
        for pool_type in ['c_above', 'c_below', 'c_soil', 'c_dead']:
            carbon_pool_by_type = carbon_pool_df[pool_type].to_dict()

            lulc_key = f'lulc_{scenario_type}_path'
            storage_key = f'{pool_type}_{scenario_type}'
            LOGGER.info(
                f"Mapping carbon from '{lulc_key}' to '{storage_key}' scenario.")

            carbon_map_task = graph.add_task(
                _generate_carbon_map,
                args=(args[lulc_key], carbon_pool_by_type,
                      file_registry[storage_key]),
                target_path_list=[file_registry[storage_key]],
                task_name=f'carbon_map_{storage_key}')
            storage_path_list.append(file_registry[storage_key])
            carbon_map_task_lookup[scenario_type].append(carbon_map_task)

        output_key = 'c_storage_' + scenario_type
        LOGGER.info(
            "Calculate carbon storage for '%s'", output_key)

        sum_rasters_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=sum_op,
                rasters=storage_path_list,
                target_path=file_registry[output_key],
                target_nodata=_CARBON_NODATA),
            target_path_list=[file_registry[output_key]],
            dependent_task_list=carbon_map_task_lookup[scenario_type],
            task_name='sum_rasters_for_total_c_%s' % output_key)
        sum_rasters_task_lookup[scenario_type] = sum_rasters_task
        tifs_to_summarize.add(file_registry[output_key])

    # calculate sequestration
    diff_rasters_task_lookup = {}
    if 'alt' in valid_scenarios:
        output_key = 'c_change_bas_alt'
        LOGGER.info("Calculate sequestration scenario '%s'", output_key)

        diff_rasters_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs=dict(
                op=numpy.subtract,  # c_change = scenario C - baseline C
                rasters=[file_registry['c_storage_alt'],
                         file_registry['c_storage_bas']],
                target_path=file_registry[output_key],
                target_nodata=_CARBON_NODATA),
            target_path_list=[file_registry[output_key]],
            dependent_task_list=[
                sum_rasters_task_lookup['bas'],
                sum_rasters_task_lookup['alt']],
            task_name='diff_rasters_for_%s' % output_key)
        diff_rasters_task_lookup['alt'] = diff_rasters_task
        tifs_to_summarize.add(file_registry[output_key])

    # calculate net present value
    calculate_npv_tasks = []
    if args['do_valuation']:
        LOGGER.info('Constructing valuation formula.')
        valuation_constant = _calculate_valuation_constant(
            args['lulc_bas_year'],
            args['lulc_alt_year'],
            args['discount_rate'],
            args['rate_change'],
            args['price_per_metric_ton_of_c'])

        if 'alt' in valid_scenarios:
            output_key = 'npv_alt'
            LOGGER.info("Calculating NPV for scenario 'alt'")

            calculate_npv_task = graph.add_task(
                _calculate_npv,
                args=(file_registry['c_change_bas_alt'],
                      valuation_constant, file_registry[output_key]),
                target_path_list=[file_registry[output_key]],
                dependent_task_list=[diff_rasters_task_lookup['alt']],
                task_name='calculate_%s' % output_key)
            calculate_npv_tasks.append(calculate_npv_task)
            tifs_to_summarize.add(file_registry[output_key])

    # Report aggregate results
    tasks_to_report = (list(sum_rasters_task_lookup.values())
                       + list(diff_rasters_task_lookup.values())
                       + calculate_npv_tasks)
    _ = graph.add_task(
        _generate_report,
        args=(tifs_to_summarize, args, file_registry),
        target_path_list=[file_registry['html_report']],
        dependent_task_list=tasks_to_report,
        task_name='generate_report')
    graph.join()
    return file_registry.registry


# element-wise sum function to pass to raster_map
def sum_op(*xs): return numpy.sum(xs, axis=0)


def _accumulate_totals(raster_path):
    """Sum all non-nodata pixels in `raster_path` and return result."""
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    raster_sum = 0.0
    for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
        # The float64 dtype in the sum is needed to reduce numerical error in
        # the sum.  Users calculated the sum with ArcGIS zonal statistics,
        # noticed a difference and wrote to us about it on the forum.
        raster_sum += numpy.sum(
            block[~pygeoprocessing.array_equals_nodata(
                    block, nodata)], dtype=numpy.float64)
    return raster_sum


def _generate_carbon_map(
        lulc_path, carbon_pool_by_type, out_carbon_stock_path):
    """Generate carbon stock raster by mapping LULC values to carbon pools.

    Args:
        lulc_path (string): landcover raster with integer pixels.
        out_carbon_stock_path (string): path to output raster that will have
            pixels with carbon storage values in them with units of Mg/ha.
        carbon_pool_by_type (dict): a dictionary that maps landcover values
            to carbon storage densities per area (Mg C/Ha).

    Returns:
        None.
    """
    carbon_stock_by_type = dict([
        (lulcid, stock)
        for lulcid, stock in carbon_pool_by_type.items()])

    reclass_error_details = {
        'raster_name': 'LULC', 'column_name': 'lucode',
        'table_name': 'Carbon Pools'}
    utils.reclassify_raster(
        (lulc_path, 1), carbon_stock_by_type, out_carbon_stock_path,
        gdal.GDT_Float32, _CARBON_NODATA, reclass_error_details)


def _calculate_valuation_constant(
        lulc_bas_year, lulc_alt_year, discount_rate, rate_change,
        price_per_metric_ton_of_c):
    """Calculate a net present valuation constant to multiply carbon storage.

    Args:
        lulc_bas_year (int): calendar year for baseline
        lulc_alt_year (int): calendar year for alternate
        discount_rate (float): annual discount rate as a percentage
        rate_change (float): annual change in price of carbon as a percentage
        price_per_metric_ton_of_c (float): currency amount of Mg of carbon

    Returns:
        a floating point number that can be used to multiply a carbon
        storage change value by to calculate NPV.
    """
    n_years = lulc_alt_year - lulc_bas_year
    ratio = (
        1 / ((1 + discount_rate / 100) *
             (1 + rate_change / 100)))
    valuation_constant = (price_per_metric_ton_of_c / n_years)
    # note: the valuation formula in the user's guide uses sum notation.
    # here it's been simplified to remove the sum using the general rule
    # sum(r^k) from k=0 to N  ==  (r^(N+1) - 1) / (r - 1)
    # where N = n_years-1 and r = ratio
    if ratio == 1:
        # if ratio == 1, we would divide by zero in the equation below
        # so use the limit as ratio goes to 1, which is n_years
        valuation_constant *= n_years
    else:
        valuation_constant *= (1 - ratio ** n_years) / (1 - ratio)
    return valuation_constant


def _calculate_npv(c_change_carbon_path, valuation_constant, npv_out_path):
    """Calculate net present value.

    Args:
        c_change_carbon_path (string): path to change in carbon storage over
            time.
        valuation_constant (float): value to multiply each carbon storage
            value by to calculate NPV.
        npv_out_path (string): path to output net present value raster.

    Returns:
        None.
    """
    pygeoprocessing.raster_map(
        op=lambda carbon: carbon * valuation_constant,
        rasters=[c_change_carbon_path],
        target_path=npv_out_path)


def _generate_report(raster_file_set, model_args, file_registry):
    """Generate a human readable HTML report of summary stats of model run.

    Args:
        raster_file_set (set): paths to rasters that need summary stats.
        model_args (dict): InVEST argument dictionary.
        file_registry (dict): file path dictionary for InVEST workspace.

    Returns:
        None.
    """
    with codecs.open(file_registry['html_report'], 'w', encoding='utf-8') as report_doc:
        # Boilerplate header that defines style and intro header.
        header = (
            """
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="utf-8">
            <title>Carbon Results</title>
            <style type="text/css">
                body {
                    --invest-green: #148f68;
                    background: #ffffff;
                    color: #000000;
                    font-family: Roboto, "Helvetica Neue", Arial, sans-serif;
                }
                h1, h2, th {
                    font-weight: bold;
                }
                h1, h2 {
                    color: var(--invest-green);
                }
                h1 {
                    font-size: 2rem;
                }
                h2 {
                    font-size: 1.5rem;
                }
                table {
                    border: 0.25rem solid var(--invest-green);
                    border-collapse: collapse;
                }
                thead tr {
                    background: #e9ecef;
                    border-bottom: 0.1875rem solid var(--invest-green);
                }
                tbody tr:nth-child(even) {
                    background: ghostwhite;
                }
                th {
                    padding: 0.5rem;
                    text-align:left;
                }
                td {
                    padding: 0.375rem 0.5rem;
                }
                .number {
                    text-align: right;
                    font-family: monospace;
                }
            </style>
            </head>
            <body>
            <h1>InVEST Carbon Model Results</h1>
            <p>This document summarizes the results from
            running the InVEST carbon model with the following data.</p>
            """
        )

        report_doc.write(header)
        report_doc.write('<p>Report generated at %s</p>' % (
            time.strftime("%Y-%m-%d %H:%M")))

        # Report input arguments
        report_doc.write('<h2>Inputs</h2>')
        report_doc.write('<table><thead><tr><th>arg id</th><th>arg value</th>'
                         '</tr></thead><tbody>')
        for key, value in model_args.items():
            report_doc.write('<tr><td>%s</td><td>%s</td></tr>' % (key, value))
        report_doc.write('</tbody></table>')

        # Report aggregate results
        report_doc.write('<h2>Aggregate Results</h2>')
        report_doc.write(
            '<table><thead><tr><th>Description</th><th>Value</th><th>Units'
            '</th><th>Raw File</th></tr></thead><tbody>')

        carbon_units = 'metric tons'

        # value lists are [sort priority, description, statistic, units]
        report = [
            (file_registry['c_storage_bas'], 'Baseline Carbon Storage',
             carbon_units),
            (file_registry['c_storage_alt'], 'Alternate Carbon Storage',
             carbon_units),
            (file_registry['c_change_bas_alt'], 'Change in Carbon Storage',
             carbon_units),
            (file_registry['npv_alt'],
             'Net Present Value of Carbon Change', 'currency units'),
        ]

        for raster_uri, description, units in report:
            if raster_uri in raster_file_set:
                total = _accumulate_totals(raster_uri)
                raster_info = pygeoprocessing.get_raster_info(raster_uri)
                pixel_area = abs(numpy.prod(raster_info['pixel_size']))
                # Since each pixel value is in Mg/ha, ``total`` is in (Mg/ha * px) = Mg•px/ha.
                # Adjusted sum = ([total] Mg•px/ha) * ([pixel_area] m^2 / 1 px) * (1 ha / 10000 m^2) = Mg.
                summary_stat = total * pixel_area / 10000
                report_doc.write(
                    '<tr><td>%s</td><td class="number" data-summary-stat="%s">'
                    '%.2f</td><td>%s</td><td>%s</td></tr>' % (
                        description, description, summary_stat, units,
                        raster_uri))
        report_doc.write('</tbody></table></body></html>')


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
    return validation.validate(args, MODEL_SPEC)
