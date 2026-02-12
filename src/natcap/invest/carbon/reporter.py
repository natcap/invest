import logging
import os
import time

import numpy
import pandas
from pint import Unit
import pygeoprocessing

from natcap.invest import __version__
from natcap.invest import gettext
from natcap.invest.reports import jinja_env, raster_utils, report_constants
from natcap.invest.spec import ModelSpec
from natcap.invest.unit_registry import u

LOGGER = logging.getLogger(__name__)

TEMPLATE = jinja_env.get_template('models/carbon.html')


def _get_raster_plot_tuples(args_dict: dict) -> tuple[
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[list[tuple[str, ...]]]]:
    input_raster_plot_tuples = [
        ('lulc_bas_path', 'nominal'),
    ]
    if args_dict['calc_sequestration']:
        input_raster_plot_tuples.extend([
            ('lulc_alt_path', 'nominal'),
        ])

    output_raster_plot_tuples = [
        ('c_storage_bas', 'continuous', 'linear'),
    ]
    if args_dict['calc_sequestration']:
        output_raster_plot_tuples.extend([
            ('c_storage_alt', 'continuous', 'linear'),
            ('c_change_bas_alt', 'divergent', 'linear'),
        ])
    if args_dict['do_valuation']:
        output_raster_plot_tuples.extend([
            ('npv_alt', 'divergent', 'linear'),
        ])

    if args_dict['calc_sequestration']:
        intermediate_output_raster_plot_tuples = [[
            (f'c_{pool_type}_bas', 'continuous', 'linear'),
            (f'c_{pool_type}_alt', 'continuous', 'linear')
         ] for pool_type in ['above', 'below', 'dead', 'soil']]
    else:
        intermediate_output_raster_plot_tuples = [[
            (f'c_{pool_type}_bas', 'continuous', 'linear')
            for pool_type in ['above', 'below', 'dead', 'soil']]]

    return (input_raster_plot_tuples,
            output_raster_plot_tuples,
            intermediate_output_raster_plot_tuples)


def _get_intermediate_output_headings(args_dict: dict) -> list[str]:
    """Get headings for Intermediate Outputs sections of the report.

    Args:
        args_dict (dict): the arguments passed to the model's ``execute``
            function.

    Returns:
        A list containing exactly one string or exactly four strings.
        If the model was run with ``calc_sequestration = False``, the report
        will group all four intermediate outputs into one section, with the
        heading "Carbon Maps by Pool Type".
        If the model was run with ``calc_sequestration = True``, the report
        will group intermediate outputs by carbon pool type, resulting in four
        sections containing two rasters each. This structure facilitates
        side-by-side comparisons of baseline vs. alternate scenarios for each
        carbon pool type.
    """
    if args_dict['calc_sequestration']:
        return [
            gettext('Carbon Maps: Aboveground'),
            gettext('Carbon Maps: Belowground'),
            gettext('Carbon Maps: Dead'),
            gettext('Carbon Maps: Soil'),
        ]
    else:
        return [gettext('Carbon Maps by Pool Type')]


def _get_table_inputs(args_dict: dict) -> list[tuple[str, str, Unit]]:
    table_inputs = [
        ('c_storage_bas', gettext('Baseline Carbon Storage'), u.metric_ton),
    ]
    if args_dict['calc_sequestration']:
        table_inputs.extend([
            ('c_storage_alt', gettext('Alternate Carbon Storage'),
             u.metric_ton),
            ('c_change_bas_alt', gettext('Change in Carbon Storage'),
             u.metric_ton),
        ])
    if args_dict['do_valuation']:
        table_inputs.extend([
            ('npv_alt', gettext('Net Present Value of Carbon Change'),
             u.currency),
        ])
    return table_inputs


def _generate_agg_results_table(args_dict: dict, file_registry: dict) -> str:
    table_inputs = _get_table_inputs(args_dict)

    table_df = pandas.DataFrame()

    total_col_name = gettext('Total')
    units_col_name = gettext('Units')
    filename_col_name = gettext('Filename')

    for (raster_id, description, units) in table_inputs:
        raster_path = file_registry[raster_id]
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        nodata = raster_info['nodata'][0]

        # Calculate sum.
        raster_sum = 0.0
        for _, block in pygeoprocessing.iterblocks((raster_path, 1)):
            raster_sum += numpy.sum(
                block[~pygeoprocessing.array_equals_nodata(
                        block, nodata)], dtype=numpy.float64)

        # Adjust for units.
        pixel_area = abs(numpy.prod(raster_info['pixel_size']))
        # Since each pixel value is in t/ha, ``total`` is in (t/ha * px) = t•px/ha.
        # Adjusted sum = ([total] t•px/ha) * ([pixel_area] m^2 / 1 px) * (1 ha / 10000 m^2) = t.
        summary_stat = raster_sum * pixel_area / 10000

        # Populate table row.
        table_df.loc[description, [total_col_name, units_col_name, filename_col_name]] = [
            summary_stat, units, os.path.basename(raster_path)]

    return table_df.to_html()


def report(file_registry: dict, args_dict: dict, model_spec: ModelSpec,
           target_html_filepath: str):
    """Generate an HTML summary of model results.

    Args:
        file_registry (dict): The ``natcap.invest.FileRegistry.registry``
            that was returned by the model's ``execute`` method.
        args_dict (dict): The arguments that were passed to the model's
            ``execute`` method.
        model_spec (natcap.invest.spec.ModelSpec): the model's ``MODEL_SPEC``.
        target_html_filepath (str): path to an HTML file to be generated by
            this function.

    Returns:
        ``None``
    """

    model_description = gettext(
        """
        The InVEST Carbon Storage and Sequestration model uses maps of land use
        along with stocks in four carbon pools (aboveground biomass,
        belowground biomass, soil, and dead organic matter) to estimate the
        amount of carbon stored in a landscape at baseline or the amount of
        carbon sequestered over time. Optionally, the market or social value of
        sequestered carbon, its annual rate of change, and a discount rate can
        be used to estimate the value of this ecosystem service to society.
        """)

    (input_raster_tuples,
     output_raster_tuples,
     intermediate_raster_tuples) = _get_raster_plot_tuples(args_dict)

    input_raster_plot_configs = raster_utils.build_raster_plot_configs(
        args_dict, input_raster_tuples)
    inputs_img_src = raster_utils.plot_and_base64_encode_rasters(
        input_raster_plot_configs)
    input_raster_caption = raster_utils.generate_caption_from_raster_list(
        [(id, 'input') for (id, _) in input_raster_tuples],
        args_dict, file_registry, model_spec)

    output_raster_plot_configs = raster_utils.build_raster_plot_configs(
            file_registry, output_raster_tuples)
    outputs_img_src = raster_utils.plot_and_base64_encode_rasters(
        output_raster_plot_configs)
    output_raster_caption = raster_utils.generate_caption_from_raster_list(
        [(id, 'output') for (id, _, _) in output_raster_tuples],
        args_dict, file_registry, model_spec)

    intermediate_raster_plot_configs = [raster_utils.build_raster_plot_configs(
            file_registry, tuples) for tuples in intermediate_raster_tuples]
    intermediate_img_srcs = [raster_utils.plot_and_base64_encode_rasters(
        configs) for configs in intermediate_raster_plot_configs]
    intermediate_raster_captions = [raster_utils.generate_caption_from_raster_list(
        [(id, 'output') for (id, _, _) in tuples],
        args_dict, file_registry, model_spec) for tuples in intermediate_raster_tuples]

    intermediate_headings = _get_intermediate_output_headings(args_dict)

    intermediate_raster_sections = [
        {'heading': heading, 'img_src': img_src, 'caption': caption}
        for (heading, img_src, caption)
        in zip(intermediate_headings,
               intermediate_img_srcs,
               intermediate_raster_captions)
    ]

    input_raster_stats_table = raster_utils.raster_inputs_summary(
        args_dict).to_html(na_rep='')

    output_raster_stats_table = raster_utils.raster_workspace_summary(
        file_registry).to_html(na_rep='')

    agg_results_table = _generate_agg_results_table(args_dict, file_registry)

    lulc_pre_caption = gettext(
        'Values in the legend are listed in order of frequency (most common '
        'first).')

    with open(target_html_filepath, 'w', encoding='utf-8') as target_file:
        target_file.write(TEMPLATE.render(
            report_script=model_spec.reporter,
            invest_version=__version__,
            report_filepath=target_html_filepath,
            model_id=model_spec.model_id,
            model_name=model_spec.model_title,
            model_description=model_description,
            userguide_page=model_spec.userguide,
            timestamp=time.strftime('%Y-%m-%d %H:%M'),
            args_dict=args_dict,
            agg_results_table=agg_results_table,
            inputs_img_src=inputs_img_src,
            inputs_caption=input_raster_caption,
            lulc_pre_caption=lulc_pre_caption,
            outputs_img_src=outputs_img_src,
            outputs_caption=output_raster_caption,
            intermediate_raster_sections=intermediate_raster_sections,
            raster_group_caption=report_constants.RASTER_GROUP_CAPTION,
            output_raster_stats_table=output_raster_stats_table,
            input_raster_stats_table=input_raster_stats_table,
            stats_table_note=report_constants.STATS_TABLE_NOTE,
            model_spec_outputs=model_spec.outputs,
        ))

    LOGGER.info(f'Created {target_html_filepath}')
