import logging
import time

import altair
import geopandas
import pandas

from natcap.invest import __version__
from natcap.invest import gettext
from natcap.invest.reports import jinja_env, raster_utils, report_constants, \
    vector_utils
from natcap.invest.spec import ModelSpec, FileRegistry

from natcap.invest.reports.raster_utils import RasterDatatype, \
    RasterPlotConfig, RasterTransform

LOGGER = logging.getLogger(__name__)

TEMPLATE = jinja_env.get_template('models/urban_mental_health.html')

MAP_WIDTH = 450  # pixels


def _get_conditional_raster_plot_tuples(model_spec: ModelSpec,
                                        args_dict: dict,
                                        file_registry: FileRegistry) -> tuple[
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[list[tuple[str, ...]]]]:

    """
    inputs
        - population raster
        - baseline prevalence vector

    if NDVI:
        inputs:
            - ndvi_bas_path
            - ndvi_alt_path
            # - if lulc_attr_csv: show lulc_attr_csv and lulc_base
            #     - if lulc_alt: show lulc_alt as well

    if LULC:
        inputs:
            - lulc_base
            - lulc_alt
            - if not lulc_attr_csv: show lulc_attr_table that was built
            - if ndvi_base: show it as well

        intermediates:
            - LULC baseline reclassified to NDVI
            - if lulc_alt: LULC alternate reclassified to NDVI


    intermediates:
        - ndvi_delta
        - baseline cases
        - baseline prevalence

    outputs:
        - preventable_cases
        - preventable_cost
        - preventable cases cost sum table (just total cases and total cost)
        - vector symbolized with field 'sum_cost'
        - vector symbolized with field 'sum_cases'


    """
    # if args_dict['scenario'] == 'ndvi':
    #     delta_ndvi_colorramp = matplotlib.colors.LinearSegmentedColormap.from_list(
    #         'truncated_PuOr', matplotlib.cm.PuOr(numpy.linspace(0.20, 0.80, 256)))
    # else:
    #     delta_ndvi_colorramp = None
    # LOGGER.info("custom color: %s", delta_ndvi_colorramp)

    input_raster_config_list = []
    intermediate_raster_config_lists = [[
        RasterPlotConfig(
            raster_path=file_registry['ndvi_base_buffer_mean_clipped'],
            datatype=RasterDatatype.divergent,
            spec=model_spec.get_output('ndvi_base_buffer_mean_clipped')
        ),
        RasterPlotConfig(
            raster_path=file_registry['ndvi_alt_buffer_mean_clipped'],
            datatype=RasterDatatype.divergent,
            spec=model_spec.get_output('ndvi_alt_buffer_mean_clipped')
        ),
        RasterPlotConfig(
            raster_path=file_registry['delta_ndvi'], #TODO: if scenario is NDVI, contrain the colorramp by another 10%
            datatype=RasterDatatype.divergent,
            spec=model_spec.get_output('delta_ndvi')
            # custom_colormap=delta_ndvi_colorramp
        )],
        [RasterPlotConfig(
            raster_path=file_registry['baseline_cases'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('baseline_cases')
        ),
        RasterPlotConfig(
            raster_path=file_registry['baseline_prevalence_raster'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('baseline_prevalence_raster')
        )
    ]]

    if args_dict['lulc_base']:
        input_raster_config_list.append(
            RasterPlotConfig(
                raster_path=args_dict['lulc_base'],
                datatype=RasterDatatype.nominal,
                spec=model_spec.get_input('lulc_base')
            )
        )
    if args_dict['lulc_alt']:
        input_raster_config_list.append(
            RasterPlotConfig(
                raster_path=args_dict['lulc_alt'],
                datatype=RasterDatatype.nominal,
                spec=model_spec.get_input('lulc_alt')
            )
        )

    if args_dict["scenario"] == 'ndvi':
        input_raster_config_list.insert(0,
            RasterPlotConfig(
                raster_path=args_dict['ndvi_base'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_input('ndvi_base')
            )
        )
        input_raster_config_list.insert(1,
            RasterPlotConfig(
                raster_path=args_dict['ndvi_alt'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_input('ndvi_alt')
            )
        )

    else:
        intermediate_raster_config_lists.insert(0, [
            RasterPlotConfig(
                raster_path=file_registry['ndvi_base_aligned_masked'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('ndvi_base_aligned_masked')
            ),
            # reclassified baseline lulc to ndvi values based on means
            RasterPlotConfig(
                raster_path=file_registry['ndvi_alt_aligned_masked'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('ndvi_alt_aligned_masked')
            )
        ])
    input_raster_config_list.append(
        RasterPlotConfig(
            raster_path=args_dict['population_raster'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_input('population_raster')
        )
    )

    output_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['preventable_cases'],
            datatype=RasterDatatype.divergent, #TODO - should this be continuous?
            spec=model_spec.get_output('preventable_cases'),
            transform=RasterTransform.log
        )
        ]
    if args_dict['health_cost_rate']:
        output_raster_config_list.append(
            RasterPlotConfig(
                raster_path=file_registry['preventable_cost'],
                datatype=RasterDatatype.divergent,
                spec=model_spec.get_output('preventable_cost'),
                transform=RasterTransform.log
            )
        )

    return (input_raster_config_list,
            intermediate_raster_config_lists,
            output_raster_config_list)


def _get_intermediate_output_headings(args_dict: dict) -> list[str]:
    """Get headings for Intermediate Outputs sections of the report.

    Args:
        args_dict (dict): the arguments passed to the model's ``execute``
            function.

    Returns:
        A list containing exactly two strings or exactly three strings.
        If the model was run with ``scenario==ndvi``, the report will display
        delta ndvi, baseline cases, and baseline prevalence as three separate
        sections with the headings "Change in NDVI", "Baseline Cases", and "Baseline Prevalence". 
        If the model was run with ``scenario==lulc``, the report will also show the reclassified
        LULC-to-NDVI rasters as intermediate outputs.
    """
    intermediate_captions = [
        gettext('Difference in NDVI between Baseline and Alternate'),
        gettext('Baseline Cases & Prevalence')
    ]
    if args_dict['scenario'] == 'lulc':
        intermediate_captions.insert(0,
            gettext('Reclassified Baseline & Alternate LULC (to NDVI)'))

    return intermediate_captions


def _generate_agg_results_table(file_registry: dict) -> str:
    table_path = file_registry['preventable_cases_cost_sum_table']
    full_table_df = pandas.read_csv(table_path)
    total_cases = list(full_table_df['total_cases'])[-1]
    table_df = pandas.DataFrame({'Total Preventable Cases': [total_cases]})
    if file_registry.get('preventable_cost'):
        total_cost = list(full_table_df['total_cost'])[-1]
        table_df['Total Preventable Cost'] = [total_cost]

    return table_df.to_html(index=False)


def _create_aggregate_map(
        geodataframe,
        extent_feature,
        xy_ratio,
        attribute,
        title):
    """Create a choropleth map for a given attribute and return Vega JSON."""

    chart = altair.Chart(geodataframe).mark_geoshape(
        clip=True,
        stroke="white",
        strokeWidth=0.5
    ).project(
        type='identity',
        reflectY=True,
        fit=extent_feature
    ).encode(
        color=altair.Color(
            f'{attribute}:Q',
            scale=altair.Scale(scheme='purpleorange', reverse=True),
            legend=altair.Legend(title=attribute)
        ),
        tooltip=[
            altair.Tooltip(f'{attribute}:Q', title=attribute, format=',.2f')
        ]
    ).properties(
        width=MAP_WIDTH,
        height=MAP_WIDTH / xy_ratio,
        title=title
    ).configure_legend(**vector_utils.LEGEND_CONFIG)

    return chart.to_json()


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

    input_raster_config_list, \
        intermediate_raster_config_lists, \
            output_raster_config_list = _get_conditional_raster_plot_tuples(model_spec, args_dict, file_registry)

    inputs_img_src = raster_utils.plot_and_base64_encode_rasters(
        input_raster_config_list)
    input_raster_caption = raster_utils.caption_raster_list(
        input_raster_config_list)

    outputs_img_src = raster_utils.plot_and_base64_encode_rasters(
        output_raster_config_list)
    output_raster_caption = raster_utils.caption_raster_list(
        output_raster_config_list)

    # There can be multiple sections for intermediate rasters
    intermediate_img_srcs = [raster_utils.plot_and_base64_encode_rasters(
        config_list) for config_list in intermediate_raster_config_lists]
    intermediate_raster_captions = [raster_utils.caption_raster_list(
        config_list) for config_list in intermediate_raster_config_lists]

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

    agg_results_table = _generate_agg_results_table(file_registry)

    # Vector maps
    aggregate_vector_path = file_registry['preventable_cases_cost_sum_vector']
    aggregate_gdf = geopandas.read_file(aggregate_vector_path)
    extent_feature, xy_ratio = vector_utils.get_geojson_bbox(aggregate_gdf)

    cases_map_json = _create_aggregate_map(
        aggregate_gdf,
        extent_feature,
        xy_ratio,
        'sum_cases',
        gettext('Preventable Cases by AOI Feature')
    )
    cases_map_caption = [
        model_spec.get_output('preventable_cases_cost_sum_vector')
        .get_field('sum_cases').about
    ]

    cost_map_json = None
    cost_map_caption = None

    if 'sum_cost' in aggregate_gdf.columns:
        nonnull_cost = aggregate_gdf['sum_cost'].notna().any()
        nonzero_cost = (aggregate_gdf['sum_cost'].fillna(0) != 0).any()
        if nonnull_cost and nonzero_cost:
            cost_map_json = _create_aggregate_map(
                aggregate_gdf,
                extent_feature,
                xy_ratio,
                'sum_cost',
                gettext('Preventable Cost by AOI Feature')
            )
            cost_map_caption = [
                model_spec.get_output('preventable_cases_cost_sum_vector')
                .get_field('sum_cost').about
            ]

    aggregate_map_source_list = [
        model_spec.get_output('preventable_cases_cost_sum_vector').path
    ]

    with open(target_html_filepath, 'w', encoding='utf-8') as target_file:
        target_file.write(TEMPLATE.render(
            report_script=model_spec.reporter,
            invest_version=__version__,
            report_filepath=target_html_filepath,
            model_id=model_spec.model_id,
            model_name=model_spec.model_title,
            model_description=model_spec.about,
            userguide_page=model_spec.userguide,
            timestamp=time.strftime('%Y-%m-%d %H:%M'),
            args_dict=args_dict,
            agg_results_table=agg_results_table,
            inputs_img_src=inputs_img_src,
            inputs_caption=input_raster_caption,
            outputs_img_src=outputs_img_src,
            outputs_caption=output_raster_caption,
            intermediate_raster_sections=intermediate_raster_sections,
            raster_group_caption=report_constants.RASTER_GROUP_CAPTION,
            output_raster_stats_table=output_raster_stats_table,
            input_raster_stats_table=input_raster_stats_table,
            stats_table_note=report_constants.STATS_TABLE_NOTE,
            cases_map_json=cases_map_json,
            cost_map_json=cost_map_json,
            cases_map_caption=cases_map_caption,
            cost_map_caption=cost_map_caption,
            aggregate_map_source_list=aggregate_map_source_list,
            model_spec_outputs=model_spec.outputs,
        ))

    LOGGER.info(f'Created {target_html_filepath}')
