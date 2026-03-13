import csv
import logging
import os
import time

import altair
import geopandas
import numpy
import pandas
import pygeoprocessing
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr

from natcap.invest import __version__
from natcap.invest import gettext
import natcap.invest.spec
from natcap.invest.reports import jinja_env, raster_utils, report_constants, vector_utils
from natcap.invest.reports.raster_utils import RasterDatatype, RasterPlotConfig
from natcap.invest.unit_registry import u


LOGGER = logging.getLogger(__name__)

TEMPLATE = jinja_env.get_template('models/seasonal_water_yield.html')

MAP_WIDTH = 450 # pixels
TARGET_NODATA = -1
TARGET_DTYPE = gdal.GDT_Float32

qf_label_month_map = {
    f"qf_{month_index+1}": str(month_index+1) for month_index in range(12)
}


def _label_to_month(row):
    return qf_label_month_map[row['MonthLabel']]


def _create_aggregate_map(geodataframe, extent_feature, xy_ratio, attribute,
                          title):
    attr_map = altair.Chart(geodataframe).mark_geoshape(
        clip=True,
        stroke="white",
        strokeWidth=0.5
    ).project(
        type='identity',
        reflectY=True,
        fit=extent_feature
    ).encode(
        color=attribute,
        tooltip=[altair.Tooltip(attribute, title=attribute)]
    ).properties(
        width=MAP_WIDTH,
        height=MAP_WIDTH / xy_ratio,
        title=title
    ).configure_legend(**vector_utils.LEGEND_CONFIG)

    return attr_map.to_json()


def _create_linked_monthly_plots(aoi_vector_path, aggregate_csv_path):
    map_df = geopandas.read_file(aoi_vector_path)
    values_df = pandas.read_csv(aggregate_csv_path)
    values_df.month = values_df.month.astype(str)

    extent_feature, xy_ratio = vector_utils.get_geojson_bbox(map_df)

    feat_select = altair.selection_point(fields=["geom_id"], name="feat_select", value=0)

    attr_map = altair.Chart(map_df).mark_geoshape(
        clip=True, stroke="white", strokeWidth=0.5
        ).project(
            type='identity',
            reflectY=True,
            fit=extent_feature
    ).encode(
        color=altair.condition(
            feat_select,
            altair.value("seagreen"),
            altair.value("lightgray")
        ),
        tooltip=[altair.Tooltip("geom_id", title="FID")]
    ).properties(
        width=MAP_WIDTH,
        height=MAP_WIDTH / xy_ratio,
        title="AOI"
    ).add_params(
        feat_select
    )

    base_chart = altair.Chart(values_df)

    bar_chart = base_chart.mark_bar().transform_fold(
        ['baseflow', 'quickflow']
    ).encode(
        altair.X("month(month):O").title("Month"),
        altair.Y("sum(value):Q").title("Quickflow + Baseflow (m3/s)"),
        altair.Order(field='key', sort='ascending'),
        color=altair.Color('key:N').scale(
            domain=['quickflow', "baseflow", "precipitation"],
            range=['#fdae6b', '#9ecae1', "#0500a3"]
        ),
        tooltip=[altair.Tooltip(val, aggregate="sum", type="quantitative",
                                format='.5f', title=val)
                 for val in ["quickflow", "baseflow", "precipitation"]]
    )

    precip_chart = base_chart.mark_line().encode(
            altair.X("month(month):O").title("Month"),
        altair.Y(
            "sum(precipitation)",
            axis=altair.Axis(orient="right")
        ).title("Precipitation (m3/s)"),
        color=altair.value('#0500a3')
    )

    combined_chart = altair.layer(bar_chart, precip_chart).resolve_scale(
        y='independent'
    ).transform_filter(
        feat_select
    ).properties(
        title=altair.Title(altair.expr(
            f'"Mean Quickflow + Baseflow for Feature, FID " + {feat_select.name}.geom_id')
        )
    )

    chart = attr_map | combined_chart
    return chart.to_json()


def _mask_raster_list(source_raster_path_list, mask_stream_raster_path,
                      target_masked_vrt_path_list):
    """Using a raster of 1s and 0s, determine which pixels remain in output.

    Args:
        source_raster_path_list (list): List of paths to source rasters containing
            pixel values to mask by the binary raster provided as a mask.
        mask_stream_raster_path (str): The path to a stream network raster of
            1s and 0s indicating whether a pixel should (0) or should not (1)
            be copied to the target raster.
        target_masked_vrt_path_list (list): List of paths to VRTs where the target
            rasters should be written.

    Returns:
        None
    """
    source_raster_info = pygeoprocessing.get_raster_info(source_raster_path_list[0])
    source_nodata = source_raster_info['nodata'][0]
    target_numpy_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(TARGET_DTYPE)

    def _mask_op(mask, raster):
        result = numpy.full(mask.shape, TARGET_NODATA,
                            dtype=target_numpy_dtype)
        valid_pixels = (
            ~pygeoprocessing.array_equals_nodata(raster, source_nodata) &
            (mask == 0))
        result[valid_pixels] = raster[valid_pixels]
        return result

    for source_raster_path, target_vrt_path in zip(
            source_raster_path_list, target_masked_vrt_path_list):
        pygeoprocessing.raster_calculator(
            [(mask_stream_raster_path, 1), (source_raster_path, 1)], _mask_op,
            target_vrt_path, TARGET_DTYPE, TARGET_NODATA)


def report(file_registry, args_dict, model_spec, target_html_filepath):
    """Generate an html summary of Seasonal Water Yield results.

    Args:
        file_registry (dict): The ``natcap.invest.FileRegistry.registry``
            that was returned by ``natcap.invest.seasonal_water_yield.execute``.
        args_dict (dict): The arguments that were passed to
            ``natcap.invest.seasonal_water_yield.execute``.
        model_spec (natcap.invest.spec.ModelSpec):
            ``natcap.invest.seasonal_water_yield.MODEL_SPEC``
        target_html_filepath (str): path to an html file generated by this
            function.

    Returns:
        None
    """

    # qb and vri_sum plots from the output aggregate vector
    aggregated_results = geopandas.read_file(file_registry['aggregate_vector'])
    extent_feature, xy_ratio = vector_utils.get_geojson_bbox(aggregated_results)

    qb_map_json = _create_aggregate_map(
        aggregated_results, extent_feature, xy_ratio, 'qb',
        gettext("Mean local recharge value within the watershed "
                f"({model_spec.get_output('aggregate_vector').get_field('qb').units})"))
    qb_map_caption = [
        model_spec.get_output('aggregate_vector').get_field('qb').about,
        gettext('Values are in millimeters, but should be interpreted as '
                'relative values, not absolute values.')]

    vri_sum_map_json = _create_aggregate_map(
        aggregated_results, extent_feature, xy_ratio, 'vri_sum',
        gettext("Total recharge contribution of the watershed "
                f"({model_spec.get_output('aggregate_vector').get_field('vri_sum').units})"))
    vri_sum_map_caption = [
        model_spec.get_output('aggregate_vector').get_field('vri_sum').about,
        gettext('The sum of ``Vri_[suffix].tif`` pixel values within the watershed.')]

    vector_map_source_list = [model_spec.get_output('aggregate_vector').path]

    if args_dict['user_defined_local_recharge']:
        # Quickflow isn't calculated if `user_defined_local_recharge`
        # so we cannot construct monthly average qf + b charts
        qf_b_charts = None
    else:
        # Monthly quickflow + baseflow plots and map
        qf_b_charts_json = _create_linked_monthly_plots(
                file_registry['aggregate_vector'],
                file_registry['monthly_qf_table'])
        qf_b_charts_caption = gettext(
            """
            This chart displays the monthly combined average baseflow + quickflow for
            each feature within the AOI. Select a feature on the AOI map to see the
            values for that feature. Selecting multiple features will display the sum
            of their values.
            """
        )
        qf_b_charts_source_list = [
            model_spec.get_output('monthly_qf_table').path,
            model_spec.get_output('aggregate_vector').path]
        qf_b_charts = {
            'json': qf_b_charts_json,
            'caption': qf_b_charts_caption,
            'sources': qf_b_charts_source_list
        }

    # Raster config lists
    stream_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['pit_filled_dem'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('pit_filled_dem')),
        RasterPlotConfig(
            raster_path=file_registry['stream'],
            datatype=RasterDatatype.binary_high_contrast,
            spec=model_spec.get_output('stream'))]

    output_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['b'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('b'))
    ]

    input_raster_config_list = [
        RasterPlotConfig(
            raster_path=args_dict['dem_raster_path'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_input('dem_raster_path')),
        RasterPlotConfig(
            raster_path=args_dict['lulc_raster_path'],
            datatype=RasterDatatype.nominal,
            spec=model_spec.get_input('lulc_raster_path'))
    ]

    if args_dict['user_defined_local_recharge']:
        input_raster_config_list.append(
            RasterPlotConfig(
                raster_path=args_dict['l_path'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_input('l_path')))
        qf_rasters = None
        raster_outputs_title = 'Annual Baseflow'

    else:
        output_raster_config_list.extend([
            RasterPlotConfig(
                raster_path=file_registry['annual_precip'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('annual_precip')),
            RasterPlotConfig(
                raster_path=file_registry['aet'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('aet')),
            RasterPlotConfig(
                raster_path=file_registry['cn'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('cn')),
            RasterPlotConfig(
                raster_path=file_registry['l'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('l'))])
        raster_outputs_title = 'Additional Raster Outputs'

        input_raster_config_list.append(
            RasterPlotConfig(
                raster_path=args_dict['soil_group_path'],
                datatype=RasterDatatype.nominal,
                spec=model_spec.get_input('soil_group_path')))

        # Quickflow outputs are only created if not `user_defined_local_recharge`
        masked_annual_qf_path = os.path.join(
                args_dict['workspace_dir'], 'intermediate_outputs',
                'annual_qf_masked' + (args_dict.get('results_suffix') or '') + '.vrt')
        masked_monthly_qf_paths = [os.path.join(
                args_dict['workspace_dir'], 'intermediate_outputs',
                f'qf_{month}_masked' + (args_dict.get('results_suffix') or '') + '.vrt')
                for month in range(1, 13)]
        _mask_raster_list(
                [file_registry['qf']] + [file_registry['qf_[MONTH]'][str(month)]
                                         for month in range(1, 13)],
                file_registry['stream'],
                [masked_annual_qf_path] + masked_monthly_qf_paths)

        annual_qf_raster_config = RasterPlotConfig(
                raster_path=masked_annual_qf_path,
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('qf'),
                title=gettext(
                    f"Annual Quickflow ({os.path.basename(file_registry['qf'])})"
                ))

        monthly_qf_raster_config_list = [
            RasterPlotConfig(
                raster_path=masked_qf_path,
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('qf_[MONTH]'),
                title=gettext(
                    f"Quickflow for month {month} "
                    f"({os.path.basename(file_registry['qf_[MONTH]'][str(month)])})"
                )
            ) for month, masked_qf_path in zip(range(1, 13), masked_monthly_qf_paths)]

        annual_qf_img_src = raster_utils.plot_and_base64_encode_rasters(
                [annual_qf_raster_config])
        monthly_qf_plots = raster_utils.plot_raster_facets(
                [raster_config.raster_path for raster_config
                 in monthly_qf_raster_config_list],
                'continuous',
                title_list=[raster_config.title for raster_config
                            in monthly_qf_raster_config_list])
        monthly_qf_img_src = raster_utils.base64_encode(monthly_qf_plots)
        monthly_qf_displayname = os.path.basename(
                monthly_qf_raster_config_list[0].raster_path).replace('1', '[MONTH]')
        monthly_qf_base_raster_name = os.path.basename(
                file_registry['qf_[MONTH]']['1']).replace('1', '[MONTH]')
        stream_path = os.path.basename(file_registry['stream'])
        qf_caption_note = gettext(
                f"The stream network in `{stream_path}`"
                " has been used as a mask to help improve contrast for the"
                " quickflow values of non-stream pixels."),
        qf_raster_caption = [
            (f'{os.path.basename(annual_qf_raster_config.raster_path)}:'
             f'{annual_qf_raster_config.spec.about}. This VRT is derived from'
             f' `{os.path.basename(file_registry['qf'])}`, masked by `{stream_path}`.'
             f' (Units: {annual_qf_raster_config.spec.units})'),
            (f'{monthly_qf_displayname}:'
             f'{monthly_qf_raster_config_list[0].spec.about}. This VRT is derived from'
             f' the monthly quickflow rasters, `{monthly_qf_base_raster_name}`,'
             f' masked by `{stream_path}`.'
             f' (Units: {monthly_qf_raster_config_list[0].spec.units})')
        ]

        qf_rasters = {
            'annual_qf_img_src': annual_qf_img_src,
            'monthly_qf_img_src': monthly_qf_img_src,
            'qf_caption_note': qf_caption_note,
            'qf_caption': qf_raster_caption}

    # Create raster image sources and captions:
    stream_img_src = raster_utils.plot_and_base64_encode_rasters(
            stream_raster_config_list)
    stream_raster_caption = raster_utils.caption_raster_list(
            stream_raster_config_list)
    stream_outputs_heading = gettext(
            'Stream Network Maps (Flow Algorithm: '
            f'{args_dict["flow_dir_algorithm"]}, '
            'Threshold Flow Accumulation value: '
            f'{args_dict["threshold_flow_accumulation"]})')

    outputs_img_src = raster_utils.plot_and_base64_encode_rasters(
            output_raster_config_list)
    output_raster_caption = raster_utils.caption_raster_list(
            output_raster_config_list)

    output_raster_stats_table = raster_utils.raster_workspace_summary(
            file_registry).to_html(na_rep='')

    input_raster_stats_table = raster_utils.raster_inputs_summary(
            args_dict, model_spec, check_csv_for_rasters=True).to_html(na_rep='')

    inputs_img_src = raster_utils.plot_and_base64_encode_rasters(
            input_raster_config_list)
    inputs_raster_caption = raster_utils.caption_raster_list(
            input_raster_config_list)

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
            raster_group_caption=report_constants.RASTER_GROUP_CAPTION,
            stats_table_note=report_constants.STATS_TABLE_NOTE,
            stream_img_src=stream_img_src,
            stream_caption=stream_raster_caption,
            stream_outputs_heading=stream_outputs_heading,
            outputs_title=raster_outputs_title,
            outputs_img_src=outputs_img_src,
            outputs_caption=output_raster_caption,
            qf_rasters=qf_rasters,
            output_raster_stats_table=output_raster_stats_table,
            input_raster_stats_table=input_raster_stats_table,
            inputs_img_src=inputs_img_src,
            inputs_caption=inputs_raster_caption,
            qf_b_charts=qf_b_charts,
            qb_map_json=qb_map_json,
            qb_map_caption=qb_map_caption,
            vri_sum_map_json=vri_sum_map_json,
            vri_sum_map_caption=vri_sum_map_caption,
            aggregate_map_source_list=vector_map_source_list,
            model_spec_outputs=model_spec.outputs,
        ))

    LOGGER.info(f'Created {target_html_filepath}')
