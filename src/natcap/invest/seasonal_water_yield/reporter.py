import logging
import os
import time

import altair
import geopandas
import numpy
import pandas
import pygeoprocessing
from osgeo import gdal
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

qf_label_month_map = {
    f"qf_{month_index+1}": str(month_index+1) for month_index in range(12)
}


def _label_to_month(row):
    return qf_label_month_map[row['MonthLabel']]


def _make_qf_plots(input_vector_path, id_var):
    df = geopandas.read_file(input_vector_path)
    melted_df = df.melt(
        id_vars=[id_var],
        var_name="MonthLabel",
        value_vars=[f'qf_{month_index+1}' for month_index in range(12)],
        value_name="Quickflow")

    melted_df['Month'] = melted_df.apply(_label_to_month, axis=1)

    max_qf = round(melted_df.max()['Quickflow']) + 1
    plots = []

    for poly_id in set(melted_df[id_var]):
        plt = altair.Chart(melted_df[melted_df[id_var] == poly_id]).mark_bar().encode(
                altair.X("month(Month):T").title("Month"),
                altair.Y("Quickflow").title("Quickflow (m^3/s)").scale(
                    domain=(0, max_qf), clamp=True)
        ).properties(title=f"Mean Quickflow for Polygon, FID {poly_id}")

        plots.append(plt.to_json())

    return plots


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
#    qb_map = altair.Chart(aggregated_results).mark_geoshape(
#        clip=True,
#        stroke="white",
#        strokeWidth=0.5
#    ).project(
#        type='identity',
#        reflectY=True,
#        fit=extent_feature
#    ).encode(
#        color='qb',
#        tooltip=[altair.Tooltip("qb", title="qb")]
#    ).properties(
#        width=MAP_WIDTH,
#        height=MAP_WIDTH / xy_ratio,
#        title='Mean local recharge value within the watershed'
#    ).configure_legend(**vector_utils.LEGEND_CONFIG)
#    qb_map_caption = [model_spec.get_output(
#        'aggregate_vector').get_field('qb').about]


def _aggregate_monthly_qf_cubic_meters(aoi_path, file_registry,
                                       aggregate_vector_path):
    if os.path.exists(aggregate_vector_path):
        print(
            '%s exists, deleting and writing new output',
            aggregate_vector_path)
        os.remove(aggregate_vector_path)

    original_aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)

    driver = gdal.GetDriverByName('ESRI Shapefile')
    driver.CreateCopy(aggregate_vector_path, original_aoi_vector)
    gdal.Dataset.__swig_destroy__(original_aoi_vector)
    original_aoi_vector = None
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, 1)
    aggregate_layer = aggregate_vector.GetLayer()

    path_field_tuples = [(file_registry['qf_[MONTH]'][str(month_index +1)],
                         f"qf_{month_index+1}") for month_index in range(12)]

    raster_info = pygeoprocessing.get_raster_info(path_field_tuples[0][0])
    pixel_area_m2 = numpy.prod([abs(x) for x in raster_info['pixel_size']])

    for raster_path, aggregate_field_id in path_field_tuples:
        aggregate_stats = pygeoprocessing.zonal_statistics(
            (raster_path, 1), aggregate_vector_path)

        aggregate_field = ogr.FieldDefn(aggregate_field_id, ogr.OFTReal)
        aggregate_field.SetWidth(24)
        aggregate_field.SetPrecision(11)
        aggregate_layer.CreateField(aggregate_field)

        aggregate_layer.ResetReading()
        for poly_index, poly_feat in enumerate(aggregate_layer):
            qf_sum_meters = aggregate_stats[poly_index]['sum'] * 0.001
            total_qf = qf_sum_meters * pixel_area_m2

            poly_feat.SetField(aggregate_field_id, float(total_qf))
            aggregate_layer.SetFeature(poly_feat)

    fid_field = ogr.FieldDefn('geom_fid', ogr.OFTInteger)
    aggregate_layer.CreateField(fid_field)
    for feature in aggregate_layer:
        feature_id = feature.GetFID()
        feature.SetField('geom_fid', feature_id)
        aggregate_layer.SetFeature(feature)

    aggregate_layer.SyncToDisk()
    aggregate_layer = None
    gdal.Dataset.__swig_destroy__(aggregate_vector)
    aggregate_vector = None


def report(file_registry, args_dict, model_spec, target_html_filepath):
    """Generate an html summary of Coastal Vulnerability results.

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

    aggregated_results = geopandas.read_file(file_registry['aggregate_vector'])
    extent_feature, xy_ratio = vector_utils.get_geojson_bbox(aggregated_results)

    aggregated_monthly_qf_path = os.path.join(args_dict['workspace_dir'],
                                              'monthly_qf_mean_aggregate.shp')
    _aggregate_monthly_qf_cubic_meters(args_dict['aoi_path'], file_registry,
                                       aggregated_monthly_qf_path)
    qf_plots_json = _make_qf_plots(aggregated_monthly_qf_path, 'geom_fid')
    qf_plots_json_tuples = [(qf_json, f'qf_plot_{poly_id}') for poly_id, qf_json in
                            enumerate(qf_plots_json)]
    qf_plots_caption = gettext(
        """

        """
    )

    qb_map_json = _create_aggregate_map(
        aggregated_results, extent_feature, xy_ratio, 'qb',
        gettext('Mean local recharge value within the watershed'))
    qb_map_caption = [
        model_spec.get_output('aggregate_vector').get_field('qb').about,
        gettext('Values are in millimeters, but should be interpreted as '
                'relative values, not absolute values.')]

    vri_sum_map_json = _create_aggregate_map(
        aggregated_results, extent_feature, xy_ratio, 'vri_sum',
        gettext('Total recharge contribution of the watershed'))
    vri_sum_map_caption = [
        model_spec.get_output('aggregate_vector').get_field('vri_sum').about,
        gettext('The sum of ``Vri_[suffix].tif`` pixel values within the watershed.')]

    vector_map_source_list = [model_spec.get_output('aggregate_vector').path]

    # Raster config lists
    stream_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['pit_filled_dem'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('pit_filled_dem')),
        RasterPlotConfig(
            raster_path=file_registry['stream'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('stream'))]

    output_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['b'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('b')),
        RasterPlotConfig(
            raster_path=file_registry['annual_precip'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('annual_precip')),
        RasterPlotConfig(
            raster_path=file_registry['aet'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('aet'))]

    qf_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['qf'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('qf'))]

    intermediate_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['cn'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_output('cn'))]

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

    local_recharge_config = RasterPlotConfig(
        raster_path=file_registry['l_aligned'],
        datatype=RasterDatatype.continuous,
        spec=model_spec.get_output('l_aligned'))

    if args['user_defined_local_recharge']:
        input_raster_config_list.append(local_recharge_config)
    else:
        output_raster_config_list.append(local_recharge_config)

    # Create raster image sources and captions:
    stream_img_src = raster_utils.plot_and_base64_encode_rasters(
            stream_raster_config_list)
    stream_raster_caption = raster_utils.caption_raster_list(
            stream_raster_config_list)
    stream_outputs_heading = gettext(
            'Stream Network Maps (Flow Algorithm: '
            f'{args_dict["flow_dir_algorithm"]}, '
            'Threshold Flow Accumulation value: '
            f'{args_dict["thresholdf_flow_accumulation"]})')

    outputs_img_src = raster_utils.plot_and_base64_encode_rasters(
            output_raster_config_list)
    output_raster_caption = raster_utils.caption_raster_list(
            output_raster_config_list)

    qf_img_src = raster_utils.plot_and_base64_encode_rasters(
            qf_raster_config_list)
    qf_raster_caption = raster_utils.caption_raster_list(
            qf_raster_config_list)

    intermediate_img_src = raster_utils.plot_and_base64_encode_rasters(
            intermediate_raster_config_list)
    intermediate_raster_caption = raster_utils.caption_raster_list(
            intermediate_raster_config_list)

    output_raster_stats_table = raster_utils.raster_workspace_summary(
            file_registry).to_html(na_rep='')

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
            stream_img_src=stream_img_src,
            stream_caption=stream_raster_caption,
            stream_outputs_heading=stream_outputs_heading,
            outputs_img_src=outputs_img_src,
            outputs_caption=output_raster_caption,
            qf_img_src=qf_img_src,
            qf_caption=qf_raster_caption,
            intermediate_img_src=intermediate_img_src,
            intermediate_caption=intermediate_raster_caption,
            output_raster_stats_table=output_raster_stats_table,
            inputs_img_src=inputs_img_src,
            inputs_caption=inputs_raster_caption,
            qf_plots_json_tuples=qf_plots_json_tuples,
            qf_plots_caption=qf_plots_caption,
            qb_map_json=qb_map_json,
            qb_map_caption=qb_map_caption,
            vri_sum_map_json=vri_sum_map_json,
            vri_sum_map_caption=vri_sum_map_caption,
            aggregate_map_source_list=vector_map_source_list,
            model_spec_outputs=model_spec.outputs,
        ))

    LOGGER.info(f'Created {target_html_filepath}')
