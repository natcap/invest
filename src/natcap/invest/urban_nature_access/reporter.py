import logging
import time

import altair
import geopandas
import geometamaker
import matplotlib
import numpy
import pandas
import pygeoprocessing
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from osgeo import gdal
import re

from natcap.invest import validation
from natcap.invest import __version__
from natcap.invest import gettext
from natcap.invest.reports import jinja_env, raster_utils, report_constants, \
    vector_utils
from natcap.invest.spec import ModelSpec, FileRegistry

from natcap.invest.reports.raster_utils import RasterDatatype, \
    RasterPlotConfig, RasterTransform, SpecialValueConfig

LOGGER = logging.getLogger(__name__)

TEMPLATE = jinja_env.get_template('models/urban_nature_access.html')

MAP_WIDTH = 450  # pixels

NEAR_ZERO_RANGE = (-0.01, 0.01)


def infer_continuous_or_divergent(raster_path: str) -> str:
    """Infer if raster should have a 'continuous' or 'divergent' color ramp.

    Rules:
        - If min value < ~0 and max value > ~0--> 'divergent'
        - Else --> 'continuous'

    Args:
        raster_path (str): Path to raster.

    Returns:
        str: 'continuous' or 'divergent'
    """
    # read raster's geometamaker metadata to get min value
    resource = geometamaker.describe(raster_path)
    min_val = float(
        resource.data_model.bands[0].gdal_metadata['STATISTICS_MINIMUM'])
    max_val = float(
        resource.data_model.bands[0].gdal_metadata['STATISTICS_MAXIMUM'])

    if min_val < NEAR_ZERO_RANGE[0] and max_val > NEAR_ZERO_RANGE[1]:
        return RasterDatatype.divergent
    else:
        return RasterDatatype.continuous


def get_min_max_for_vector_colorbar(datamin, datamax):
    """Get a min and max for colorbar range that ensures legend displays nicely"""
    if abs(datamin) < 0.2*datamax:
        # set min to -20% of max if that value is smaller than actual min to
        # avoid having an overly compressed colorbar
        domain_min = round(min(datamin, -0.2*datamax), -1)
        return (domain_min, round(datamax, -1))

    elif datamax < -0.2*datamin:
        domain_max = round(max(datamax, -0.2*datamin), -1)
        return (round(datamin, -1), round(domain_max, -1))

    return (round(datamin, -1), round(datamax, -1))


def _create_vector_maps(
        geodataframe,
        xy_ratio):
    """Create choropleth maps and return one Vega JSON spec."""

    display_names = {
        'SUP_DEMadm_cap': (
            gettext('Balance'),
            gettext('Per Person Average Urban Nature Supply/Demand Balance by Admin Unit')),
        'Pund_adm': (
            gettext('Population'),
            gettext('Population Undersupplied with Urban Nature by Admin Unit')),
        'Povr_adm': (
            gettext('Population'),
            gettext('Population Oversupplied with Urban Nature by Admin Unit'))
    }

    charts = []
    for attribute, (legend_title, chart_title) in display_names.items():
        if (geodataframe[attribute] < 0).any():
            domain_min, domain_max = get_min_max_for_vector_colorbar(
                min(geodataframe[attribute]), max(geodataframe[attribute])
            )
            scale = altair.Scale(
                scheme='purpleorange',
                reverse=True,
                domain=[domain_min, 0, domain_max])
            legend = altair.Legend(
                title=legend_title,
                orient='right',
                values=[domain_min, numpy.mean([domain_max, domain_min]), domain_max])
        else:
            scale = altair.Scale(scheme='purples')
            legend = altair.Legend(title=legend_title, orient='right')

        chart = altair.Chart(geodataframe).mark_geoshape(
            stroke='white',
            strokeWidth=0.5
        ).project(
            type='identity',
            reflectY=True
        ).encode(
            color=altair.Color(
                f'{attribute}:Q',
                scale=scale,
                legend=legend),
            tooltip=[
                altair.Tooltip(
                    f'{attribute}:Q',
                    title=legend_title,
                    format=',.2f')
            ]
        ).properties(
            width=MAP_WIDTH,
            height=MAP_WIDTH / xy_ratio,
            title=altair.TitleParams(
                text=chart_title,
                subtitle=gettext('Attribute: ') + attribute
            ))

        charts.append(chart)

    combined_chart = altair.concat(
        *charts,
        columns=2
    ).resolve_scale(
        color='independent'
    ).configure_legend(
        **vector_utils.LEGEND_CONFIG
    )

    return combined_chart.to_json()


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

    input_raster_config_list = [
        RasterPlotConfig(
            raster_path=args_dict['lulc_raster_path'],
            datatype=RasterDatatype.nominal,
            spec=model_spec.get_input('lulc_raster_path')
        ),
        RasterPlotConfig(
            raster_path=args_dict['population_raster_path'],
            datatype=RasterDatatype.continuous,
            spec=model_spec.get_input('population_raster_path')
        ),
    ]

    if args_dict['search_radius_mode'] != "radius per urban nature class":
        intermediate_raster_config_list = [
            RasterPlotConfig(
                raster_path=file_registry['urban_nature_area'],#TODO check if these should be divergent
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('urban_nature_area')
            ),
            RasterPlotConfig(
                raster_path=file_registry['urban_nature_population_ratio'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('urban_nature_population_ratio')
            ),
        ]
        intermediates_img_src = raster_utils.plot_and_base64_encode_rasters(
            intermediate_raster_config_list)
        intermediates_caption = raster_utils.caption_raster_list(
            intermediate_raster_config_list)
    else:
        intermediates_img_src = intermediates_caption = None

    output_accessible_list = []
    output_balance_list = []
    output_balance_percapita_list = []
    if args_dict['search_radius_mode'] == 'radius per population group':
        pop_group_list = list(
            filter(lambda x: re.match('^pop_', x),
                   validation.load_fields_from_vector(
                       args_dict['admin_boundaries_vector_path'])))

        for group in pop_group_list:
            output_accessible_list.append(
                RasterPlotConfig(
                    raster_path=file_registry['accessible_urban_nature_to_[POP_GROUP]'][group], # cannot used patterned outputs bc using file registry.registry dict
                    datatype=RasterDatatype.continuous,
                    spec=model_spec.get_output('accessible_urban_nature_to_[POP_GROUP]'),
                )
            )
            output_balance_list.append(
                RasterPlotConfig(
                    raster_path=file_registry['urban_nature_balance_[POP_GROUP]'][group],
                    datatype=RasterDatatype.divergent,
                    spec=model_spec.get_output('urban_nature_balance_[POP_GROUP]'),
                    transform=RasterTransform.log
                )
            )
            output_balance_percapita_list.append(
                RasterPlotConfig(
                    raster_path=file_registry['urban_nature_balance_percapita_[POP_GROUP]'][group],
                    datatype=RasterDatatype.divergent,
                    spec=model_spec.get_output('urban_nature_balance_percapita_[POP_GROUP]'),
                    transform=RasterTransform.log
                )
            )

            output_accessible_raster_caption = re.sub(
                r'(pop_).*?(\.tif)', r'\1POP_GROUP\2',
                output_accessible_list[0].caption)

            output_balance_img_src = raster_utils.plot_and_base64_encode_rasters(
                output_balance_list)
            output_balance_raster_caption = re.sub(
                r'(pop_).*?(\.tif)', r'\1POP_GROUP\2',
                output_balance_list[0].caption)

            output_balance_percapita_img_src = raster_utils.plot_and_base64_encode_rasters(
                output_balance_percapita_list)
            output_balance_percapita_raster_caption = re.sub(
                r'(pop_).*?(\.tif)', r'\1POP_GROUP\2',
                output_balance_percapita_list[0].caption)

    elif args_dict['search_radius_mode'] == 'radius per urban nature class':
        nature_class_list = pandas.read_csv(args_dict[
            'lulc_attribute_table'])['lucode']

        for lucode in nature_class_list:
            if str(lucode) in file_registry['accessible_urban_nature_lucode_[LUCODE]']:
                output_accessible_list.append(
                    RasterPlotConfig(
                        raster_path=file_registry['accessible_urban_nature_lucode_[LUCODE]'][str(lucode)],
                        datatype=RasterDatatype.continuous,
                        spec=model_spec.get_output('accessible_urban_nature_lucode_[LUCODE]'),
                    )
                )
                output_accessible_raster_caption = raster_utils.caption_raster_list(
                    output_accessible_list)[0].replace("lucode_1", "lucode_LUCODE")
            output_balance_img_src = None
            output_balance_raster_caption = None
            output_balance_percapita_img_src = None
            output_balance_percapita_raster_caption = None

    else:
        output_accessible_list = [
            RasterPlotConfig(
                raster_path=file_registry['accessible_urban_nature'],
                datatype=RasterDatatype.continuous,
                spec=model_spec.get_output('accessible_urban_nature'),
            )]
        output_accessible_raster_caption = raster_utils.caption_raster_list(
                output_accessible_list)
        output_balance_img_src = None
        output_balance_raster_caption = None
        output_balance_percapita_img_src = None
        output_balance_percapita_raster_caption = None

    output_raster_config_list = [
        RasterPlotConfig(
            raster_path=file_registry['urban_nature_supply_percapita'],
            spec=model_spec.get_output('urban_nature_supply_percapita'),
            datatype=RasterDatatype.continuous,
            transform=RasterTransform.linear
        ),
        RasterPlotConfig(
            raster_path=file_registry['urban_nature_demand'],
            spec=model_spec.get_output('urban_nature_demand'),
            datatype=RasterDatatype.continuous,
            transform=RasterTransform.linear
        ),
        RasterPlotConfig(
            raster_path=file_registry['urban_nature_balance_percapita'],
            spec=model_spec.get_output('urban_nature_balance_percapita'),
            datatype=RasterDatatype.divergent,
            transform=RasterTransform.linear,
            special_values=SpecialValueConfig(
                extend="both",
                threshold=(-1000, 1000),
                label=('<-1000', '>1000'),
                color=('#8C4300', '#1F0737')
            )
        ),
        RasterPlotConfig(
            raster_path=file_registry['urban_nature_balance_totalpop'],
            spec=model_spec.get_output('urban_nature_balance_totalpop'),
            datatype=RasterDatatype.divergent,
            transform=RasterTransform.linear,
            special_values=SpecialValueConfig(
                extend="both",
                threshold=(-1000, 1000),
                label=('<-1000', '>1000'),
                color=('#8C4300', '#1F0737')
            )
        )
    ]

    inputs_img_src = raster_utils.plot_and_base64_encode_rasters(
        input_raster_config_list)
    inputs_raster_caption = raster_utils.caption_raster_list(
        input_raster_config_list)

    outputs_img_src = raster_utils.plot_and_base64_encode_rasters(
        output_raster_config_list)
    outputs_raster_caption = raster_utils.caption_raster_list(
        output_raster_config_list)

    output_accessible_img_src = raster_utils.plot_and_base64_encode_rasters(
        output_accessible_list)

    input_raster_stats_table = raster_utils.raster_inputs_summary(
        args_dict, model_spec).to_html(na_rep='')

    output_raster_stats_table = raster_utils.raster_workspace_summary(
        file_registry).to_html(na_rep='')

    admin_boundaries_gdf = geopandas.read_file(file_registry['admin_boundaries'])
    _, xy_ratio = vector_utils.get_geojson_bbox(admin_boundaries_gdf)

    admin_map_json = _create_vector_maps(
        admin_boundaries_gdf,
        xy_ratio
    )

    # TODO would people prefer to see the various attrs in vector ie by admin unit, or rasters
    vector_field_list = ['SUP_DEMadm_cap', 'Pund_adm', 'Povr_adm']
    vector_plot_captions = [
        field + ':' + model_spec.get_output('admin_boundaries').get_field(field).about
        for field in vector_field_list
    ]

    admin_boundaries_map_source_list = [
        model_spec.get_output('admin_boundaries').path
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
            # agg_results_table=agg_results_table,
            inputs_img_src=inputs_img_src,
            inputs_caption=inputs_raster_caption,
            outputs_img_src=outputs_img_src,
            outputs_caption=outputs_raster_caption,
            output_accessible_img_src=output_accessible_img_src,
            output_accessible_raster_caption=output_accessible_raster_caption,
            output_balance_img_src=output_balance_img_src,
            output_balance_raster_caption=output_balance_raster_caption,
            output_balance_percapita_img_src=output_balance_percapita_img_src,
            output_balance_percapita_raster_caption=output_balance_percapita_raster_caption,
            intermediates_img_src=intermediates_img_src,
            intermediates_caption=intermediates_caption,
            raster_group_caption=report_constants.RASTER_GROUP_CAPTION,
            lulc_pre_caption=report_constants.LULC_PRE_CAPTION,
            output_raster_stats_table=output_raster_stats_table,
            input_raster_stats_table=input_raster_stats_table,
            stats_table_note=report_constants.STATS_TABLE_NOTE,
            admin_map_json=admin_map_json,
            vector_plot_captions=vector_plot_captions,
            admin_boundaries_map_source_list=admin_boundaries_map_source_list,
            model_spec_outputs=model_spec.outputs,
        ))

    LOGGER.info(f'Created {target_html_filepath}')
