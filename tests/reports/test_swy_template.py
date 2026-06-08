import time
import unittest

import pandas
from bs4 import BeautifulSoup

from natcap.invest.seasonal_water_yield import MODEL_SPEC
from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/seasonal_water_yield.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args(model_spec):
    locale = 'en'
    report_filepath = 'swy_report_test.html'
    invest_version = '987.65.0'
    timestamp = time.strftime('%Y-%m-%d %H:%M')
    args_dict = {'suffix': 'test'}
    img_src = 'bAse64eNcoDEdIMagE'
    output_stats_table = '<table class="test__output-stats-table"></table>'
    input_stats_table = '<table class="test__input-stats-table"></table>'
    stats_table_note = 'This is a test!'
    raster_group_caption = 'This is another test!'
    stream_caption = ['stream.tif:Stream map.']
    heading = 'Test heading'
    outputs_caption = ['results.tif:Results map.']
    inputs_caption = ['input.tif:Input map.']
    vegalite_json = '{}'
    caption = 'figure caption'
    agg_map_source_list = ['/source/file.shp']

    return {
        'locale': locale,
        'report_script': model_spec.reporter,
        'invest_version': invest_version,
        'report_filepath': report_filepath,
        'model_id': model_spec.model_id,
        'model_name': model_spec.model_title,
        'model_description': model_spec.about,
        'userguide_page': model_spec.userguide,
        'timestamp': timestamp,
        'args_dict': args_dict,
        'raster_group_caption': raster_group_caption,
        'stats_table_note': stats_table_note,
        'stream_img_src': img_src,
        'stream_caption': stream_caption,
        'stream_outputs_heading': heading,
        'outputs_heading': heading,
        'outputs_img_src': img_src,
        'outputs_caption': outputs_caption,
        'qf_rasters': None,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'qf_b_charts': None,
        'qb_map_json': vegalite_json,
        'qb_map_caption': caption,
        'vri_sum_map_json': vegalite_json,
        'vri_sum_map_caption': caption,
        'aggregate_map_source_list': agg_map_source_list,
        'model_spec_outputs': model_spec.outputs
    }

class SeasonalWaterYieldTemplateTests(unittest.TestCase):
    """Unit tests for SWY template."""

    def test_render_without_user_defined_recharge(self):
        """Test report rendering when user_defined_local_recharge is False."""

        render_args = _get_render_args(MODEL_SPEC)

        render_args['qf_rasters'] = {
            'annual_qf_img_src': 'bAse64eNcoDEdIMagE',
            'monthly_qf_img_src': 'bAse64eNcoDEdIMagE',
            'qf_caption': 'test caption'
        }
        render_args['qf_b_charts'] = {
            'json': '{}',
            'caption': 'test caption',
            'sources': ['/src/path.shp', '/src/path.csv']
        }
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        # Includes quickflow raster section and monthly qf + b charts section
        self.assertEqual(len(sections), 10)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_render_with_user_defined_recharge(self):
        """Test report rendering when user_defined_local_recharge is True."""

        render_args = _get_render_args(MODEL_SPEC)
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        # No quickflow raster section or monthly qf + b charts section
        self.assertEqual(len(sections), 8)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')
