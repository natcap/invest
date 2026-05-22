import time
import unittest

import pandas
from bs4 import BeautifulSoup

from natcap.invest.annual_water_yield import MODEL_SPEC
from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/annual_water_yield.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args(model_spec):
    report_filepath = 'awy_report_test.html'
    invest_version = '987.65.0'
    timestamp = time.strftime('%Y-%m-%d %H:%M')
    args_dict = {'suffix': 'test'}
    img_src = 'bAse64eNcoDEdIMagE'
    output_stats_table = '<table class="test__output-stats-table"></table>'
    input_stats_table = '<table class="test__input-stats-table"></table>'
    watershed_table = '<table class="test__watersheds-table"></table>',
    stats_table_note = 'This is a test!'
    raster_group_caption = 'This is another test!'
    img_caption = ['output.tif:Output map.']
    heading = 'Test heading'
    outputs_caption = ['results.tif:Results map.']
    inputs_caption = ['input.tif:Input map.']
    caption = 'figure caption'
    agg_map_source_list = ['/source/file.shp']

    return {
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
        'wyield_img_src': img_src,
        'precip_aet_img_src': img_src,
        'wyield_raster_caption': img_caption,
        'watershed_table': watershed_table,
        'watershed_table_caption': caption,
        'watershed_map_params': None,
        'valuation_results': None,
        'subwatershed_results': None,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'aggregate_map_source_list': agg_map_source_list,
        'model_spec_outputs': model_spec.outputs
    }

class AnnualWaterYieldTemplateTests(unittest.TestCase):
    """Unit tests for AWY template."""

    def test_render_with_subwatersheds(self):
        """Test report rendering with subwatersheds."""

        render_args = _get_render_args(MODEL_SPEC)

        render_args['subwatershed_results'] = {
            'map_json': '{}',
            'map_caption': 'test caption',
            'map_sources': ['/src/path.shp'],
            'table': '<table class="test__subwatersheds-table"></table>',
            'table_caption': 'table caption'
        }
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 8)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_render_with_valuation(self):
        """Test report rendering with valuation_table_path."""

        render_args = _get_render_args(MODEL_SPEC)

        render_args['watershed_map_params'] = {
            'json': '{}',
            'caption': 'test caption',
            'sources': ['/src/path.shp']
        }
        render_args['valuation_results'] = {
            'energy_json': '{}',
            'energy_caption': 'test caption',
            'val_json': '{}',
            'val_caption': 'test caption',
            'source_list': ['/src/path.shp', '/src/path.csv'],
            'title': 'valuation title'
        }
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 7)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_render_without_valuation(self):
        """Test report rendering when no valuation_table_path."""

        render_args = _get_render_args(MODEL_SPEC)
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 7)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')
