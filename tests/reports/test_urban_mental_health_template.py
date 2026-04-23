import unittest
from bs4 import BeautifulSoup

from natcap.invest.urban_mental_health import MODEL_SPEC
from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/urban_mental_health.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args(model_spec):
    locale = 'en'
    report_filepath = 'carbon_report_test.html'
    invest_version = '987.65.0'
    timestamp = '1970-01-01'
    args_dict = {'suffix': 'test', 'lulc_base': ''}
    img_src = 'bAse64eNcoDEdIMagE'
    output_stats_table = '<table class="test__output-stats-table"></table>'
    input_stats_table = '<table class="test__input-stats-table"></table>'
    stats_table_note = 'This is a test!'
    inputs_caption = ['input.tif:Input map.']
    lulc_pre_caption = 'This is a test of the LULC broadcasting system!'
    outputs_caption = ['results.tif:Results map.']
    vegalite_json = '{}'
    test_caption = 'This is another test!'
    agg_results_table = '<table class="test__agg-results-table"></table>'
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
        'agg_results_table': agg_results_table,
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'outputs_img_src': img_src,
        'outputs_caption': outputs_caption,
        'intermediate_baseline_img_src': img_src,
        'intermediate_baseline_img_caption': test_caption,
        'intermediate_delta_ndvi_img_src': img_src,
        'intermediate_delta_ndvi_img_caption': test_caption,
        'intermediate_reclass_lulc_img_src': img_src,
        'intermediate_reclass_lulc_img_caption': test_caption,
        'raster_group_caption': test_caption,
        'lulc_pre_caption': lulc_pre_caption,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'stats_table_note': stats_table_note,
        'cases_map_json': vegalite_json,
        'cost_map_json': vegalite_json,
        'cases_map_caption': test_caption,
        'cost_map_caption': test_caption,
        'aggregate_map_source_list': agg_map_source_list,
        'model_spec_outputs': model_spec.outputs,
    }


class UrbanMentalHealthTemplateTests(unittest.TestCase):
    """Unit tests for Urban Mental Health template."""

    def test_render_lulc_option_without_cost(self):
        """Test report rendering with LULC option without health_cost_rate"""

        render_args = _get_render_args(MODEL_SPEC)
        render_args['args_dict']['model_option'] = 'lulc'
        render_args['args_dict']['lulc_base'] = 'lulc_base.tif'
        # if no health_cost_rate, no cost_map produced
        render_args['cost_map_json'] = None
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)
        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 11)
        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

        # should only have plot for preventable cases, not preventable cost
        vega_plots = soup.find_all(class_='vega-fit-horizontal')
        self.assertEqual(len(vega_plots), 1)

    def test_render_lulc_option_with_ndvi(self):
        """Test report rendering with LULC option with baseline NDVI input"""
        render_args = _get_render_args(MODEL_SPEC)
        render_args['args_dict']['model_option'] = 'lulc'
        render_args['args_dict']['lulc_base'] = 'lulc_base.tif'
        render_args['args_dict']['ndvi_base'] = 'ndvi_base.tif'
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)
        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 11)

    def test_render_ndvi_option_with_cost(self):
        """Test report rendering with NDVI model option with cost"""
        render_args = _get_render_args(MODEL_SPEC)
        render_args['args_dict']['model_option'] = 'ndvi'
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)
        sections = soup.find_all(class_='accordion-section')
        # Does not have reclassified LULC to NDVI section
        self.assertEqual(len(sections), 10)

        # should have plots for both preventable cases and cost
        vega_plots = soup.find_all(class_='vega-fit-horizontal')
        self.assertEqual(len(vega_plots), 2)
