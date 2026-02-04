import unittest

from bs4 import BeautifulSoup

from natcap.invest.carbon import MODEL_SPEC
from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/carbon.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args(model_spec):
    model_description = 'This is a description of the carbon model.'
    timestamp = '1970-01-01'
    args_dict = {'suffix': 'test'}
    img_src = 'bAse64eNcoDEdIMagE'
    output_stats_table = '<table class="test__output-stats-table"></table>'
    input_stats_table = '<table class="test__input-stats-table"></table>'
    stats_table_note = 'This is a test!'
    inputs_caption = ['input.tif:Input map.']
    lulc_pre_caption = 'This is a test of the LULC broadcasting system!'
    outputs_caption = ['results.tif:Results map.']
    intermediate_raster_sections = []
    raster_group_caption = 'This is another test!'
    agg_results_table = '<table class="test__agg-results-table"></table>'

    return {
        'report_script': __file__,
        'model_id': model_spec.model_id,
        'model_name': model_spec.model_title,
        'model_description': model_description,
        'userguide_page': model_spec.userguide,
        'timestamp': timestamp,
        'args_dict': args_dict,
        'agg_results_table': agg_results_table,
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'lulc_pre_caption': lulc_pre_caption,
        'outputs_img_src': img_src,
        'outputs_caption': outputs_caption,
        'intermediate_raster_sections': intermediate_raster_sections,
        'raster_group_caption': raster_group_caption,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'stats_table_note': stats_table_note,
        'model_spec_outputs': model_spec.outputs,
    }


def _mock_intermediate_output_sections(num_sections):
    return [
        {
            'heading': f'Intermediate Outputs {i + 1}',
            'img_src': 'bAse64eNcoDEdIMagE',
            'caption': ['map1.tif:Map of baseline-scenario carbon values.',
                        'map2.tif:Map of alternate-scenario carbon values.'],
        } for i in range(num_sections)
    ]


class CarbonTemplateTests(unittest.TestCase):
    """Unit tests for Carbon template."""

    def test_render_without_alt_scenario(self):
        """Test report rendering without alternate scenario."""

        render_args = _get_render_args(MODEL_SPEC)
        render_args['intermediate_raster_sections'] = (
            _mock_intermediate_output_sections(1))

        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        # 7 default sections plus 1 section for intermediate outputs.
        self.assertEqual(len(sections), 8)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_render_with_alt_scenario(self):
        """Test report rendering with alternate scenario."""

        render_args = _get_render_args(MODEL_SPEC)
        render_args['intermediate_raster_sections'] = (
            _mock_intermediate_output_sections(4))

        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        # 7 default sections plus 4 sections for intermediate outputs.
        self.assertEqual(len(sections), 11)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')
