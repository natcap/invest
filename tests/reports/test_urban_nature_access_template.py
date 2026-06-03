import unittest
from bs4 import BeautifulSoup

from natcap.invest.urban_nature_access import MODEL_SPEC
from natcap.invest.urban_nature_access.urban_nature_access import \
    RADIUS_OPT_URBAN_NATURE, RADIUS_OPT_POP_GROUP, RADIUS_OPT_UNIFORM
from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/urban_nature_access.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args(model_spec):
    report_filepath = 'urban_nature_access.html'
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
    admin_map_source_list = ['/source/file.shp']
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
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'outputs_img_src': img_src,
        'outputs_caption': outputs_caption,
        'output_accessible_img_src': img_src,
        'output_accessible_raster_caption': test_caption,
        'output_balance_img_src': img_src,
        'output_balance_raster_caption': test_caption,
        'output_balance_percapita_img_src': img_src,
        'output_balance_percapita_raster_caption': test_caption,
        'intermediates_img_src': img_src,
        'intermediates_caption': test_caption,
        'raster_group_caption': test_caption,
        'lulc_pre_caption': lulc_pre_caption,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'stats_table_note': stats_table_note,
        'admin_map_json': vegalite_json,
        'vector_plot_captions': test_caption,
        'admin_boundaries_map_source_list': admin_map_source_list,
        'model_spec_outputs': model_spec.outputs
    }


class UrbanNatureAccessTemplateTests(unittest.TestCase):
    """Unit tests for Urban Nature Access template."""

    def test_render_search_radius_by_lulc_option(self):
        """Test report rendering with LULC option without health_cost_rate"""

        render_args = _get_render_args(MODEL_SPEC)
        render_args['args_dict']['search_radius_mode'] = RADIUS_OPT_URBAN_NATURE
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)
        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 8)
        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

        plots = soup.select('#Admin_Units_Plots')
        self.assertEqual(len(plots), 1)

    def test_render_search_radius_by_pop_option(self):
        """Test report rendering with LULC option with baseline NDVI input"""
        render_args = _get_render_args(MODEL_SPEC)
        render_args['args_dict']['search_radius_mode'] = RADIUS_OPT_POP_GROUP
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)
        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 11)

    def test_render_uniform_search_radius(self):
        """Test report rendering with LULC option with baseline NDVI input"""
        render_args = _get_render_args(MODEL_SPEC)
        render_args['args_dict']['search_radius_mode'] = RADIUS_OPT_UNIFORM
        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)
        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 8)
