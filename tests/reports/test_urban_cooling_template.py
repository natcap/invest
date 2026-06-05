import unittest

from bs4 import BeautifulSoup

from natcap.invest.urban_cooling_model import MODEL_SPEC
from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/urban_cooling.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args(model_spec):

    report_filepath = 'urban_cooling_report_test.html'
    invest_version = '987.65.0'
    timestamp = '1970-01-01'
    args_dict = {'suffix': 'test'}
    img_src = 'bAse64eNcoDEdIMagE'
    mock_json = '{"property": "value"}'
    output_stats_table = '<table class="test__output-stats-table"></table>'
    input_stats_table = '<table class="test__input-stats-table"></table>'
    stats_table_note = 'This is a test!'
    inputs_caption = ['input.tif:Input map.']
    lulc_pre_caption = 'This is a test of the LULC broadcasting system!'
    outputs_caption = ['results.tif:Results map.']
    raster_group_caption = 'This is another test!'
    uhi_table = '<table class="test__uhi-table"></table>'
    bldg_table = '<table class="test__bldg-table"></table>'
    bldg_totals_table = '<table class="test__bldg-table-totals"></table>'
    vector_caption = [
        'field_1:Things. (Units: unitless)', 'field_2:Stuff. (Units: kg)']

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
        'uhi_table': uhi_table,
        'uhi_table_caption': vector_caption,
        'aoi_map_json': mock_json,
        'aoi_map_caption': vector_caption,
        'aoi_map_source_list': ['aoi.shp'],
        'bldg_table': bldg_table,
        'bldg_totals_table': bldg_totals_table,
        'bldg_table_caption': vector_caption,
        'bldg_map_json': mock_json,
        'bldg_map_caption': vector_caption,
        'bldg_map_source_list': ['bldg.shp'],
        'input_raster_heading': 'LULC and Reference Evapotranspiration',
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'lulc_pre_caption': lulc_pre_caption,
        'output_raster_heading': 'Air Temperature',
        'outputs_img_src': img_src,
        'outputs_caption': outputs_caption,
        'biophysical_heading': 'Biophysical Maps',
        'biophysical_img_src': img_src,
        'biophysical_raster_caption': outputs_caption,
        'raster_group_caption': raster_group_caption,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'stats_table_note': stats_table_note,
        'model_spec_outputs': model_spec.outputs,
    }


class UrbanCoolingTemplateTests(unittest.TestCase):
    """Unit tests for Urban Cooling template."""

    def test_render_without_valuation(self):
        """Test report rendering without valuation."""

        render_args = _get_render_args(MODEL_SPEC)
        render_args['bldg_table'] = None

        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        self.assertEqual(len(sections), 9)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_render_with_valuation(self):
        """Test report rendering with energy savings valuation."""

        render_args = _get_render_args(MODEL_SPEC)

        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        sections = soup.find_all(class_='accordion-section')
        # 9 default sections plus 2 for building stats.
        self.assertEqual(len(sections), 11)

        self.assertEqual(
            soup.h1.string, f'InVEST Results: {MODEL_SPEC.model_title}')
