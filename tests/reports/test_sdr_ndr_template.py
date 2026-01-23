import sys
import unittest

import lxml.html

from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('models/sdr-ndr-report.html')


def _get_render_args(model_spec):
    timestamp = '1970-01-01'
    args_dict = {'suffix': 'test'}
    img_src = 'bAse64eNcoDEdIMagE'
    intermediate_outputs_heading = 'Intermediate Outputs'
    ws_vector_table = '<table class="test__results-table></table>'
    ws_vector_totals_table = '<table class="test__totals-table"></table>'
    output_stats_table = '<table class="test__output-stats-table></table>'
    input_stats_table = '<table class="test__input-stats-table></table>'
    stats_table_note = 'This is a test!'
    inputs_caption = ['input.tif:Input map.']
    outputs_caption = ['results.tif:Results map.']
    intermediate_outputs_caption = ['stream.tif:Stream network map.']
    raster_group_caption = 'This is another test!'

    return {
        'report_script': __file__,
        'model_id': model_spec.model_id,
        'model_name': model_spec.model_title,
        'userguide_page': model_spec.userguide,
        'timestamp': timestamp,
        'args_dict': args_dict,
        'inputs_img_src': img_src,
        'inputs_caption': inputs_caption,
        'outputs_img_src': img_src,
        'outputs_caption': outputs_caption,
        'intermediate_outputs_heading': intermediate_outputs_heading,
        'intermediate_outputs_img_src': img_src,
        'intermediate_outputs_caption': intermediate_outputs_caption,
        'raster_group_caption': raster_group_caption,
        'ws_vector_table': ws_vector_table,
        'ws_vector_totals_table': ws_vector_totals_table,
        'output_raster_stats_table': output_stats_table,
        'input_raster_stats_table': input_stats_table,
        'stats_table_note': stats_table_note,
        'model_spec_outputs': model_spec.outputs,
    }


@unittest.skipIf(sys.platform.startswith("win"), "segfaults on Windows")
class SDR_NDR_TemplateTests(unittest.TestCase):
    """Unit tests for SDR/NDR template."""

    def test_render_sdr(self):
        """Make sure the template renders for SDR without error."""

        from natcap.invest.sdr.sdr import MODEL_SPEC

        html = TEMPLATE.render(_get_render_args(MODEL_SPEC))

        root = lxml.html.document_fromstring(html)

        sections = root.find_class('accordion-section')
        self.assertEqual(len(sections), 8)

        h1 = root.find('.//h1')
        self.assertEqual(h1.text, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_render_ndr(self):
        """Make sure the template renders for NDR without error."""

        from natcap.invest.ndr.ndr import MODEL_SPEC

        html = TEMPLATE.render(_get_render_args(MODEL_SPEC))
        root = lxml.html.document_fromstring(html)

        sections = root.find_class('accordion-section')
        self.assertEqual(len(sections), 8)

        h1 = root.find('.//h1')
        self.assertEqual(h1.text, f'InVEST Results: {MODEL_SPEC.model_title}')

    def test_watershed_results_totals(self):
        """Totals should be rendered when passed to the render function."""

        ws_vector_table = '<table class="test__results-table></table>'
        ws_vector_totals_table = '<table class="test__totals-table"></table>'

        # Note that args_dict=None, which isn't exactly realistic,
        # but omitting the args table from the HTML is crucial for this test.
        html = TEMPLATE.render(
            report_script=__file__,
            model_id='',
            model_name='',
            userguide_page='',
            timestamp='',
            args_dict=None,
            inputs_img_src='',
            inputs_caption=[],
            outputs_img_src='',
            outputs_caption=[],
            intermediate_outputs_heading='',
            intermediate_outputs_img_src='',
            intermediate_outputs_caption=[],
            raster_group_caption='',
            ws_vector_table=ws_vector_table,
            ws_vector_totals_table=ws_vector_totals_table,
            output_raster_stats_table='',
            input_raster_stats_table='',
            stats_table_note='',
            model_spec_outputs=[],
        )

        root = lxml.html.document_fromstring(html)

        totals_table = root.find_class('test__totals-table')
        self.assertTrue(len(totals_table) == 1)

        # Ideally, this test would select only the tables in the 'Results by
        # Watershed' section, but doing so seems impossible without cumbersome
        # tree traversal that would likely result in a brittle test.
        tables = root.findall('.//table')
        self.assertEqual(len(tables), 2)

    def test_watershed_results_without_totals(self):
        """Totals should be not be rendered when there are none to render."""

        ws_vector_table = '<table class="test__results-table></table>'
        ws_vector_totals_table = None

        # Note that args_dict=None, which isn't exactly realistic,
        # but omitting the args table from the HTML is crucial for this test.
        html = TEMPLATE.render(
            report_script=__file__,
            model_id='',
            model_name='',
            userguide_page='',
            timestamp='',
            args_dict=None,
            inputs_img_src='',
            inputs_caption=[],
            outputs_img_src='',
            outputs_caption=[],
            intermediate_outputs_heading='',
            intermediate_outputs_img_src='',
            intermediate_outputs_caption=[],
            raster_group_caption='',
            ws_vector_table=ws_vector_table,
            ws_vector_totals_table=ws_vector_totals_table,
            output_raster_stats_table='',
            input_raster_stats_table='',
            stats_table_note='',
            model_spec_outputs=[],
        )

        root = lxml.html.document_fromstring(html)

        # Ideally, this test would select only the tables in the 'Results by
        # Watershed' section, but doing so seems impossible without cumbersome
        # tree traversal that would likely result in a brittle test.
        tables = root.findall('.//table')
        self.assertEqual(len(tables), 1)
