import re
import unittest

from bs4 import BeautifulSoup

from natcap.invest.reports import jinja_env

TEMPLATE = jinja_env.get_template('base.html')

BSOUP_HTML_PARSER = 'html.parser'


def _get_render_args():
    return {
        'locale': 'en',
        'model_name': 'Test Model',
        'model_description': 'This is a test of the base template.',
        'userguide_page': 'testmodel.html',
        'report_script': 'natcap.invest.test.reporter',
        'invest_version': '987.65.0',
        'report_filepath': 'test_report.html',
        'timestamp': '1970-01-01',
    }


class BaseTemplateTests(unittest.TestCase):
    """Unit tests for base template."""

    def test_base_template_content(self):
        """Test that rendered HTML includes key details passed to template."""

        render_args = _get_render_args()

        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        self.assertIn(render_args['model_name'], soup.title.string)
        self.assertIn(render_args['model_name'], soup.h1.string)

        self.assertIsNotNone(soup.main.find(
            string=render_args['model_description']))

        ug_link = soup.find('a')
        self.assertIn(render_args['userguide_page'], ug_link['href'])

        for arg in ['report_script', 'invest_version',
                    'report_filepath', 'timestamp']:
            # self.assertIn(render_args[arg], soup.footer.string)
            self.assertIsNotNone(
                soup.footer.find(string=re.compile(render_args[arg])))

    def test_locale(self):
        """Test that locale is set on ``lang`` attribute and in UG URL."""

        render_args = _get_render_args()
        render_args['locale'] = 'es'

        html = TEMPLATE.render(render_args)
        soup = BeautifulSoup(html, BSOUP_HTML_PARSER)

        self.assertEqual(soup.html['lang'], 'es')

        ug_link = soup.find('a')
        self.assertIn('/es/', ug_link['href'])
