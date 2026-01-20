import unittest

from natcap.invest.reports import jinja_env


class JinjaTemplateUnitTests(unittest.TestCase):
    """Unit tests for partial templates."""

    def test_list_metadata(self):
        """Test list_metadata macro."""
        from natcap.invest.coastal_vulnerability import MODEL_SPEC

        template_str = \
            """
            <html>
                {% from 'metadata.html' import list_metadata %}
                <div>{{ list_metadata(model_spec_outputs) }}</div>
            </html>
            """
        template = jinja_env.from_string(template_str)
        html = template.render(model_spec_outputs=MODEL_SPEC.outputs)
        for output in MODEL_SPEC.outputs:
            self.assertIn(output.path, html)

    def test_caption_with_text_string_and_list(self):
        """Test caption macro with string and list."""

        template_str = \
            """
            <html>
                {% from 'caption.html' import caption %}
                <div>{{ caption(text, source_list) }}</div>
            </html>
            """
        text = 'description'
        source_list = ['/foo/bar']
        template = jinja_env.from_string(template_str)
        html = template.render(
            text=text,
            source_list=source_list)
        self.assertIn(text, html)
        for source in source_list:
            self.assertIn(source, html)

    def test_caption_with_text_list(self):
        """Test caption macro with list of text."""

        template_str = \
            """
            <html>
                {% from 'caption.html' import caption %}
                <div>{{ caption(text) }}</div>
            </html>
            """
        text_list = ['description', 'paragraph']
        template = jinja_env.from_string(template_str)
        html = template.render(text=text_list)
        for text in text_list:
            self.assertIn(text, html)
        self.assertNotIn('Sources', html)

    def test_caption_with_definition_list_option(self):
        """Test caption macro with definition_list=True."""

        template_str = (
            """
            <html>
                {% from 'caption.html' import caption %}
                {{ caption(text, definition_list=True) }}
            </html>
            """
        )
        definitions = [
            ('Simile',
             ('A comparison that uses like or as, e.g., '
              'life is like a box of chocolates.')),
            ('Analogy',
             ('A binary relationship defined in terms of another binary '
              'relationship. Typically follows the form A is to B as '
              'C is to D, or, in shorthand: A : B :: C : D.')),
        ]
        text = [f'{term}:{definition}' for (term, definition) in definitions]

        template = jinja_env.from_string(template_str)
        html = template.render(text=text)

        self.assertIn('<dl>', html)
        for (term, definition) in definitions:
            self.assertIn(f'<dt>{term}</dt>', html)
            self.assertIn(f'<dd>{definition}</dd>', html)
        self.assertIn('</dl>', html)
        self.assertNotIn('<p>', html)

    def test_caption_with_pre_caption_option(self):
        """Test caption macro with pre_caption=True."""
        template_str = (
            """
            <html>
                {% from 'caption.html' import caption %}
                {{ caption(text, pre_caption=True) }}
            </html>
            """
        )
        text = ('This is meant to appear above an image (instead of below) '
                'and should be styled accordingly.')

        template = jinja_env.from_string(template_str)
        html = template.render(text=text)

        self.assertIn('<div class="caption pre-caption">', html)
        self.assertIn(text, html)

    def test_raster_plot_img(self):
        """Test raster_plot_img macro."""

        template_str = (
            """
            <html>
                {% from 'raster-plot-img.html' import raster_plot_img %}
                {{ raster_plot_img(img_src, img_name) }}
            </html>
            """
        )
        img_src = 'PiNeAPpLeUNdeRtHeSEa'
        img_name = 'Bathymetry Maps'

        template = jinja_env.from_string(template_str)
        html = template.render(img_src=img_src, img_name=img_name)
        # Discard newlines and tabs.
        html = ' '.join(html.split())

        self.assertIn((f'<img src="data:image/png;base64,{img_src}" '
                       f'alt="Raster plots: {img_name}" />'), html)

    def test_args_table(self):
        """Test args_table macro."""

        template_str = (
            """
            <html>
                {% from 'args-table.html' import args_table with context %}
                {{ args_table() }}
            </html>
            """
        )
        args_dict = {
            'Fruit': 'Orange',
            'Vegetable': 'Okra',
            'Herb': 'Oregano',
        }

        template = jinja_env.from_string(template_str)
        html = template.render(args_dict=args_dict)

        self.assertIn('<table>', html)
        self.assertIn('<th>Name</th>', html)
        self.assertIn('<th>Value</th>', html)
        for (key, value) in args_dict.items():
            self.assertIn(f'<td>{key}</td>', html)
            self.assertIn(f'<td>{value}</td>', html)

    def test_wide_table(self):
        """Test wide_table macro."""

        template_str = (
            """
            <html>
                {% from 'wide-table.html' import wide_table %}
                {{ wide_table(table | safe) }}
            </html>
            """
        )
        table = '<table class="test__table"></table>'

        template = jinja_env.from_string(template_str)
        html = template.render(table=table)
        # Discard newlines and tabs.
        html = ' '.join(html.split())

        self.assertIn(
            '<div class="wide-table-wrapper" style="font-size: 0.875rem;" >',
            html)
        self.assertIn(table, html)

    def test_wide_table_with_custom_font_size(self):
        """Test wide_table macro with custom font size."""

        template_str = (
            """
            <html>
                {% from 'wide-table.html' import wide_table %}
                {{ wide_table(table | safe, font_size_px) }}
            </html>
            """
        )
        table = '<table class="test__table"></table>'
        font_size_px = 20

        template = jinja_env.from_string(template_str)
        html = template.render(table=table, font_size_px=font_size_px)
        # Discard newlines and tabs.
        html = ' '.join(html.split())

        self.assertIn(
            '<div class="wide-table-wrapper" style="font-size: 1.25rem;" >',
            html)
        self.assertIn(table, html)

    def test_wide_table_minimum_font_size(self):
        """Test wide_table macro with custom font size that is too small."""

        template_str = (
            """
            <html>
                {% from 'wide-table.html' import wide_table %}
                {{ wide_table(table | safe, font_size_px) }}
            </html>
            """
        )
        table = '<table class="test__table"></table>'
        font_size_px = 11

        template = jinja_env.from_string(template_str)
        html = template.render(table=table, font_size_px=font_size_px)
        # Discard newlines and tabs.
        html = ' '.join(html.split())

        # Font size should fall back to macro-defined minimum: 12px = 0.75rem.
        self.assertIn(
            '<div class="wide-table-wrapper" style="font-size: 0.75rem;" >',
            html)
        self.assertIn(table, html)
