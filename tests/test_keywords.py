import unittest

from natcap.invest import keywords


class TestKeywords(unittest.TestCase):
    """Test and validate characteristics of the keywords vocabulary."""

    def test_unique(self):
        """Test values, aliases, and uuids are all unique."""
        values = set()  # values and aliases together must be unique
        uuids = set()
        for keyword in keywords.to_list():
            self.assertNotIn(
                keyword.value, values,
                msg=f'\n duplicate value {keyword.value} on {keyword}')
            values.add(keyword.value)
            for alias in keyword.aliases:
                self.assertNotIn(
                    alias, values,
                    msg=f'\n duplicate alias {alias} on {keyword}')
                values.add(alias)
            if hasattr(keyword, 'uuid'):
                self.assertNotIn(
                    keyword.uuid, uuids,
                    msg=f'\n duplicate uuid on {keyword}')
                uuids.add(keyword.uuid)
