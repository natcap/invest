# -*- coding: utf-8 -*-
"""Tests for Main Model."""
import unittest
import os
import shutil
import pprint

import numpy as np
import gdal
from pygeoprocessing import geoprocessing as geoprocess
from pygeoprocessing import testing as geotest

from natcap.invest.coastal_blue_carbon import coastal_blue_carbon as cbc

pp = pprint.PrettyPrinter(indent=4)


class TestFunctions(unittest.TestCase):

    """Function-Level Tests."""

    def setUp(self):
        pass

    def test_func(self):
        pass

    def tearDown(self):
        pass


class TestModel(unittest.TestCase):

    """Model-Level Tests."""

    def setUp(self):
        pass

    def test_func(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
