import os
import shutil
import tempfile
import unittest

import matplotlib
import matplotlib.testing.compare
from matplotlib.testing import set_font_settings_for_testing
from matplotlib.testing.exceptions import ImageComparisonFailure
import numpy
import pygeoprocessing
from osgeo import osr

from natcap.invest.reports import raster_utils
from natcap.invest.reports import MATPLOTLIB_PARAMS

projection = osr.SpatialReference()
projection.ImportFromEPSG(3857)
PROJ_WKT = projection.ExportToWkt()

REFS_DIR = os.path.join('tests', 'reports', 'refs')


def setUpModule():
    # There can be platform-specific differences in text and fonts
    # that make image comparisons fail. These settings can standardize
    # those across platforms.
    set_font_settings_for_testing()
    matplotlib.rcParams.update(MATPLOTLIB_PARAMS)


def tearDownModule():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def save_figure(fig, filepath):
    # Overriding the default dpi to reduce file size since
    # these figures are under source control.
    fig.savefig(filepath, **raster_utils.MPL_SAVE_FIG_KWARGS, dpi=50)


def make_simple_raster(target_filepath, shape):
    array = numpy.linspace(0, 1, num=numpy.multiply(*shape)).reshape(*shape)
    pygeoprocessing.numpy_array_to_raster(
        array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
        projection_wkt=PROJ_WKT, target_path=target_filepath)


def make_simple_nominal_raster(target_filepath, shape):
    array = numpy.arange(0, numpy.multiply(*shape), 1).reshape(*shape)
    pygeoprocessing.numpy_array_to_raster(
        array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
        projection_wkt=PROJ_WKT, target_path=target_filepath)


def compare_snapshots(reference, actual):
    ref, ext = os.path.splitext(reference)
    new_reference = f'{ref}_fail{ext}'
    try:
        comparison = matplotlib.testing.compare.compare_images(
            reference, actual, 1e-6)
    except OSError:
        shutil.copy(actual, reference)
        raise OSError(
            f'Reference image did not exist. '
            f'Now it does. Re-run this test and also run '
            f'`git add {reference}`')
    except ImageComparisonFailure as error:
        # Raised if the images are different sizes, for example.
        shutil.copy(actual, new_reference)
        raise AssertionError(
            str(error), f'actual image saved to {new_reference}')

    # If comparison is not None, then the images are not identical.
    if comparison is None:
        return
    shutil.copy(actual, new_reference)
    raise AssertionError(comparison, f'actual image saved to {new_reference}')


class RasterPlotLayoutTests(unittest.TestCase):
    """Unit tests for matplotlib utilities."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'foo.tif'),
            'continuous', 'linear')

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_plot_raster_list_square_aoi(self):
        """Test figure has 1 row and 3 columns."""
        figname = 'plot_raster_list_square_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*3
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_wide_aoi(self):
        """Test figure has 2 rows and 2 columns."""
        figname = 'plot_raster_list_wide_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 8)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*3
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_tall_aoi(self):
        """Test figure has 1 rows and 3 columns."""
        figname = 'plot_raster_list_tall_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (8, 2)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*3
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_single(self):
        """Test figure has 1 plot."""
        figname = 'plot_raster_list_single.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)


class RasterPlotConfig(unittest.TestCase):
    """Unit tests for datatype and transform options."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_plot_raster_list_different_datatypes(self):
        """Test figure plots for different datatypes."""
        figname = 'plot_raster_list_datatypes.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 8)
        continuous_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            'continuous')
        make_simple_raster(continuous_raster.raster_path, shape)

        binary_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'binary.tif'),
            'binary')
        binary_array = numpy.zeros(shape=shape)
        binary_array[0] = 1
        pygeoprocessing.numpy_array_to_raster(
            binary_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=binary_raster.raster_path)

        divergent_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'divergent.tif'),
            'divergent')
        divergent_array = numpy.linspace(
            -1, 1, num=numpy.multiply(*shape)).reshape(*shape)
        pygeoprocessing.numpy_array_to_raster(
            divergent_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=divergent_raster.raster_path)

        nominal_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'nominal.tif'),
            'nominal')
        make_simple_nominal_raster(nominal_raster.raster_path, shape)

        config_list = [
            continuous_raster, nominal_raster, binary_raster, divergent_raster]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_different_transforms(self):
        """Test figure plots for different transforms."""
        figname = 'plot_raster_list_transforms.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 8)
        continuous_raster_linear = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            'continuous', 'linear')
        continuous_raster_log = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            'continuous', 'log')
        make_simple_raster(continuous_raster_linear.raster_path, shape)

        divergent_raster_log = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'divergent.tif'),
            'divergent', 'log')
        divergent_array = numpy.linspace(
            -1, 1, num=numpy.multiply(*shape)).reshape(*shape)
        pygeoprocessing.numpy_array_to_raster(
            divergent_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=divergent_raster_log.raster_path)

        config_list = [
            continuous_raster_linear, continuous_raster_log,
            divergent_raster_log]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)


class RasterPlotLegends(unittest.TestCase):
    """Unit tests for legend placement on nominal rasters."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'foo.tif'),
            'nominal', 'linear')
        
    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_plot_raster_list_tall_nominal(self):
        """Test legend is single column on right side."""
        figname = 'plot_raster_list_tall_nominal.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        make_simple_nominal_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_wide_nominal(self):
        """Test legend is multi-column below plot."""
        figname = 'plot_raster_list_wide_nominal.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 8)
        make_simple_nominal_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)


class RasterPlotFacets(unittest.TestCase):
    """Unit tests for plotting multiple rasters on the same colorscale."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_plot_raster_facets(self):
        """Test rasters share a common colorscale."""
        figname = 'plot_raster_facets.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        a_raster_filepath = os.path.join(self.workspace_dir, 'a.tif')
        b_raster_filepath = os.path.join(self.workspace_dir, 'b.tif')
        a_array = numpy.linspace(0, 1, num=numpy.multiply(*shape)).reshape(*shape)
        b_array = numpy.linspace(1, 2, num=numpy.multiply(*shape)).reshape(*shape)
        pygeoprocessing.numpy_array_to_raster(
            a_array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
            projection_wkt=PROJ_WKT, target_path=a_raster_filepath)
        pygeoprocessing.numpy_array_to_raster(
            b_array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
            projection_wkt=PROJ_WKT, target_path=b_raster_filepath)
        
        fig = raster_utils.plot_raster_facets(
            [a_raster_filepath, b_raster_filepath], 'continuous')
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)
