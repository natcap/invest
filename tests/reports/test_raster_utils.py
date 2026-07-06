import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
import geometamaker
import matplotlib
import matplotlib.testing.compare
from matplotlib.testing import set_font_settings_for_testing
from matplotlib.testing.exceptions import ImageComparisonFailure
import numpy
from osgeo import gdal, osr
import pandas
import pygeoprocessing
from pydantic import ValidationError

from natcap.invest import spec
from natcap.invest.reports import MATPLOTLIB_PARAMS, raster_utils
from natcap.invest.reports.raster_utils import RasterDatatype
from natcap.invest.reports.raster_utils import RasterTransform
from natcap.invest.reports.raster_utils import RasterPlotConfig
from natcap.invest.reports.raster_utils import SpecialValueConfig

projection = osr.SpatialReference()
projection.ImportFromEPSG(3857)
PROJ_WKT = projection.ExportToWkt()

REFS_DIR = os.path.join('data', 'invest-test-data', 'reports', 'snapshots')
MPL_VERSION = tuple(int(_) for _ in matplotlib.__version__.split('.'))


def setUpModule():
    # There can be platform-specific differences in text and fonts
    # that make image comparisons fail. These settings can standardize
    # those across platforms.
    set_font_settings_for_testing()
    matplotlib.rcParams.update(MATPLOTLIB_PARAMS)


def tearDownModule():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def save_figure(fig, filepath):
    fig.savefig(filepath, **raster_utils.MPL_SAVE_FIG_KWARGS)


def make_simple_raster(target_filepath, shape):
    array = numpy.linspace(0, 1, num=numpy.multiply(*shape)).reshape(*shape)
    pygeoprocessing.numpy_array_to_raster(
        array, target_nodata=None, pixel_size=(1, -1), origin=(0, 0),
        projection_wkt=PROJ_WKT, target_path=target_filepath)


def make_simple_nominal_raster(target_filepath, shape):
    array = numpy.arange(0, numpy.multiply(*shape), 1).reshape(*shape)
    pygeoprocessing.numpy_array_to_raster(
        array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
        projection_wkt=PROJ_WKT, target_path=target_filepath)


def make_nominal_raster_with_distinct_counts(target_filepath, num_vals):
    """Create raster with one 1, two 2s, three 3s, ..., num_vals num_vals."""
    if not num_vals % 2 == 0:
        raise ValueError('num_vals must be an even number.')
    val = numpy.arange(1, num_vals + 1, 1)
    array = numpy.repeat(val, val).reshape(num_vals + 1, num_vals // 2)
    pygeoprocessing.numpy_array_to_raster(
        array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
        projection_wkt=PROJ_WKT, target_path=target_filepath)


def add_raster_attribute_table(target_filepath, value_name='value',
                               count_name='count', extra_cols=[]):
    raster = gdal.OpenEx(target_filepath, gdal.OF_UPDATE)
    band = raster.GetRasterBand(1)
    rat = gdal.RasterAttributeTable()
    rat.CreateColumn(value_name, gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn(count_name, gdal.GFT_Integer, gdal.GFU_PixelCount)
    for name in extra_cols:
        rat.CreateColumn(name, gdal.GFT_String, gdal.GFU_Generic)

    array = band.ReadAsArray()
    values, counts = numpy.unique(array, return_counts=True)
    for i in range(len(values)):
        rat.SetValueAsInt(i, 0, int(values[i]))
        rat.SetValueAsInt(i, 1, int(counts[i]))
        for idx, name in enumerate(extra_cols):
            rat.SetValueAsString(i, 2 + idx, 'foo')
    band.SetDefaultRAT(rat)
    band = raster = None


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


@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class RasterPlotLayoutTests(unittest.TestCase):
    """Snapshot tests for matplotlib figure layouts."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'foo.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'))

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
        """Test figure has 1 row and 3 columns."""
        figname = 'plot_raster_list_tall_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (8, 2)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*3
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_extra_wide_aoi(self):
        """Test figure has 2 rows and 1 column."""
        figname = 'plot_raster_list_extra_wide_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 10)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*2
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

    def test_plot_raster_list_units_subtitle_padding(self):
        """Test subtitle offset for plots of rasters with and without units"""
        shape = (4, 4)
        make_simple_raster(self.raster_config.raster_path, shape)

        # check that no subtitle offset because no units
        fig = raster_utils.plot_raster_list([self.raster_config])
        raster_axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
        for ax in raster_axes:
            ylim = ax.get_ylim()
            expected_ylim = (3.5, -0.5)
            self.assertEqual(ylim, expected_ylim)

        # Test plot ylims if one raster has units and other does not
        raster_path_with_units = os.path.join(self.workspace_dir, 'foo.tif')
        make_simple_raster(raster_path_with_units, shape)
        # units determined by checking metadata
        metadata = geometamaker.describe(raster_path_with_units)
        metadata.set_band_description(1, units="meters")
        metadata.write()
        raster_config_with_units = RasterPlotConfig(
            raster_path_with_units,
            RasterDatatype.continuous,
            spec.Output(id='foo', units="meter"))

        config_list = [self.raster_config, raster_config_with_units]
        fig = raster_utils.plot_raster_list(config_list)
        raster_axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
        for ax in raster_axes:
            bottom, top = ax.get_ylim()
            expected_ylim = (shape[0], -0.1 * shape[0])
            self.assertEqual((bottom, top), expected_ylim)


@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class RasterPlotDatatypeAndTransformTests(unittest.TestCase):
    """Snapshot tests for datatype and transform options."""

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
        continuous_raster = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'))
        make_simple_raster(continuous_raster.raster_path, shape)

        binary_raster = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'binary.tif'),
            RasterDatatype.binary,
            spec.Output(id='foo'))
        binary_array = numpy.zeros(shape=shape)
        binary_array[0] = 1
        pygeoprocessing.numpy_array_to_raster(
            binary_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=binary_raster.raster_path)

        divergent_raster = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'divergent.tif'),
            RasterDatatype.divergent,
            spec.Output(id='foo'))
        divergent_array = numpy.linspace(
            -1, 1, num=numpy.multiply(*shape)).reshape(*shape)
        pygeoprocessing.numpy_array_to_raster(
            divergent_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=divergent_raster.raster_path)

        nominal_raster = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'nominal.tif'),
            RasterDatatype.nominal,
            spec.Input(id='foo'))
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
        continuous_raster_linear = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'),
            transform=RasterTransform.linear)
        continuous_raster_log = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'),
            transform=RasterTransform.log)
        make_simple_raster(continuous_raster_linear.raster_path, shape)

        divergent_raster_log = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'divergent.tif'),
            RasterDatatype.divergent,
            spec.Output(id='foo'),
            transform=RasterTransform.log)
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


@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class RasterPlotLegendTests(unittest.TestCase):
    """Snapshot tests for legend placement on nominal rasters."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'foo.tif'),
            RasterDatatype.nominal,
            spec.Output(id='foo'))

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

    def test_plot_raster_list_tall_nominal_many_classes(self):
        """Test legend is multi-column on right side and has enough colors."""
        figname = 'plot_raster_list_tall_nominal_many_classes.png'
        reference = os.path.join(REFS_DIR, figname)
        # More than 20 values will require a distinctipy-generated palette.
        # More than 30 values will result in a multi-column legend.
        num_unique_vals = 40
        # We need a unique number of pixels for each unique value to ensure the
        # sort order in the legend is deterministic (otherwise we see variance
        # across platforms).
        make_nominal_raster_with_distinct_counts(
            self.raster_config.raster_path, num_unique_vals)

        config_list = [self.raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)


@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class RasterPlotFacetsTests(unittest.TestCase):
    """Snapshot tests for plotting multiple rasters on the same colorscale."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    @patch('natcap.invest.reports.raster_utils._get_raster_units')
    def create_small_plots_grid(workspace_dir, shape, mock_get_raster_units,
                                supertitle=None):
        raster_paths = [os.path.join(workspace_dir, f'{s}.tif')
                        for s in ['a', 'b', 'c', 'd']]
        arrays = [numpy.linspace(
            i, i+1, num=numpy.multiply(*shape)).reshape(*shape) for i in range(4)]
        for raster_path, array in zip(raster_paths, arrays):
            pygeoprocessing.numpy_array_to_raster(
                array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
                projection_wkt=PROJ_WKT, target_path=raster_path)

        mock_get_raster_units.return_value = 'flux capacitrons'
        return raster_utils.plot_raster_facets(
            raster_paths, RasterDatatype.continuous, small_plots=True,
            supertitle=supertitle)

    def test_plot_raster_facets(self):
        """Test rasters share a common colorscale."""
        figname = 'plot_raster_facets.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        a_raster_filepath = os.path.join(self.workspace_dir, 'a.tif')
        b_raster_filepath = os.path.join(self.workspace_dir, 'b.tif')
        a_array = numpy.linspace(
            0, 1, num=numpy.multiply(*shape)).reshape(*shape)
        b_array = numpy.linspace(
            1, 2, num=numpy.multiply(*shape)).reshape(*shape)
        pygeoprocessing.numpy_array_to_raster(
            a_array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
            projection_wkt=PROJ_WKT, target_path=a_raster_filepath)
        pygeoprocessing.numpy_array_to_raster(
            b_array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
            projection_wkt=PROJ_WKT, target_path=b_raster_filepath)

        fig = raster_utils.plot_raster_facets(
            [a_raster_filepath, b_raster_filepath], RasterDatatype.continuous)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_facets_small_plots(self):
        """Test small plots: standard AOI width should have 4 columns."""
        figname = 'plot_raster_facets_small_plots.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        fig = self.create_small_plots_grid(self.workspace_dir, shape)

        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_facets_small_plots_wide_aoi(self):
        """Test small plots: wide AOI width should have 3 columns."""
        figname = 'plot_raster_facets_small_plots_wide_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (6, 12)
        fig = self.create_small_plots_grid(self.workspace_dir, shape)

        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_facets_small_plots_supertitle(self):
        """Test small plots with optional supertitle."""
        figname = 'plot_raster_facets_small_plots_supertitle.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 10)
        fig = self.create_small_plots_grid(self.workspace_dir, shape,
                                           supertitle="Custom Title")

        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)


@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class RasterPlotTitleTests(unittest.TestCase):
    """Snapshot tests for plotting rasters with various titles."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = RasterPlotConfig(
            os.path.join(
                self.workspace_dir,
                'raster_with_extra_long_filename-for_testing_text_wrapping_in_images_for_reports.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'))

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_plot_raster_list_long_title_extra_wide_1_col(self):
        """Test title wraps appropriately for 1-col layout, extra-wide AOI."""
        figname = 'plot_raster_list_long_title_extra_wide_1_col.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (2, 10)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_long_title_wide_2_col(self):
        """Test title wraps appropriately for 2-col layout, wide AOI."""
        figname = 'plot_raster_list_long_title_wide_2_col.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (3, 8)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*2
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_long_title_square_3_col(self):
        """Test title wraps appropriately for 3-col layout, square AOI."""
        figname = 'plot_raster_list_long_title_3_col.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*3
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_override_title(self):
        """Test default title can be overriden."""
        figname = 'plot_raster_list_override_title.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        make_simple_raster(self.raster_config.raster_path, shape)

        self.raster_config.title = 'Special Title'
        config_list = [self.raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)


@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class RasterPlotUnitTextTests(unittest.TestCase):
    """Snapshot tests for plotting rasters with unit text."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = RasterPlotConfig(
            os.path.join(self.workspace_dir, 'test.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'))

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @patch('natcap.invest.reports.raster_utils._get_raster_units')
    def test_plot_raster_list_unit_text_tall_aoi(self, mock_get_raster_units):
        """Test unit text is above raster plot and not overlapping."""
        mock_get_raster_units.return_value = 'flux capacitrons'
        figname = 'plot_raster_list_unit_text_tall_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (8, 4)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*3
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    @patch('natcap.invest.reports.raster_utils._get_raster_units')
    def test_plot_raster_list_unit_text_wide_aoi(self, mock_get_raster_units):
        """Test unit text is above raster plot and not overlapping."""
        mock_get_raster_units.return_value = 'flux capacitrons'
        figname = 'plot_raster_list_unit_text_wide_aoi.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (6, 12)
        make_simple_raster(self.raster_config.raster_path, shape)

        config_list = [self.raster_config]*2
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

@unittest.skipIf(
    MPL_VERSION < (3, 11, 0), 'Snapshots were created with matplotlib 3.11.0')
class SpecialConfigValueUnitTests(unittest.TestCase):
    """Unit tests for SpecialConfigValue constructions."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_special_value_config(self):
        """Should pass when only index 0 (lower bound) is fully populated."""
        config = SpecialValueConfig(
            thresholds=(0.0, 100.0),
            labels=("Low", "High"),
            colors=("blue", "red")
        )
        self.assertEqual(config.thresholds, (0.0, 100.0))

        config = SpecialValueConfig(
            thresholds=(0.0, None),
            labels=("Low", None),
            colors=("blue", None)
        )
        self.assertEqual(config.colors, ("blue", None))

        with self.assertRaises(ValidationError) as context:
            SpecialValueConfig(
                thresholds=(0.0, None),
                labels=(None, None),  # missing lower label
                colors=("blue", None)
            )
        self.assertTrue(
            "If index 0 is `None` in any of the special config tuples" in
            str(context.exception))

        with self.assertRaises(ValidationError) as context:
            SpecialValueConfig(
                thresholds=(0.0, None),  # missing upper threshold
                labels=("Low", "High"),
                colors=("blue", None)  # missing upper color
            )
        self.assertTrue(
            "If index 1 is `None` in any of the special config tuples" in
            str(context.exception))

    def test_special_values_rejected_for_nominal(self):
        """RasterPlotConfig does not allow special values for nominal raster"""
        with self.assertRaisesRegex(
                ValueError, '`special_values` may only be defined'):
            raster_utils.RasterPlotConfig(
                raster_path=os.path.join(self.workspace_dir, 'foo.tif'),
                datatype=raster_utils.RasterDatatype.nominal,
                spec=spec.Output(id='foo', about='foo output'),
                special_values=raster_utils.SpecialValueConfig(
                    thresholds=(-1, 1),
                    labels=('low', 'high'),
                    colors=('red', 'blue')))

    def test_special_values_rejected_for_binary(self):
        """RasterPlotConfig does not allow special values for binary raster"""
        with self.assertRaisesRegex(
                ValueError, '`special_values` may only be defined'):
            raster_utils.RasterPlotConfig(
                raster_path=os.path.join(self.workspace_dir, 'foo.tif'),
                datatype=raster_utils.RasterDatatype.binary,
                spec=spec.Output(id='foo', about='foo output'),
                special_values=raster_utils.SpecialValueConfig(
                    thresholds=(-1, 1),
                    labels=('low', 'high'),
                    colors=('red', 'blue')))

    def test_configure_special_values_both_bounds(self):
        """_configure_special_values configures both colorbar extensions."""
        cmap = matplotlib.colormaps['viridis'].copy()
        special_values = raster_utils.SpecialValueConfig(
            thresholds=(-1, 1),
            labels=('low', 'high'),
            colors=('red', 'blue'))

        cmap, extend, thresholds, labels, text_specs = (
            raster_utils._configure_special_values(cmap, special_values))

        self.assertEqual(extend, 'both')
        self.assertEqual(thresholds, [-1, 1])
        self.assertEqual(labels, ['low', 'high'])
        self.assertEqual(
            text_specs, [(0, -0.05, 'top'), (0, 1.05, 'bottom')])

    def test_plot_divergent_log_raster_requires_symmetric_thresholds(
            self):
        """Divergent log special values must be symmetric around 0."""
        shape = (4, 4)
        raster_config = raster_utils.RasterPlotConfig(
            raster_path=os.path.join(self.workspace_dir, 'foo.tif'),
            datatype=raster_utils.RasterDatatype.divergent,
            transform="log",
            spec=spec.Output(id='foo', about='foo output'),
            special_values=raster_utils.SpecialValueConfig(
                thresholds=(0.4, 1),
                labels=('low', 'high'),
                colors=('black', 'orange')))
        make_simple_raster(raster_config.raster_path, shape)

        with self.assertRaisesRegex(
                UserWarning, 'To ensure that 0 falls at the logical break'):
            raster_utils.plot_raster_list([raster_config])

    def test_plot_continuous_raster_special_values(self):
        """Test correct plot for continuous raster with special values"""
        figname = 'plot_raster_list_special_values.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        raster_config = raster_utils.RasterPlotConfig(
            raster_path=os.path.join(self.workspace_dir, 'foo.tif'),
            datatype=raster_utils.RasterDatatype.continuous,
            spec=spec.Output(id='foo', about='foo output'),
            special_values=raster_utils.SpecialValueConfig(
                thresholds=(0.4, 1),
                labels=('low', 'high'),
                colors=('black', 'orange')))
        make_simple_raster(raster_config.raster_path, shape)

        config_list = [raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_divergent_raster_max_special_value(self):
        """Test divergent raster plot w special value has a correct colorbar"""
        figname = 'plot_raster_list_special_max_value.png'
        reference = os.path.join(REFS_DIR, figname)
        shape = (4, 4)
        raster_config = raster_utils.RasterPlotConfig(
            raster_path=os.path.join(self.workspace_dir, 'foo.tif'),
            datatype=raster_utils.RasterDatatype.divergent,
            spec=spec.Output(id='foo', about='foo output'),
            special_values=raster_utils.SpecialValueConfig(
                thresholds=(None, 0.8),
                labels=(None, 'high'),
                colors=(None, 'darkblue')))
        # Note this raster doesn't actually have negative values
        make_simple_raster(raster_config.raster_path, shape)

        config_list = [raster_config]
        fig = raster_utils.plot_raster_list(config_list)
        actual_png = os.path.join(self.workspace_dir, figname)
        save_figure(fig, actual_png)
        compare_snapshots(reference, actual_png)

    def test_plot_raster_list_special_values_adds_threshold_ticks(self):
        """Test plot_raster_list adds special values as colorbar ticks."""
        thresholds = (-.8, .9)
        shape = (4, 4)
        raster_config = raster_utils.RasterPlotConfig(
            raster_path=os.path.join(self.workspace_dir, 'foo.tif'),
            datatype=raster_utils.RasterDatatype.continuous,
            spec=spec.Output(id='foo', about='foo output'),
            special_values=raster_utils.SpecialValueConfig(
                thresholds=thresholds,
                labels=('low', 'high'),
                colors=('red', 'blue')))
        make_simple_raster(raster_config.raster_path, shape)

        fig = raster_utils.plot_raster_list([raster_config])
        colorbar_ax = fig.axes[1]
        ticks = list(colorbar_ax.get_yticks())

        self.assertIn(thresholds[0], ticks)
        self.assertIn(thresholds[1], ticks)


class RasterCaptionTests(unittest.TestCase):
    """Unit tests for caption-generating utility."""

    def test_generate_caption_from_raster_list(self):
        args_dict = {'raster_1': 'path/to/raster_1.tif'}
        file_registry = {'raster_2': 'path/to/raster_2.tif'}

        about_raster_1 = 'Map of land use/land cover codes.'
        raster1_config = RasterPlotConfig(
            raster_path=args_dict['raster_1'],
            datatype=RasterDatatype.nominal,
            spec=spec.Input(
                id='raster_1',
                about=about_raster_1))

        about_raster_2 = ('The total amount of sediment exported from each '
                          'pixel that reaches the stream.')
        caption_appendix = 'extra text'
        raster2_config = RasterPlotConfig(
            raster_path=file_registry['raster_2'],
            datatype=RasterDatatype.continuous,
            spec=spec.Output(
                id='raster_2',
                about=about_raster_2))
        raster2_config.caption += caption_appendix

        expected_caption = [
            f'raster_1.tif:{about_raster_1}',
            f'raster_2.tif:{about_raster_2}{caption_appendix}'
        ]

        generated_caption = raster_utils.caption_raster_list(
            [raster1_config, raster2_config])

        self.assertEqual(generated_caption, expected_caption)


class PlotCategoricalRastersTest(unittest.TestCase):
    """Unit tests for plotting categorical rasters with attribute table."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    @staticmethod
    def html_string_to_dataframe(html_string):
        """Helper to convert an HTML table string to a pandas dataframe."""
        soup = BeautifulSoup(html_string, 'html.parser')
        rows = soup.find_all('tr')
        header = rows[0]
        cells = header.find_all(['th', 'td'])
        col_names = [cell.get_text() for cell in cells]
        df = pandas.DataFrame(columns=col_names)

        for idx, row in enumerate(rows[1:]):
            cells = row.find_all(['td'])
            cell_values = [cell.get_text() for cell in cells]
            df.loc[idx + 1] = cell_values
        return df

    def test_plot_single_raster_with_rat(self):
        """Test single raster with RAT."""
        target_filepath = os.path.join(self.workspace_dir, 'lulc.tif')
        num_vals = 12
        make_nominal_raster_with_distinct_counts(
            target_filepath, num_vals)
        add_raster_attribute_table(target_filepath, extra_cols=['foo'])
        img_src, table = raster_utils.plot_categorical_raster_with_table(
            [target_filepath])
        df = self.html_string_to_dataframe(table)
        # Our RAT has 3 columns, a 4th was added for the color swatch
        self.assertEqual(df.shape, (num_vals, 4))

    def test_plot_single_raster_with_rat_color_table(self):
        """Test that the RAT color table will be filtered out."""
        target_filepath = os.path.join(self.workspace_dir, 'lulc.tif')
        num_vals = 12
        make_nominal_raster_with_distinct_counts(
            target_filepath, num_vals)
        add_raster_attribute_table(
            target_filepath,
            extra_cols=['RED', 'green', 'Blue', 'ALPHA', 'opacity'])
        img_src, table = raster_utils.plot_categorical_raster_with_table(
            [target_filepath])
        df = self.html_string_to_dataframe(table)
        # All the RGBA (and opacity) columns should be filtered out
        self.assertEqual(df.shape, (num_vals, 3))

    def test_plot_single_raster_without_rat(self):
        """Test single raster without RAT."""
        target_filepath = os.path.join(self.workspace_dir, 'lulc.tif')
        num_vals = 12
        make_nominal_raster_with_distinct_counts(
            target_filepath, num_vals)
        img_src, table = raster_utils.plot_categorical_raster_with_table(
            [target_filepath])
        # In the absence of a RAT, a frequency table was constructed
        df = self.html_string_to_dataframe(table)
        self.assertEqual(df.shape, (num_vals, 3))
        self.assertCountEqual(
            df.columns,
            ['color', 'value', 'count'])

    def test_plot_two_rasters_with_rat(self):
        """Test two rasters with RATS and expect RATS are joined."""
        filepath_a = os.path.join(self.workspace_dir, 'lulc_a.tif')
        num_vals_a = 6
        make_nominal_raster_with_distinct_counts(
            filepath_a, num_vals_a)
        add_raster_attribute_table(filepath_a)

        filepath_b = os.path.join(self.workspace_dir, 'lulc_b.tif')
        num_vals_b = 8
        make_nominal_raster_with_distinct_counts(
            filepath_b, num_vals_b)
        add_raster_attribute_table(filepath_b, extra_cols=['foo'])
        img_src, table = raster_utils.plot_categorical_raster_with_table(
            [filepath_a, filepath_b])
        df = self.html_string_to_dataframe(table)
        # In this example, the union of unique values in 'a' and 'b' happens to
        # equal the unique values in 'b'
        unique_values = num_vals_b
        # The value and color columns are shared; each has a count col;
        # and b has 1 extra column.
        n_cols = 5
        self.assertEqual(df.shape, (unique_values, n_cols))

    def test_plot_two_rasters_one_missing_rat(self):
        """Test two rasters with one missing a RAT."""
        filepath_a = os.path.join(self.workspace_dir, 'lulc_a.tif')
        num_vals_a = 6
        make_nominal_raster_with_distinct_counts(
            filepath_a, num_vals_a)
        # This RAT will be ignored because the 2nd raster does not also
        # have a RAT.
        add_raster_attribute_table(
            filepath_a, extra_cols=['foo', 'bar', 'baz'])

        filepath_b = os.path.join(self.workspace_dir, 'lulc_b.tif')
        num_vals_b = 8
        make_nominal_raster_with_distinct_counts(
            filepath_b, num_vals_b)
        img_src, table = raster_utils.plot_categorical_raster_with_table(
            [filepath_a, filepath_b])
        df = self.html_string_to_dataframe(table)
        # Since one was missing a rat, we expect the plot function to ignore
        # the existing RAT and build its own frequency table for both.
        # In this example, the union of unique values in 'a' and 'b' happens to
        # equal the unique values in 'b'
        unique_values = num_vals_b
        # The value and color columns are shared; each has a count col;
        n_cols = 4
        self.assertEqual(df.shape, (unique_values, n_cols))

    def test_plot_two_rasters_with_incompatible_rat(self):
        """Test two rasters with non-matching RATS."""
        filepath_a = os.path.join(self.workspace_dir, 'lulc_a.tif')
        num_vals_a = 6
        make_nominal_raster_with_distinct_counts(
            filepath_a, num_vals_a)
        add_raster_attribute_table(filepath_a, value_name='VALUE')

        filepath_b = os.path.join(self.workspace_dir, 'lulc_b.tif')
        num_vals_b = 8
        make_nominal_raster_with_distinct_counts(
            filepath_b, num_vals_b)
        add_raster_attribute_table(filepath_b, value_name='VAL')
        img_src, table = raster_utils.plot_categorical_raster_with_table(
            [filepath_a, filepath_b])
        df = self.html_string_to_dataframe(table)

        # Since the two RAT do not have a common value column, they could
        # not be joined. Expect the default frequency table instead.
        unique_values = num_vals_b
        # The value and color columns are shared; each has a count col;
        n_cols = 4
        self.assertEqual(df.shape, (unique_values, n_cols))
        self.assertCountEqual(
            df.columns,
            ['color', 'value', 'count_left', 'count_right'])

    def test_plot_three_rasters_without_rat(self):
        """Test three rasters raises ValueError."""
        target_filepath = os.path.join(self.workspace_dir, 'lulc.tif')
        num_vals = 12
        make_nominal_raster_with_distinct_counts(
            target_filepath, num_vals)
        with self.assertRaises(ValueError):
            img_src, table = raster_utils.plot_categorical_raster_with_table(
                [target_filepath] * 3)


class RasterWorkspaceSummaryTests(unittest.TestCase):
    """Unit tests for output raster stats table-generating utility."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override tearDown function to remove temporary directory."""
        shutil.rmtree(self.workspace_dir)

    def test_raster_workspace_summary(self):
        from tests.utils import fake_execute, SAMPLE_MODEL_SPEC

        # Generate an output workspace with real files & metadata
        # without running an invest model.
        args_dict = {'workspace_dir': self.workspace_dir}
        SAMPLE_MODEL_SPEC.create_output_directories(args_dict)
        file_registry = fake_execute(
            SAMPLE_MODEL_SPEC.outputs, self.workspace_dir)
        SAMPLE_MODEL_SPEC.generate_metadata_for_outputs(
            file_registry, args_dict)
        dataframe = raster_utils.raster_workspace_summary(file_registry)

        # There are 3 rasters in the sample output spec
        self.assertEqual(dataframe.shape, (3, 7))
