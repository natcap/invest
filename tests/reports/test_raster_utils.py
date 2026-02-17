import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.testing.compare
from matplotlib.testing import set_font_settings_for_testing
from matplotlib.testing.exceptions import ImageComparisonFailure
import numpy
from osgeo import osr
import pygeoprocessing

from natcap.invest import spec
from natcap.invest.unit_registry import u
from natcap.invest.reports import MATPLOTLIB_PARAMS, raster_utils
from natcap.invest.reports.raster_utils import RasterDatatype, RasterTransform

projection = osr.SpatialReference()
projection.ImportFromEPSG(3857)
PROJ_WKT = projection.ExportToWkt()

REFS_DIR = os.path.join('data', 'invest-test-data', 'reports', 'snapshots')


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
        array, target_nodata=None, pixel_size=(1, 1), origin=(0, 0),
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
    """Snapshot tests for matplotlib figure layouts."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = raster_utils.RasterPlotConfig(
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
        continuous_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'))
        make_simple_raster(continuous_raster.raster_path, shape)

        binary_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'binary.tif'),
            RasterDatatype.binary,
            spec.Output(id='foo'))
        binary_array = numpy.zeros(shape=shape)
        binary_array[0] = 1
        pygeoprocessing.numpy_array_to_raster(
            binary_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=binary_raster.raster_path)

        divergent_raster = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'divergent.tif'),
            RasterDatatype.divergent,
            spec.Output(id='foo'))
        divergent_array = numpy.linspace(
            -1, 1, num=numpy.multiply(*shape)).reshape(*shape)
        pygeoprocessing.numpy_array_to_raster(
            divergent_array, target_nodata=None, pixel_size=(1, 1),
            origin=(0, 0), projection_wkt=PROJ_WKT,
            target_path=divergent_raster.raster_path)

        nominal_raster = raster_utils.RasterPlotConfig(
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
        continuous_raster_linear = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'),
            transform=RasterTransform.linear)
        continuous_raster_log = raster_utils.RasterPlotConfig(
            os.path.join(self.workspace_dir, 'continuous.tif'),
            RasterDatatype.continuous,
            spec.Output(id='foo'),
            transform=RasterTransform.log)
        make_simple_raster(continuous_raster_linear.raster_path, shape)

        divergent_raster_log = raster_utils.RasterPlotConfig(
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


class RasterPlotLegendTests(unittest.TestCase):
    """Snapshot tests for legend placement on nominal rasters."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = raster_utils.RasterPlotConfig(
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


class RasterPlotFacetsTests(unittest.TestCase):
    """Snapshot tests for plotting multiple rasters on the same colorscale."""

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


class RasterPlotTitleTests(unittest.TestCase):
    """Snapshot tests for plotting rasters with various titles."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = raster_utils.RasterPlotConfig(
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


class RasterPlotUnitTextTests(unittest.TestCase):
    """Snapshot tests for plotting rasters with unit text."""

    def setUp(self):
        """Override setUp function to create temp workspace directory."""
        self.workspace_dir = tempfile.mkdtemp()
        self.raster_config = raster_utils.RasterPlotConfig(
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


class RasterCaptionTests(unittest.TestCase):
    """Unit tests for caption-generating utility."""

    def test_generate_caption_from_raster_list(self):
        raster_list = [('raster_1', 'input'), ('raster_2', 'output')]
        args_dict = {'raster_1': 'path/to/raster_1.tif'}
        file_registry = {'raster_2': 'path/to/raster_2.tif'}
        model_spec = spec.ModelSpec(
            model_id='',
            model_title='',
            userguide='',
            module_name='',
            input_field_order=[['raster_1']],
            inputs=[
                spec.SingleBandRasterInput(
                    id='raster_1',
                    units=u.none,
                    about='Map of land use/land cover codes.',
                ),
            ],
            outputs=[
                spec.SingleBandRasterOutput(
                    id='raster_2',
                    path='path/to/raster_2.tif',
                    units=u.metric_ton / u.hectare,
                    about=('The total amount of sediment exported from each '
                           'pixel that reaches the stream.'),
                )
            ],
        )

        expected_caption = [
            'raster_1.tif:Map of land use/land cover codes.',
            ('raster_2.tif:The total amount of sediment exported from each '
             'pixel that reaches the stream.')
        ]

        generated_caption = raster_utils.generate_caption_from_raster_list(
            raster_list, args_dict, file_registry, model_spec)

        self.assertEqual(generated_caption, expected_caption)


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

        # There are 2 rasters in the sample output spec
        self.assertEqual(dataframe.shape, (2, 7))
