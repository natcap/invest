import base64
import collections
import logging
import math
import os
from io import BytesIO
from enum import Enum

import distinctipy
import geometamaker
import numpy
import pygeoprocessing
import matplotlib
import matplotlib.colors
from matplotlib.colors import ListedColormap
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas
import yaml
from osgeo import gdal
from pydantic.dataclasses import dataclass

from natcap.invest.spec import ModelSpec
from natcap.invest.reports.report_constants import TABLE_PAGINATION_THRESHOLD

LOGGER = logging.getLogger(__name__)

MPL_SAVE_FIG_KWARGS = {
    'format': 'png',
    'bbox_inches': 'tight'
}

# Our CSS sets report container max width to 80rem, which is 1280px
# (with default browser settings, i.e., root font size = 16px).
# After accounting for padding and borders,
# an img typically has a max width of 75.5rem, or 1208px.
# It's best if figures are sized to fill their containers
# with minimal rescaling, since they contain rasterized text.
# Other variables:
#   - When creating a figure, e.g., with plt.subplots, default dpi is 100.
#   - Creating a figure with layout='constrained' may automagically adjust
#     subplot sizes and grid spacing (potentially affecting overall figure
#     size) to ensure colorbars/legends don't overlap or get cut off.
#   - Savefig with tight bbox layout shrinks the figure after it is sized.
# In practice, with all the variables mentioned above, final image width tends
# to be a few hundredths of an inch larger than what is specified when creating
# the figure. With that in mind, we set max figure width slightly smaller than
# the desired image width.
MAX_FIGURE_WIDTH_INCHES = 12  # 1208px / 100 dpi = 12.08 in => round down to 12.0
# The goal of max height is to ensure a given subplot fits within the vertical
# bounds of a maximized window on any laptop/desktop screen. At one extreme,
# some laptop screens are only 768px high. Allowing some buffer for browser
# toolbars (~ 10% of window height) leaves around 700px for the subplot itself.
MAX_SUBPLOT_HEIGHT_INCHES = 7  # 700px / 100 dpi = 7 in

TITLE_FONT_SIZE = 13  # 13pt ≈ 18.1px
SUBTITLE_FONT_SIZE = 11  # 11pt ≈ 15.3px

# Mapping 'datatype' to colormaps and resampling algorithms
COLORMAPS = {
    'continuous': 'viridis',
    'divergent': 'BrBG',
    # Default for nominal data is matplotlib's tab20.
    # If > 20 colors are needed, colormap will be generated with distinctipy.
    'nominal': 'tab20',
    # This `1` color has good (but not especially high) contrast against both
    # black (the `0` color) and white (the figure background).
    'binary': ListedColormap(["#000000", "#aa44dd"]),
    # The `1` color has very high contrast against the `0` color but
    # very low contrast against white (the figure background).
    'binary_high_contrast': ListedColormap(["#1a1a1a", "#4de4ff"]),
}
RESAMPLE_ALGS = {
    'continuous': 'bilinear',
    'divergent': 'bilinear',
    'nominal': 'nearest',
    'binary': 'nearest',
    'binary_high_contrast': 'nearest'
}

# XY-ratio thresholds used in determining grid layouts and legend layouts
WIDE_AOI_THRESHOLD = 1
EX_WIDE_AOI_THRESHOLD = 4


class RasterDatatype(str, Enum):
    """The type of measurement represented by the raster."""

    binary = 'binary'
    """
    Use `binary` where `1` pixels are likely to be adjacent to white
    background and `0` (black) pixels, as in what_drains_to_stream maps.
    """
    binary_high_contrast = 'binary_high_contrast'
    """
    Use `binary_high_contrast` where `1` pixels are likely to be surrounded
    # by `0` pixels but _not_ adjacent to white background,
    # as in stream network maps.
    """
    continuous = 'continuous'
    """For numeric data."""
    divergent = 'divergent'
    """For rasters where values span positive and negative values."""
    nominal = 'nominal'
    """For rasters where the pixel values represent categories."""


class RasterTransform(str, Enum):
    """The transformation to apply to values before mapping to colors.

    Original values are plotted, but the colorbar will be use this scale.
    """

    linear = 'linear'
    log = 'log'


@dataclass
class RasterPlotConfig:
    """A definition for how to plot a raster."""

    raster_path: str
    """Filepath to a raster to plot."""
    datatype: RasterDatatype
    """Datatype will determine colormap, legend, and resampling algorithm"""
    transform: RasterTransform = RasterTransform.linear
    """For highly skewed data, a transformation can help reveal variation."""


@dataclass
class RasterPlotConfigGroup:
    inputs: list[RasterPlotConfig] | None
    outputs: list[RasterPlotConfig] | None
    intermediates: list[RasterPlotConfig] | None


@dataclass
class RasterPlotCaptionGroup:
    inputs: list[str] | None
    outputs: list[str] | None
    intermediates: list[str] | None


def build_raster_plot_configs(id_lookup_table, raster_plot_tuples):
    """Build RasterPlotConfigs for use in plotting input or output rasters.

    Args:
        id_lookup_table (dict): Where to look up each raster id to find its
            filepath. Typically this will be either the ``args`` dict that was
            passed to the model's ``execute`` method, or the ``FileRegistry``
            that was generated by the model run.
        raster_plot_tuples (list[tuple[str]]): A list of 2- and/or 3-tuples,
            each of which should contain the following:
            - first, the id of the raster (as defined in the model spec),
            - second, the datatype of the raster ('continuous', 'divergent',
            'nominal', 'binary', or 'binary_high_contrast'), and
            - third (optionally), the transform to apply to the colormap when
            plotting (either 'linear' or 'log'; will default to 'linear' if
            not specified).

    Returns:
        A list of ``RasterPlotConfig`` suitable for passing to
            ``natcap.invest.reports.raster_utils.plot_and_base64_encode_rasters``

    """
    raster_plot_configs = []
    for (raster_id, *other_args) in raster_plot_tuples:
        raster_path = id_lookup_table[raster_id]
        raster_plot_configs.append(RasterPlotConfig(raster_path, *other_args))
    return raster_plot_configs


def generate_caption_from_raster_list(
        raster_list: list[tuple[str, str]], args_dict,
        file_registry, model_spec: ModelSpec):
    """Concatenate filenames and metadata descriptions to create captions."""
    caption = []
    for (id, input_or_output) in raster_list:
        if input_or_output == 'input':
            filename = os.path.basename(args_dict[id])
            about_text = model_spec.get_input(id).about
        elif input_or_output == 'output':
            about_text = model_spec.get_output(id).about
            filename = os.path.basename(file_registry[id])
        caption.append(f'{filename}:{about_text}')
    return caption


def _read_masked_array(filepath, resample_method):
    """Read a raster into a masked numpy array.

    If the raster is large, build overviews and then read array from the
    smallest-sized overview. Nodata values are assigned ``numpy.nan``
    to facilitate matplotlib plotting.

    Args:
        filepath (str): path to the raster file.
        resample_method (str): GDAL resampling method to use if resampling

    Returns:
        tuple: A 2-tuple containing:
            - masked_array (numpy.ndarray): the raster data as a numpy array
            - resampled (boolean): whether or not the array is an overview

    """
    info = pygeoprocessing.get_raster_info(filepath)
    nodata = info['nodata'][0]
    resampled = False
    if os.path.getsize(filepath) > 4e6:
        resampled = True
        raster = gdal.OpenEx(filepath)
        band = raster.GetRasterBand(1)
        if band.GetOverviewCount() == 0:
            pygeoprocessing.build_overviews(
                filepath,
                internal=False,
                resample_method=resample_method,
                overwrite=False, levels='auto')

        raster = gdal.OpenEx(filepath)
        band = raster.GetRasterBand(1)
        n = band.GetOverviewCount()
        array = band.GetOverview(n - 1).ReadAsArray()
        raster = band = None
    else:
        array = pygeoprocessing.raster_to_numpy_array(filepath)
    masked_array = numpy.full(array.shape, numpy.nan)
    if nodata is not None:
        valid_mask = ~numpy.isclose(array, nodata, equal_nan=True)
    else:
        valid_mask = numpy.full(array.shape, True)
    masked_array[valid_mask] = array[valid_mask]
    return (masked_array, resampled)


def _get_aspect_ratio(map_bbox):
    return (map_bbox[2] - map_bbox[0]) / (map_bbox[3] - map_bbox[1])


def _choose_n_rows_n_cols(xy_ratio, n_plots):
    if xy_ratio <= WIDE_AOI_THRESHOLD:
        n_cols = 3
    elif xy_ratio > EX_WIDE_AOI_THRESHOLD:
        n_cols = 1
    elif xy_ratio > WIDE_AOI_THRESHOLD:
        n_cols = 2

    if n_cols > n_plots:
        n_cols = n_plots
    n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols


def _figure_subplots(xy_ratio, n_plots):
    n_rows, n_cols = _choose_n_rows_n_cols(xy_ratio, n_plots)

    figure_width = MAX_FIGURE_WIDTH_INCHES
    sub_width = figure_width / n_cols
    sub_height = (sub_width / xy_ratio)
    figure_height = sub_height * n_rows
    max_figure_height = MAX_SUBPLOT_HEIGHT_INCHES * n_rows
    if figure_height > max_figure_height:
        # Constrain height, then adjust width accordingly.
        downscale_factor = max_figure_height / figure_height
        figure_height = max_figure_height
        figure_width *= downscale_factor

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(figure_width, figure_height),
        layout='constrained')
    if n_plots == 1:
        axs = numpy.array([axs])
    return fig, axs


def _get_title_kwargs(raster_path: str, resampled: bool, subtitle: str = ''):
    filename = os.path.basename(raster_path)
    label = f"{filename}{' (resampled)' if resampled else ''}"
    label = f"{label}\n{subtitle}" if subtitle else label
    return {
        'fontfamily': 'monospace',
        'fontsize': TITLE_FONT_SIZE,
        'fontweight': 700,
        'label': label,
        'loc': 'left',
        'pad': 1.5 * SUBTITLE_FONT_SIZE,
        'verticalalignment': 'bottom',
    }


def _get_units_text_kwargs(units: str, raster_height: int):
    # This -0.1 multiplier is a bit of a 'magic number' but seems to work for now.
    subtitle_offset = -0.1 * raster_height
    # Set ylim top < 0 to add some padding above the plot.
    ylim_args = {
        'bottom': raster_height,
        'top': subtitle_offset,
    }
    # Place subtitle text immediately above that padding.
    text_args = {
        'fontsize': SUBTITLE_FONT_SIZE,
        'horizontalalignment': 'left',
        's': f'Units: {units}',
        'verticalalignment': 'bottom',
        'x': -0.5,
        'y': subtitle_offset,
    }
    return (ylim_args, text_args)


def _get_legend_kwargs(num_patches: int, n_plots: int, xy_ratio: float):
    # Num legend cols/rows determined experimentally; may change if needed.
    MAX_LEGEND_COLS_1_COL_LAYOUT = 10
    MAX_LEGEND_COLS_2_COL_LAYOUT = 4
    MAX_LEGEND_ROWS = 30
    if xy_ratio > WIDE_AOI_THRESHOLD:
        bbox_to_anchor = (-0.01, 0)
        if n_plots == 1 or xy_ratio > EX_WIDE_AOI_THRESHOLD:
            ncol = MAX_LEGEND_COLS_1_COL_LAYOUT
        else:
            ncol = MAX_LEGEND_COLS_2_COL_LAYOUT
    else:
        bbox_to_anchor = (1.02, 1)
        ncol = math.ceil(num_patches / MAX_LEGEND_ROWS)
    return {
        'bbox_to_anchor': bbox_to_anchor,
        'loc': 'upper left',
        'ncol': ncol,
    }


def plot_raster_list(raster_list: list[RasterPlotConfig]):
    """Plot a list of rasters.

    Args:
        raster_list (list[RasterPlotConfig]): a list of RasterPlotConfig
            objects, each of which contains the path to a raster, the type
            of data in the raster ('continuous', 'divergent', 'nominal',
            'binary', or 'binary_high_contrast'), and the transformation to
            apply to the colormap ('linear' or 'log').

    Returns:
        ``matplotlib.figure.Figure``
    """
    raster_info = pygeoprocessing.get_raster_info(raster_list[0].raster_path)
    bbox = raster_info['bounding_box']
    n_plots = len(raster_list)
    xy_ratio = _get_aspect_ratio(bbox)
    fig, axs = _figure_subplots(xy_ratio, n_plots)

    for ax, config in zip(axs.flatten(), raster_list):
        raster_path = config.raster_path
        dtype = config.datatype
        transform = config.transform
        resample_alg = RESAMPLE_ALGS[dtype]
        arr, resampled = _read_masked_array(raster_path, resample_alg)
        imshow_kwargs = {}
        colorbar_kwargs = {}
        imshow_kwargs['norm'] = transform
        imshow_kwargs['interpolation'] = 'none'
        cmap = COLORMAPS[dtype]
        if dtype == 'divergent':
            if transform == 'log':
                transform = matplotlib.colors.SymLogNorm(linthresh=0.03)
            else:
                transform = matplotlib.colors.CenteredNorm()
            imshow_kwargs['norm'] = transform
        if dtype.startswith('binary'):
            transform = matplotlib.colors.BoundaryNorm([0, 0.5, 1], cmap.N)
            imshow_kwargs['vmin'] = -0.5
            imshow_kwargs['vmax'] = 1.5
            colorbar_kwargs['ticks'] = [0, 1]

        ax.set_title(**_get_title_kwargs(raster_path, resampled))

        units = _get_raster_units(raster_path)
        if units:
            (ylim_kwargs,
             text_kwargs) = _get_units_text_kwargs(units, len(arr))
            ax.set_ylim(**ylim_kwargs)
            ax.text(**text_kwargs)

        if dtype == 'nominal':
            # typically a 'nominal' raster would be an int type, but we replaced
            # nodata with nan, so the array is now a float.
            values, counts = numpy.unique(arr[~numpy.isnan(arr)], return_counts=True)
            values = values[numpy.argsort(-counts)].astype(int)  # descending order
            # We need enough colors to cover the full range of values.
            # If there is only one color per unique value, and the range of
            # values is larger than the number of unique values, matplotlib's
            # normalization can cause multiple values to be represented by the
            # same color.
            # (Future work may involve writing a custom normalizer to prevent
            # this problem and generate only one color for each unique value.)
            num_colors = numpy.max(values) - numpy.min(values) + 1
            # If > 20 colors needed, generate colormap to override default.
            if num_colors > 20:
                # Values of pastel_factor and rng have been chosen specifically
                # for Carbon (Willamette) sample data. If/when we create a
                # report using sample data that is ill-suited to the color
                # palette generated with these values, we will take a different
                # approach to customizing color palettes.
                cmap = ListedColormap(
                    distinctipy.get_colors(
                        num_colors, pastel_factor=0.6, rng=0))

            mappable = ax.imshow(arr, cmap=cmap, **imshow_kwargs)
            colors = [mappable.cmap(mappable.norm(value)) for value in values]
            patches = [matplotlib.patches.Patch(
                color=colors[i],
                label=f'{values[i]}') for i in range(len(values))]
            leg = ax.legend(handles=patches, **_get_legend_kwargs(
                len(patches), n_plots, xy_ratio))
            leg.set_in_layout(True)
        else:
            mappable = ax.imshow(arr, cmap=cmap, **imshow_kwargs)
            fig.colorbar(mappable, ax=ax, **colorbar_kwargs)
    [ax.set_axis_off() for ax in axs.flatten()]
    return fig


def base64_encode(figure):
    """Encode a Matplotlib-generated figure as a base64 string.

    Args:
        figure (matplotlib.Figure): the figure to encode.

    Returns:
        A string representing the figure as a base64-encoded PNG.
    """
    figfile = BytesIO()
    figure.savefig(figfile, **MPL_SAVE_FIG_KWARGS)
    figfile.seek(0)  # rewind to beginning of file
    return base64.b64encode(figfile.getvalue()).decode('utf-8')


def base64_encode_file(filepath):
    """Encode a binary file as a base64 string.

    Args:
        filepath (str): the file to encode.

    Returns:
        A string representing the file as a base64-encoded string.
    """
    with open(filepath, 'rb') as file:
        s = base64.b64encode(file.read()).decode('utf-8')
    return s


def plot_and_base64_encode_rasters(raster_list: list[RasterPlotConfig]) -> str:
    """Plot and base-64-encode a list of rasters.

    Args:
        raster_dtype_list (list[RasterPlotConfig]): a list of RasterPlotConfig
            objects, each of which contains the path to a raster, the type
            of data in the raster ('continuous', 'divergent', 'nominal',
            'binary', or 'binary_high_contrast'), and the transformation to
            apply to the colormap ('linear' or 'log').

    Returns:
        A string representing a base64-encoded PNG in which each of the
            provided rasters is plotted as a subplot.
    """
    figure = plot_raster_list(raster_list)

    return base64_encode(figure)


def plot_raster_facets(tif_list, datatype, transform=None, subtitle_list=None):
    """Plot a list of rasters that will all share a fixed colorscale.

    When all the rasters have the same shape and represent the same variable,
    it's useful to scale the colorbar to the global min/max values across
    all rasters, so that the colors are visually comparable across the maps.
    All rasters share a datatype and a transform.

    Args:
        tif_list (list): list of filepaths to rasters
        datatype (str): string describing the datatype of rasters. One of
            ('continuous', 'divergent').
        transform (str): string describing the transformation to apply
            to the colormap. Either 'linear' or 'log'.

    """
    raster_info = pygeoprocessing.get_raster_info(tif_list[0])
    bbox = raster_info['bounding_box']
    n_plots = len(tif_list)
    xy_ratio = _get_aspect_ratio(bbox)
    fig, axes = _figure_subplots(xy_ratio, n_plots)

    cmap_str = COLORMAPS[datatype]
    if transform is None:
        transform = 'linear'
    if subtitle_list is None:
        subtitle_list = ['']*n_plots
    resample_alg = resample_alg = RESAMPLE_ALGS[datatype]
    arr, resampled = _read_masked_array(tif_list[0], resample_alg)
    ndarray = numpy.empty((n_plots, *arr.shape))
    ndarray[0] = arr
    for i, tif in enumerate(tif_list):
        # We already got the first one to initialize the ndarray with correct shape
        if i == 0:
            continue
        arr, resampled = _read_masked_array(tif, RESAMPLE_ALGS[datatype])
        ndarray[i] = arr
    # Perhaps this could be optimized by reading min/max from tif metadata
    # instead of storing all arrays in memory
    vmin = numpy.nanmin(ndarray)
    vmax = numpy.nanmax(ndarray)
    cmap = plt.get_cmap(cmap_str)
    if datatype == 'divergent':
        if transform == 'log':
            normalizer = matplotlib.colors.SymLogNorm(linthresh=0.03, vmin=vmin, vmax=vmax)
        else:
            normalizer = matplotlib.colors.CenteredNorm(vmin=vmin, vmax=vmax)
    if transform == 'log':
        if numpy.isclose(vmin, 0.0):
            vmin = 1e-6
        normalizer = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap.set_under(cmap.colors[0])  # values below vmin (0s) get this color
    else:
        normalizer = plt.Normalize(vmin=vmin, vmax=vmax)
    for arr, ax, raster_path, subtitle in zip(ndarray, axes.flatten(),
                                              tif_list, subtitle_list):
        mappable = ax.imshow(arr, cmap=cmap, norm=normalizer)
        # all rasters are identical size; `resampled` will be the same for all
        ax.set_title(**_get_title_kwargs(raster_path, resampled, subtitle))
        units = _get_raster_units(raster_path)
        if units:
            (ylim_kwargs,
             text_kwargs) = _get_units_text_kwargs(units, len(arr))
            ax.set_ylim(**ylim_kwargs)
            ax.text(**text_kwargs)
        fig.colorbar(mappable, ax=ax)
    [ax.set_axis_off() for ax in axes.flatten()]
    return fig


# TODO: this may end up in the geometamaker API
# https://github.com/natcap/geometamaker/issues/111
def geometamaker_load(filepath):
    # All geometamaker docs are written with utf-8 encoding
    with open(filepath, 'r', encoding='utf-8') as file:
        yaml_string = file.read()
        yaml_dict = yaml.safe_load(yaml_string)
        if not yaml_dict or ('metadata_version' not in yaml_dict
                             and 'geometamaker_version' not in yaml_dict):
            message = (f'{filepath} exists but is not compatible with '
                       f'geometamaker.')
            raise ValueError(message)

    return geometamaker.geometamaker.RESOURCE_MODELS[yaml_dict['type']](
        **yaml_dict)


# GDAL Metadata keys and corresponding column headers for a table
STATS_LIST = [
    ('STATISTICS_MINIMUM', 'Minimum'),
    ('STATISTICS_MAXIMUM', 'Maximum'),
    ('STATISTICS_MEAN', 'Mean'),
    ('STATISTICS_VALID_PERCENT', 'Valid percent'),
]


def _build_stats_table_row(resource, band):
    row = {}
    for (stat_key, display_name) in STATS_LIST:
        stat_val = band.gdal_metadata.get(stat_key)
        if stat_val is not None:
            row[display_name] = float(stat_val)
        else:
            row[display_name] = 'unknown'
    (width, height) = (
        resource.data_model.raster_size['width'],
        resource.data_model.raster_size['height'])
    row['Count'] = width * height
    row['Nodata value'] = band.nodata
    # band.units may be '', which can mean 'unitless', 'unknown', or 'other'
    row['Units'] = band.units
    return row


def _get_raster_metadata(filepath):
    if isinstance(filepath, collections.abc.Mapping):
        for path in filepath.values():
            return _get_raster_metadata(path)
    else:
        try:
            resource = geometamaker_load(f'{filepath}.yml')
        except (FileNotFoundError, ValueError) as err:
            LOGGER.debug(err)
            return None
        if isinstance(resource, geometamaker.models.RasterResource):
            return resource


def _get_raster_units(filepath):
    resource = _get_raster_metadata(filepath)
    return resource.get_band_description(1).units if resource else None


def raster_workspace_summary(file_registry):
    """Create a table of stats for all rasters in a file_registry."""
    raster_summary = {}
    for path in file_registry.values():
        resource = _get_raster_metadata(path)
        band = resource.get_band_description(1) if resource else None
        if band:
            filename = os.path.basename(resource.path)
            raster_summary[filename] = _build_stats_table_row(
                resource, band)

    return pandas.DataFrame(raster_summary).T


def raster_inputs_summary(args_dict):
    """Create a table of stats for all rasters in an args_dict."""
    raster_summary = {}
    for v in args_dict.values():
        if isinstance(v, str) and os.path.isfile(v):
            resource = geometamaker.describe(v, compute_stats=True)
            if isinstance(resource, geometamaker.models.RasterResource):
                filename = os.path.basename(resource.path)
                band = resource.get_band_description(1)
                raster_summary[filename] = _build_stats_table_row(
                    resource, band)
                # Remove 'Units' column if all units are blank
                if not any(raster_summary[filename]['Units']):
                    del raster_summary[filename]['Units']

    return pandas.DataFrame(raster_summary).T


def rat_to_html(raster_path: str) -> str | None:
    with gdal.OpenEx(raster_path) as raster:
        band = raster.GetRasterBand(1)
        rat = band.GetDefaultRAT()
        if rat:
            columns = [pandas.Series(
                rat.ReadAsArray(i), name=rat.GetNameOfCol(i)).astype('string')
                for i in range(rat.GetColumnCount())]
            df = pandas.concat(columns, axis=1)
            css_classes = ['datatable']
            (num_rows, _) = df.shape
            if num_rows > TABLE_PAGINATION_THRESHOLD:
                css_classes.append('paginate')
            return df.to_html(
                index=False, na_rep='', classes=css_classes)
        else:
            return None
