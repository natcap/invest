import base64
import collections
import logging
import math
import textwrap
import os
from io import BytesIO
from enum import Enum

import distinctipy
import geometamaker
import numpy
import pygeoprocessing
import matplotlib
import matplotlib.colors
from matplotlib.colors import Colormap, ListedColormap
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas
import yaml
from osgeo import gdal
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from typing import Literal

from natcap.invest import gettext
from natcap.invest.spec import ModelSpec, Input, Output, \
    CSVInput, SingleBandRasterInput

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
# In addition, while our CSS will scale an image down to fit within its
# container, it will not scale an image up to fill the width of its container.
# (This is by design, to prevent images from becoming too tall.)
# Other variables:
#   - When creating a figure, e.g., with plt.subplots, default dpi is 100.
#   - Creating a figure with layout='compressed' may automagically adjust
#     subplot sizes and grid spacing (potentially affecting overall figure
#     size) to ensure colorbars/legends don't overlap or get cut off.
#   - Savefig with tight bbox layout shrinks the figure after it is sized.
# In practice, with all the variables mentioned above, final image width tends
# to be a few hundredths of an inch larger than what is specified when creating
# the figure. With that in mind, we set max figure width slightly smaller than
# the desired image width.
MAX_FIGURE_WIDTH_DEFAULT = 12  # 1208px/100dpi = 12.08in => round down to 12.
# Two-column grids of "wide AOI" rasters (1 < X/Y ratio <= 4) require a
# deviation from the default, since such figures end up significantly narrower
# than the figure width we specify. To compensate, figure width for such
# layouts is set to a larger number, determined experimentally.
MAX_FIGURE_WIDTH_2_COL_WIDE_AOI = 15  # image will shrink to approx. 12 in.
# The goal of max height is to ensure a given subplot fits within the vertical
# bounds of a maximized window on any laptop/desktop screen. At one extreme,
# some laptop screens are only 768px high. Allowing some buffer for browser
# toolbars (~ 10% of window height) leaves around 700px for the subplot itself.
MAX_SUBPLOT_HEIGHT_INCHES = 7  # 700px / 100 dpi = 7 in

TITLE_FONT_SIZE = 13  # 13pt ≈ 18.1px
SUBTITLE_FONT_SIZE = 11  # 11pt ≈ 15.3px

# There is not enough contrast between the colors at opposite ends
# of divergent colormaps. Truncating them helps a bit.
truncated_PuOr_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'truncated_PuOr', matplotlib.cm.PuOr(numpy.linspace(0.10, 0.90, 256)))

# Mapping 'datatype' to colormaps and resampling algorithms
COLORMAPS = {
    'continuous': 'viridis',
    'divergent': truncated_PuOr_cmap,
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

# GDAL metadata keys and corresponding column headers for stats tables
STATS_LIST = [
    ('STATISTICS_MINIMUM', gettext('Minimum')),
    ('STATISTICS_MAXIMUM', gettext('Maximum')),
    ('STATISTICS_MEAN', gettext('Mean')),
    ('STATISTICS_VALID_PERCENT', gettext('Valid percent')),
]

COUNT_COL_NAME = gettext('Count')
NODATA_COL_NAME = gettext('Nodata value')
UNITS_COL_NAME = gettext('Units')
UNKNOWN_VAL_TEXT = gettext('unknown')
UNITS_TEXT = gettext('Units')


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

    Original values are plotted, but the colorbar will use this scale.
    """

    linear = 'linear'
    log = 'log'


class SpecialValueConfig(BaseModel):
    """Configuration for customizing color and labeling of special value range.

    The colorbar will be extended on either the lower or upper bounds
    (or both), and values beyond the `thresholds` will be colored with
    `colors` and the end of the colorbar will be labeled with `labels`.
    """
    thresholds: tuple[float | None, float | None]
    """Boundary value for the special upper and lower bounds, respectively"""
    labels: tuple[str | None, str | None]
    """Label(s) to show on the colorbar for the special region"""
    colors: tuple[str | None, str | None]
    """Color(s) used for the special values"""

    @model_validator(mode='after')
    def validate_special_value_config(self):
        tuples = (self.thresholds, self.labels, self.colors)
        for idx in [0, 1]:
            if any(t[idx] is None for t in tuples) and not \
                    all(t[idx] is None for t in tuples):
                raise ValueError(
                    f"If index {idx} is `None` in any of the special config "
                    f"tuples, index {idx} must be `None` in all tuples. "
                    "Current incompatible `thresholds`, `labels`, and "
                    f"`colors` tuples are: {tuples}")
        return self


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RasterPlotConfig:
    """A definition for how to plot a raster."""

    raster_path: str
    """Filepath to a raster to plot. The basename will be the plot title."""
    datatype: RasterDatatype
    """Datatype will determine colormap, legend, and resampling algorithm"""
    spec: Input | Output
    """The InVEST specification of the raster."""
    transform: RasterTransform = RasterTransform.linear
    """For highly skewed data, a transformation can help reveal variation."""
    title: str | None = None
    """An optional plot title. If ``None``, the filename is used."""
    colormap: str | Colormap | None = None
    """The string name of a registered matplotlib colormap or a colormap object."""
    special_values: SpecialValueConfig | None = None
    """Will customize the color and labeling of a special range of values"""

    def __post_init__(self):
        if self.title is None:
            self.title = os.path.basename(self.raster_path)
        self.caption = f'{self.title}:{self.spec.about}'

        self.colormap = plt.get_cmap(self.colormap if self.colormap
                                     else COLORMAPS[self.datatype])

    @model_validator(mode='after')
    def check_special_values_and_datatype(self):
        if (self.special_values and self.datatype not in [
                RasterDatatype.continuous, RasterDatatype.divergent]):
            raise ValueError(
                "`special_values` may only be defined for raster configs with "
                "`datatype` of `continuous` or `divergent`."
            )
        return self


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


def caption_raster_list(raster_list: list[RasterPlotConfig]):
    return [config.caption for config in raster_list]


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


def _wide_aoi(xy_ratio):
    return xy_ratio > WIDE_AOI_THRESHOLD and xy_ratio <= EX_WIDE_AOI_THRESHOLD


def _extra_wide_aoi(xy_ratio):
    return xy_ratio > EX_WIDE_AOI_THRESHOLD


def _choose_n_rows_n_cols(xy_ratio, n_plots, small_plots):
    if _extra_wide_aoi(xy_ratio):
        n_cols = 1
    elif _wide_aoi(xy_ratio):
        n_cols = 2
    else:
        n_cols = 3

    if small_plots:
        n_cols += 1

    if n_cols > n_plots:
        n_cols = n_plots
    n_rows = int(math.ceil(n_plots / n_cols))
    return n_rows, n_cols


def _figure_subplots(xy_ratio, n_plots, small_plots=False):
    n_rows, n_cols = _choose_n_rows_n_cols(xy_ratio, n_plots, small_plots)

    figure_width = MAX_FIGURE_WIDTH_DEFAULT
    if (n_cols == 2) and (_wide_aoi(xy_ratio)):
        figure_width = MAX_FIGURE_WIDTH_2_COL_WIDE_AOI
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
        layout='compressed')
    plt.close()
    if n_plots == 1:
        axs = numpy.array([axs])
    return fig, axs


def _get_title_line_width(n_plots: int, xy_ratio: float) -> int:
    # Max line widths determined experimentally; may change if needed.
    if n_plots == 1 or _extra_wide_aoi(xy_ratio):
        return 50  # 1-column layout
    elif n_plots == 2 or _wide_aoi(xy_ratio):
        return 40  # 2-column layout
    else:
        # carbon model sample data includes a 31 char title
        return 31  # 3-column layout


def _get_title_kwargs(title: str, resampled: bool, line_width: int, facets=False):
    label = f"{title}{' (resampled)' if resampled else ''}"
    label = textwrap.fill(label, width=line_width)
    padding = 1.5
    if not facets:
        # Faceted plots don't need extra padding for title because their units
        # label appears with the legend instead of under the title
        padding *= SUBTITLE_FONT_SIZE
    return {
        'fontfamily': 'monospace',
        'fontsize': TITLE_FONT_SIZE,
        'fontweight': 700,
        'label': label,
        'loc': 'left',
        'pad': padding,
        'verticalalignment': 'bottom',
    }


def _get_units_text_kwargs(units: str, subtitle_offset: float):
    # Place subtitle text immediately above subtitle_offset padding.
    text_args = {
        'fontsize': SUBTITLE_FONT_SIZE,
        'horizontalalignment': 'left',
        's': f'{UNITS_TEXT}: {units}',
        'verticalalignment': 'bottom',
        'x': -0.5,
        'y': subtitle_offset,
    }
    return text_args


def _get_legend_kwargs(num_patches: int, n_plots: int, xy_ratio: float):
    # Num legend cols/rows determined experimentally; may change if needed.
    MAX_LEGEND_COLS_1_COL_LAYOUT = 10
    MAX_LEGEND_COLS_2_COL_LAYOUT = 6
    MAX_LEGEND_ROWS = 30
    if _wide_aoi(xy_ratio) or _extra_wide_aoi(xy_ratio):
        bbox_to_anchor = (-0.01, 0)
        if n_plots == 1 or _extra_wide_aoi(xy_ratio):
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


def get_categorical_colors(num_colors):
    cmap = matplotlib.colormaps[COLORMAPS['nominal']]
    # If > 20 colors needed, generate colormap to override default.
    if num_colors > 20:
        # Values of pastel_factor and rng have been chosen specifically
        # for Carbon (Willamette) sample data. If/when we create a
        # report using sample data that is ill-suited to the color
        # palette generated with these values, we will take a different
        # approach to customizing color palettes.
        cmap = ListedColormap(
            distinctipy.get_colors(
                num_colors, pastel_factor=0.6, rng=0, n_attempts=10))
    colors = cmap(numpy.linspace(0, 1, num_colors))
    return colors


def _configure_special_values(
        cmap: matplotlib.colors.ListedColormap,
        special_values: SpecialValueConfig) -> tuple[
            Literal['neither', 'min', 'max', 'both'],
            list[float],
            list[str],
            list[tuple[int, float, Literal['top', 'bottom']]]]:
    """Config. colormap extensions and return colorbar label/tick settings."""
    lower_threshold, upper_threshold = special_values.thresholds
    lower_label, upper_label = special_values.labels
    lower_color, upper_color = special_values.colors

    extend = 'neither'
    thresholds = []
    labels = []
    text_specs = []

    if lower_threshold is not None:
        cmap.set_under(lower_color)
        extend = 'min'
        thresholds.append(lower_threshold)
        labels.append(lower_label)
        text_specs.append((0, -0.05, 'top'))

    if upper_threshold is not None:
        cmap.set_over(upper_color)
        extend = 'max' if extend == 'neither' else 'both'
        thresholds.append(upper_threshold)
        labels.append(upper_label)
        text_specs.append((0, 1.05, 'bottom'))

    return extend, thresholds, labels, text_specs


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

    # To ensure that plots in same group are the same size, subtitle_offset
    # padding is needed if _any_ raster in list has units (see InVEST #2471)
    any_raster_has_units = any(
        [_get_raster_units(r.raster_path) for r in raster_list])

    for ax, config in zip(axs.flatten(), raster_list):
        raster_path = config.raster_path
        dtype = config.datatype
        transform = config.transform
        resample_alg = RESAMPLE_ALGS[dtype]
        arr, resampled = _read_masked_array(raster_path, resample_alg)
        imshow_kwargs = {}
        colorbar_kwargs = {}
        imshow_kwargs['interpolation'] = 'none'
        cmap = config.colormap
        vmin = vmax = None
        if config.special_values:
            vmin, vmax = config.special_values.thresholds
        if dtype == 'divergent':
            if transform == 'log':
                if None not in [vmin, vmax] and abs(vmin) != vmax:
                    raise UserWarning(
                        "To ensure that 0 falls at the logical breakpoint of "
                        "the divergent color ramp, vmax should equal "
                        f"abs(vmin). Actual vmin: {vmin}, vmax: {vmax}")
                transform = matplotlib.colors.SymLogNorm(
                    linthresh=0.03, vmin=vmin, vmax=vmax)
            else:
                transform = matplotlib.colors.TwoSlopeNorm(
                    vmin=vmin, vcenter=0, vmax=vmax)
        elif dtype == 'continuous':
            if transform == 'log':
                transform = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                transform = matplotlib.colors.Normalize(
                    vmin=vmin, vmax=vmax)
        if dtype.startswith('binary'):
            transform = matplotlib.colors.NoNorm()
            colorbar_kwargs['ticks'] = [0, 1]
        imshow_kwargs['norm'] = transform

        title_line_width = _get_title_line_width(n_plots, xy_ratio)
        ax.set_title(**_get_title_kwargs(
            config.title, resampled, title_line_width))

        units = _get_raster_units(raster_path)
        if any_raster_has_units:
            # This -0.1 multiplier is a bit of a 'magic number' but seems to work for now.
            subtitle_offset = -0.1 * len(arr)
            # Set ylim top < 0 to add some padding above the plot.
            ylim_args = {
                'bottom': len(arr),  # raster height
                'top': subtitle_offset,
            }
            ax.set_ylim(**ylim_args)
        if units:
            text_kwargs = _get_units_text_kwargs(units, subtitle_offset)
            ax.text(**text_kwargs)

        if dtype == 'nominal':
            # typically a 'nominal' raster would be an int type, but we replaced
            # nodata with nan, so the array is now a float.
            values = list(numpy.unique(arr[~numpy.isnan(arr)]).astype(int))
            num_colors = len(values)
            colors = get_categorical_colors(num_colors)
            cmap = matplotlib.colors.ListedColormap(colors)
            bounds = values + [max(values) + 1]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            imshow_kwargs['norm'] = norm
            mappable = ax.imshow(arr, cmap=cmap, **imshow_kwargs)
            patches = [matplotlib.patches.Patch(
                color=colors[i],
                label=f'{values[i]}') for i in range(len(values))]
            leg = ax.legend(handles=patches, **_get_legend_kwargs(
                len(patches), n_plots, xy_ratio))
            leg.set_in_layout(True)
        else:
            if not config.special_values:
                mappable = ax.imshow(arr, cmap=cmap, **imshow_kwargs)
                fig.colorbar(mappable, ax=ax, **colorbar_kwargs)
            else:
                extend, thresholds, labels, text_specs = \
                    _configure_special_values(cmap, config.special_values)
                mappable = ax.imshow(
                        arr, cmap=cmap,
                        **imshow_kwargs)
                cbar = fig.colorbar(
                    mappable, ax=ax, extend=extend,
                    **colorbar_kwargs)
                # Use clim from mappable to ensure tick filtering matches actual displayed range
                vmin, vmax = mappable.get_clim()
                ticks = cbar.get_ticks()
                # Only keep ticks that are (1) within a tolerance of the data
                # range to avoid showing ticks that are outside range, (2) not
                # close to special value (to avoid overlap between special
                # value and regular tick)
                tick_dif = (ticks[1] - ticks[0])/2 if len(ticks) > 1 else 0.1
                tol = tick_dif/100
                for label, (x, y, va_) in zip(labels, text_specs):
                    cbar.ax.text(x, y, str(label), transform=cbar.ax.transAxes,
                                 va=va_, ha='left')
                ticks = [
                    tick for tick in ticks
                    if (vmin - tol) <= tick <= (vmax + tol)
                    and all(abs(tick - t) >= tick_dif for t in thresholds)
                ]
                ticks.extend(thresholds)
                cbar.set_ticks(sorted(ticks))

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


def plot_raster_facets(tif_list, datatype, transform=None, title_list=None,
                       small_plots=False, colormap=None, supertitle=None):
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
        title_list (list): Optional list of strings to use as subplot titles.
            If ``None``, the raster filename is used as the title.
        small_plots (bool): Defaults to False. If True, the typical number of
            columns calculated for plotting facets will be increased by 1,
            making the plots smaller so more can be viewed side-by-side.
        colormap (str): Optional string name of a registered matplotlib
            colormap or a colormap object to use in place of the default
            derived from the raster datatype.
        supertitle (str): Optional title to use for the entire group of
            raster facets.

    """
    raster_info = pygeoprocessing.get_raster_info(tif_list[0])
    bbox = raster_info['bounding_box']
    n_plots = len(tif_list)
    xy_ratio = _get_aspect_ratio(bbox)
    fig, axes = _figure_subplots(xy_ratio, n_plots, small_plots=small_plots)

    if transform is None:
        transform = 'linear'
    if title_list is None:
        title_list = [os.path.basename(filepath) for filepath in tif_list]
    if len(title_list) != len(tif_list):
        raise ValueError(
            f'length of title_list does not equal length of tif_list \n'
            f'title_list: {title_list} \n'
            f'tif_list: {tif_list}')
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
    cmap = plt.get_cmap(colormap if colormap else COLORMAPS[datatype])
    if datatype == 'divergent':
        if transform == 'log':
            normalizer = matplotlib.colors.SymLogNorm(linthresh=0.03, vmin=vmin, vmax=vmax)
        else:
            normalizer = matplotlib.colors.CenteredNorm(vmin=vmin, vmax=vmax)
    if transform == 'log':
        if numpy.isclose(vmin, 0.0):
            vmin = 1e-6
        normalizer = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalizer = plt.Normalize(vmin=vmin, vmax=vmax)
    for arr, ax, raster_path, title in zip(
            ndarray, axes.flatten(), tif_list, title_list):
        mappable = ax.imshow(arr, cmap=cmap, norm=normalizer)
        # all rasters are identical size; `resampled` will be the same for all
        title_line_width = _get_title_line_width(n_plots, xy_ratio)
        ax.set_title(**_get_title_kwargs(title, resampled, title_line_width, facets=True))

    units = _get_raster_units(tif_list[0])
    legend_label = f"{UNITS_TEXT}: {units}" if units else None
    fig.colorbar(mappable, ax=axes.ravel().tolist(), label=legend_label)

    if supertitle:
        fig.suptitle(supertitle, fontsize=TITLE_FONT_SIZE)

    [ax.set_axis_off() for ax in axes.flatten()]
    return fig


def _count_frequency(counter, block):
    """A reducer for ``pygeoprocessing.raster_reduce``."""
    values, counts = numpy.unique(
        block[~numpy.isnan(block)], return_counts=True)
    return counter + collections.Counter(dict(zip(values, counts)))


def plot_categorical_raster_with_table(raster_path_list: list[str]):
    """Plot one or two categorical rasters and generate an HTML table legend.

    Unique pixel values are determined either from an existing
    Raster Attribute Table or by calculating a frequency table from the array.

    If two rasters are given, it is assumed the pixel values have a shared
    meaning and the tables are joined based on the value column. All rasters
    will share the same colormap.

    Args:
        raster_path_list (list[str]): list of filepaths to categorical rasters

    Returns:
        A tuple containing:
            - A string representing a base64-encoded PNG
            - An html table to use as the legend for the image

    """
    if not isinstance(raster_path_list, list):
        raise TypeError('Expected `raster_path_list` to be a list of filepaths')
    if len(raster_path_list) > 2:
        raise ValueError('`raster_path_list` cannot include more than 2 items.')
    lulc_rat_html = None
    value_col_name = None
    rat_list = []
    for raster_path in raster_path_list:
        rat_list.append(_get_raster_attribute_table(raster_path))
    if None not in rat_list:    
        rat_df_list = []
        rat_value_names = set()
        for rat in rat_list:
            value_col = None
            for col in rat.columns:
                if col.usage == 'MinMax':
                    # GeoMetaMaker assigned the MinMax usage. GDAL uses MinMax to
                    # denote that the value is a single pixel value rather than a range.
                    value_col = col.name
            rat_value_names.add(value_col)
            df = pandas.DataFrame(rat.rows)
            # Some RATs include color tables. This is not useful to display in
            # this context because it will be confusing.
            # Best we can do is guess what the column names could be.
            col_filter = df.filter(['Red', 'Green', 'Blue', 'Alpha', 'Opacity'])
            df = df.drop(col_filter, axis=1)
            rat_df_list.append(df)
        if len(rat_value_names) == 1:
            value_col_name = list(rat_value_names)[0]
        else:
            LOGGER.debug(
                'default raster attribute tables do not match and will be ignored.')

    # There was no RAT, or the RAT value columns were ambiguous
    if value_col_name is None:
        rat_df_list = []
        value_col_name = 'value'
        count_col_name = 'count'
        for raster_path in raster_path_list:
            LOGGER.info(
                f'Calculating frequency table for classes in {raster_path}')
            table = pygeoprocessing.raster_reduce(
                _count_frequency, (raster_path, 1), collections.Counter())
            rat_df_list.append(pandas.DataFrame(
                table.items(), columns=[value_col_name, count_col_name]))
    if len(rat_df_list) == 2:
        # TODO: Support arbitrary length of tables with a reducer that calls
        # merge. The tricky part is assigning appropriate suffixes.
        rat_dataframe = pandas.merge(
            rat_df_list[0], rat_df_list[1],
            on=[value_col_name], how='outer',
            suffixes=['_left', '_right'])
    elif len(rat_df_list) == 1:
        rat_dataframe = rat_df_list[0]

    colors = get_categorical_colors(rat_dataframe.shape[0])
    # Sort values before matching them with colors to ensure the mapping is
    # the same here as in imshow.
    colors_dict = dict(zip(sorted(rat_dataframe[value_col_name]), colors))
    legend_col_name = 'color'  # the color swatch column needs no label
    rat_dataframe.insert(
        0, legend_col_name, [matplotlib.colors.to_hex(c) for c in colors])
    styler = rat_dataframe.style.format(na_rep='').map(
        lambda val: f"background-color: {val}; color: {val}",
        subset=[legend_col_name])
    classes = 'datatable legend-table'
    if rat_dataframe.shape[0] >= 20:
        classes += ' paginate'
    if rat_dataframe.shape[1] >= 5:
        # 3 or 4 columns will be typical (color, value, count, name/desc)
        # But we have no idea how many there might be.
        classes += ' colvis'
    lulc_rat_html = styler.hide(axis='index').to_html(
        table_attributes=f'class="{classes}"')
    
    resample_alg = RESAMPLE_ALGS['nominal']
    raster_info = pygeoprocessing.get_raster_info(raster_path_list[0])
    bbox = raster_info['bounding_box']
    xy_ratio = _get_aspect_ratio(bbox)
    n_plots = len(raster_path_list)
    small_plots = False
    if n_plots == 1:
        # If 1 plot, make an extra column for the html table.
        # If > 1 map, the table will always go on its own row.
        small_plots = True
    fig, axs = _figure_subplots(
        xy_ratio, n_plots=n_plots, small_plots=small_plots)
    imshow_kwargs = {}
    imshow_kwargs['interpolation'] = 'none'

    for ax, raster_path in zip(axs, raster_path_list):
        arr, resampled = _read_masked_array(raster_path, resample_alg)
        ax.set_title(**_get_title_kwargs(
            os.path.basename(raster_path), resampled, 100))

        categories = sorted(colors_dict.keys())
        colors = [colors_dict[cat] for cat in categories]
        colormap = matplotlib.colors.ListedColormap(colors)
        bounds = categories + [max(categories) + 1]
        norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
        imshow_kwargs['norm'] = norm

        ax.imshow(arr, cmap=colormap, **imshow_kwargs)
        ax.set_axis_off()
    return base64_encode(fig), lulc_rat_html


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


def _build_stats_table_row(resource, band):
    row = {}
    for (stat_key, display_name) in STATS_LIST:
        stat_val = band.gdal_metadata.get(stat_key)
        if stat_val is not None:
            row[display_name] = float(stat_val)
        else:
            row[display_name] = UNKNOWN_VAL_TEXT
    (width, height) = (
        resource.data_model.raster_size['width'],
        resource.data_model.raster_size['height'])
    row[COUNT_COL_NAME] = width * height
    row[NODATA_COL_NAME] = band.nodata
    # band.units may be '', which can mean 'unitless', 'unknown', or 'other'
    row[UNITS_COL_NAME] = band.units
    return row


def _get_raster_metadata(filepath):
    try:
        resource = geometamaker_load(f'{filepath}.yml')
    except (FileNotFoundError, ValueError) as err:
        LOGGER.debug(err)
        return None
    if isinstance(resource, geometamaker.models.RasterResource):
        return resource


def _get_raster_attribute_table(filepath):
    rat = None
    resource = _get_raster_metadata(filepath)
    if resource is None:
        # if no metadata already exists, generate it
        resource = geometamaker.describe(filepath)
    if isinstance(resource, geometamaker.models.RasterResource):
        rat = resource.get_rat(1)
    return rat


def _get_raster_units(filepath):
    resource = _get_raster_metadata(filepath)
    return resource.get_band_description(1).units if resource else None


def raster_workspace_summary(file_registry):
    """Create a table of stats for all rasters in a file_registry.

    Metadata docs were already created by invest for files in the workspace.
    """
    raster_summary = {}

    def _summarize_output(item):
        if isinstance(item, collections.abc.Mapping):
            for path in item.values():
                _summarize_output(path)
        else:
            resource = _get_raster_metadata(item)
            band = resource.get_band_description(1) if resource else None
            if band:
                filename = os.path.basename(resource.path)
                raster_summary[filename] = _build_stats_table_row(
                    resource, band)

    for item in file_registry.values():
        _summarize_output(item)

    return pandas.DataFrame(raster_summary).T


def raster_inputs_summary(args_dict, model_spec):
    """Create a table of stats for all rasters in an args_dict."""
    raster_summary = {}

    paths_to_check = [v for v in args_dict.values()
                      if isinstance(v, str) and os.path.isfile(v)]

    paths_to_check.extend(_parse_csv_paths_from_spec(args_dict, model_spec))

    for v in paths_to_check:
        # Generate metadata but do not write to disk because we
        # don't know if we have write permission
        resource = geometamaker.describe(v, compute_stats=True)
        if isinstance(resource, geometamaker.models.RasterResource):
            filename = os.path.basename(resource.path)
            band = resource.get_band_description(1)
            raster_summary[filename] = _build_stats_table_row(
                resource, band)
            # Remove 'Units' column if all units are blank
            if not any(raster_summary[filename][UNITS_COL_NAME]):
                del raster_summary[filename][UNITS_COL_NAME]

    return pandas.DataFrame(raster_summary).T


def _parse_csv_paths_from_spec(args_dict, spec):
    table_map_inputs = []
    for input_ in spec.inputs:
        if isinstance(input_, CSVInput):
            table_map_inputs.extend([
                (input_.id, col.id) for col in input_.columns
                if isinstance(col, SingleBandRasterInput)])

    paths_to_check = []
    for input_id, col_name in table_map_inputs:
        if args_dict.get(input_id):
            df = CSVInput.get_validated_dataframe(
                    spec.get_input(input_id),
                    csv_path=args_dict.get(input_id))
            paths_to_check.extend([
                v for v in df[col_name]
                if isinstance(v, str) and os.path.isfile(v)])

    return paths_to_check
