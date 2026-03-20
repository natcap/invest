from natcap.invest import gettext

# All `gettext`-wrapped strings defined in this module should be returned from
# a function, _not_ defined on the module object itself, since text defined at
# the module level is not localized unless/until the module is reloaded.
# Wrapping text strings in functions avoids the need to reload the module
# and ensures text is localized properly when it is needed.


def stats_table_note():
    return gettext(
        '"Valid percent" indicates the percent of pixels that are not '
        'nodata. Comparing "valid percent" values across rasters may help '
        'you identify cases of unexpected nodata.'
    )


def raster_group_caption():
    """Get "pre-caption" note about raster resampling."""
    return gettext(
        'If a plot title includes "resampled," that raster was resampled to '
        'a lower resolution for rendering in this report. Full resolution '
        'rasters are available in the output workspace.'
    )


def stream_caption_appendix():
    return gettext(
        ' The stream network may look incomplete at this resolution, and '
        'therefore it may be necessary to view the full-resolution raster '
        'in GIS to assess its accuracy.'
    )


TABLE_PAGINATION_THRESHOLD = 10
