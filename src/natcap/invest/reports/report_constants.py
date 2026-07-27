from natcap.invest import gettext

STATS_TABLE_NOTE = gettext(
    '"Valid percent" indicates the percent of pixels that are not '
    'nodata. Comparing "valid percent" values across rasters may help '
    'you identify cases of unexpected nodata.'
)

RASTER_GROUP_CAPTION = gettext(
    'If a plot title includes "resampled," that raster was resampled to '
    'a lower resolution for rendering in this report. Full resolution '
    'rasters are available in the output workspace.'
)

STREAM_CAPTION_APPENDIX = gettext(
    ' The stream network may look incomplete at this resolution, and '
    'therefore it may be necessary to view the full-resolution raster '
    'in GIS to assess its accuracy.'
)

LULC_PRE_CAPTION = gettext(
    'LULC values in the legend are listed in order of frequency (most common '
    'first).'
)

TABLE_PAGINATION_THRESHOLD = 10
