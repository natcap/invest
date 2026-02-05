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

TABLE_PAGINATION_THRESHOLD = 10
