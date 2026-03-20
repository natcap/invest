import altair


LEGEND_CONFIG = {
    'labelFontSize': 14,
    'titleFontSize': 14,
    'orient': 'left',
    'gradientLength': 120
}
AXIS_CONFIG = {
    'labelFontSize': 12,
    'titleFontSize': 12,
}


def get_geojson_bbox(geodataframe):
    """Get the bounding box of a GeoDataFrame as a GeoJSON feature.

    Also calculate its aspect ratio. These are useful for cropping
    other layers in altair plots.

    Args:
        geodataframe (geopandas.GeoDataFrame):

    Returns:
        tuple: A 2-tuple containing:
            - extent_feature (dict): A GeoJSON feature representing the bounding
              box of the input GeoDataFrame.
            - xy_ratio (float): The aspect ratio of the bounding box
              (width/height).

    """
    xmin, ymin, xmax, ymax = geodataframe.total_bounds
    xy_ratio = (xmax - xmin) / (ymax - ymin)
    extent_feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon",
                     "coordinates": [[
                         [xmax, ymax],
                         [xmax, ymin],
                         [xmin, ymin],
                         [xmin, ymax],
                         [xmax, ymax]]]},
        "properties": {}
    }
    return extent_feature, xy_ratio
