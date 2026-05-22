import altair


MAP_WIDTH = 450 #pixels

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


def create_aggregate_map(geodataframe, xy_ratio, attribute,
                         colorscheme, title, divergent=False):
    if divergent:
        scale_config = altair.Scale(domainMid=0, scheme=colorscheme)
    else:
        scale_config = altair.Scale(scheme=colorscheme)

    attr_map = altair.Chart(geodataframe).mark_geoshape(
        stroke="white",
        strokeWidth=0.5
    ).project(
        type='identity',
        reflectY=True
    ).encode(
        color=altair.Color(
            attribute,
            scale=scale_config
        ),
        tooltip=[altair.Tooltip(attribute, title=attribute)]
    ).properties(
        width=MAP_WIDTH,
        height=MAP_WIDTH / xy_ratio,
        title=title
    ).configure_legend(**LEGEND_CONFIG)

    return attr_map.to_json()
