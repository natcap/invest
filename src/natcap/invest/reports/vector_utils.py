from natcap.invest import gettext
from natcap.invest.spec import format_unit, Output, VectorOutput

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


def get_vector_attr_table_caption(
        vector_spec: VectorOutput,
        fields_to_include: list[str] = []) -> list[str]:
    """Generate caption for a vector attribute table.

    Useful when column names are abbreviated or otherwise unclear.

    Args:
        vector_spec (natcap.invest.spec.VectorOutput): the specification of the
            vector output.
        fields_to_include (list[str], optional): names of fields to include in
            the caption. If no list is provided, all fields defined in the spec
            are included in the caption.

    Returns:
        caption (list[str]): a list of formatted strings suitable for passing
        to the ``natcap.invest.reports`` ``caption`` macro with
        ``definition_list=True``.
    """
    def _get_units_text(field: Output) -> str:
        """Get units string to append to a field description.

        Args:
            field (natcap.invest.spec.Output): the field specification.

        Returns:
            units_text (str): a string in the format ``' (Units: {units})'`` if
                the field has units defined and they are not ``None``; empty
                string otherwise.
        """
        if hasattr(field, 'units') and field.units is not None:
            return f' ({gettext("Units:")} {format_unit(field.units)})'
        return ''

    if len(fields_to_include) > 0:
        fields = [vector_spec.get_field(field_name)
                  for field_name in fields_to_include]
    else:
        fields = vector_spec.fields

    return [f'{field.id}:{field.about}{_get_units_text(field)}'
            for field in fields]
