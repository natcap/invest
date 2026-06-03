import geopandas
import pandas

from natcap.invest import gettext
from natcap.invest.reports.report_constants import TABLE_PAGINATION_THRESHOLD
from natcap.invest.spec import format_unit, Output, VectorOutput
from natcap.invest.unit_registry import u

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
                the field has units defined and they are neither ``None`` nor
                ``u.other``; empty string otherwise.
        """
        if (hasattr(field, 'units')
            and field.units is not None
            and field.units != u.other):
                return f' ({gettext("Units:")} {format_unit(field.units)})'
        return ''

    if len(fields_to_include) > 0:
        fields = [vector_spec.get_field(field_name)
                  for field_name in fields_to_include]
    else:
        fields = vector_spec.fields

    return [f'{field.id}:{field.about}{_get_units_text(field)}'
            for field in fields]


def generate_results_table_from_vector(
        filepath: str, target_cols: list[int | str] = [],
        cols_to_sum: list[str] = []
        ) -> str | tuple[str, str | None]:
    """Generate HTML table—and, optionally, HTML table of totals—from a vector.

    Totals are calculated for every column named in ``cols_to_sum``, if the
    main table contains more than one row. If ``cols_to_sum`` is empty (the
    default), no totals table is generated. If ``cols_to_sum`` is _not_ empty,
    but the main table contains only one row, the totals table is ``None``.

    Args:
        filepath (str): path to vector file (e.g., Shapefile or GeoPackage).
        target_cols (list[int | str], optional): indices and/or names of
            columns to include in table. If not specified, all source columns
            are included.
        cols_to_sum (list[str], optional): names of columns to include in
            totals table. Defaults to an empty list.

    Returns:
        html_table_main (str): HTML table containing all data from the vector's
            attribute table.

        html_table_totals (str | None, optional): HTML table containing totals,
            if ``cols_to_sum`` is not empty and the main table contains more
            than one row. If the main table contains only one row, this output
            is ``None``. If ``cols_to_sum`` is empty, this output is omitted.
    """
    vector_df = geopandas.read_file(
        filepath, engine='fiona', ignore_geometry=True)

    if len(target_cols) > 0:
        cols_to_include = [
            vector_df.columns[col] if isinstance(col, int) else col
            for col in target_cols]
        cols_to_drop = list(
            set(vector_df.columns).difference(set(cols_to_include)))
        vector_df = vector_df.drop(columns=cols_to_drop)

    css_classes = ['datatable']
    (num_rows, _) = vector_df.shape
    if num_rows > TABLE_PAGINATION_THRESHOLD:
        css_classes.append('paginate')

    html_table_main = vector_df.to_html(
        index=False, na_rep='', classes=css_classes)

    if len(cols_to_sum) == 0:
        return html_table_main

    html_table_totals = None
    if num_rows > 1:
        totals_df = pandas.DataFrame()
        for col in cols_to_sum:
            totals_df.loc[gettext('Totals'), col] = vector_df.loc[:, col].sum()
        html_table_totals = totals_df.to_html(
            index=True, index_names=True, na_rep='', classes='full-width')

    return (html_table_main, html_table_totals)
