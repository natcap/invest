# Utils shared by SDR and NDR
# (to be extended to support other similar models, and renamed as appropriate)

import geopandas
import pandas

from natcap.invest import gettext
from natcap.invest.reports.report_constants import TABLE_PAGINATION_THRESHOLD


def generate_results_table_from_vector(
        filepath: str, cols_to_sum: list[str] = []
        ) -> str | tuple[str, str | None]:
    """Generate HTML table—and, optionally, HTML table of totals—from a vector.

    Totals are calculated for every column named in ``cols_to_sum``, if the
    main table contains more than one row. If ``cols_to_sum`` is empty (the
    default), no totals table is generated. If ``cols_to_sum`` is _not_ empty,
    but the main table contains only one row, the totals table is ``None``.

    Args:
        filepath (str): path to vector file (e.g., Shapefile or GeoPackage).
        cols_to_sum (list[str], optional): names of columns to include in totals table.
            Defaults to an empty list.

    Returns:
        html_table_main (str): HTML table containing all data from the vector's
            attribute table.

        html_table_totals (str | None, optional): HTML table containing totals,
            if ``cols_to_sum`` is not empty and the main table contains more
            than one row. If the main table contains only one row, this output
            is ``None``. If ``cols_to_sum`` is empty, this output is omitted.
    """
    vector_df = geopandas.read_file(filepath, engine='fiona')
    vector_df = vector_df.drop(columns=['geometry'])

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
