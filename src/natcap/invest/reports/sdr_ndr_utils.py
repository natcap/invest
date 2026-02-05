# Utils shared by SDR and NDR
# (to be extended to support other similar models, and renamed as appropriate)

import geopandas
import pandas

from natcap.invest import gettext
from natcap.invest.reports.report_constants import TABLE_PAGINATION_THRESHOLD


def generate_results_table_from_vector(filepath, cols_to_sum):
    vector_df = geopandas.read_file(filepath, engine='fiona')
    vector_df = vector_df.drop(columns=['geometry'])

    css_classes = ['datatable']
    (num_rows, _) = vector_df.shape
    if num_rows > TABLE_PAGINATION_THRESHOLD:
        css_classes.append('paginate')

    html_table_totals = None
    if num_rows > 1:
        totals_df = pandas.DataFrame()
        totals_df.loc[gettext('Totals'), cols_to_sum] = vector_df.sum(axis=0)
        html_table_totals = totals_df.to_html(
            index=True, index_names=True, na_rep='', classes='full-width')

    html_table_main = vector_df.to_html(
        index=False, na_rep='', classes=css_classes)

    return (html_table_main, html_table_totals)


def update_caption_with_stream_map_info(caption: list[str], flow_dir_alg: str):
    stream_map_info_part_1 = gettext((
        'Results were generated using the following flow direction '
        'algorithm:'))
    stream_map_info_part_2 = gettext(
        ('The stream network may look incomplete at  this resolution, and '
         'therefore it may be necessary to view the  full-resolution raster '
         'in GIS to assess its accuracy.'))
    stream_map_info = (
        f' {stream_map_info_part_1} {flow_dir_alg.upper()}. '
        f'{stream_map_info_part_2}')
    return [
        (list_item + stream_map_info
            if list_item.startswith('stream')
            else list_item)
        for list_item in caption]
