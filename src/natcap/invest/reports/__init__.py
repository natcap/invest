import contextlib
import logging

import altair
import geopandas
import jinja2
import matplotlib
import pandas

jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader('natcap.invest.reports', 'templates'),
    autoescape=jinja2.select_autoescape(),
    undefined=jinja2.StrictUndefined
)

MATPLOTLIB_PARAMS = {
    'backend': 'agg',
    # 'legend.fontsize': 'small',
    # 'axes.titlesize': 'small',
    # 'xtick.labelsize': 'small',
    # 'ytick.labelsize': 'small'
}
matplotlib_logger = logging.getLogger('matplotlib')


@contextlib.contextmanager
def configure_libraries():
    """Manage global configuration options from various libraries."""

    # Altair:
    # vegafusion (`altair.data_transformers.enable("vegafusion")`)
    # can perform transformations before embedding
    # data in the chart's spec in order to conserve space.
    # But vegafusion does not support geodataframe data - and geometries are
    # not something we transform anyway. Plus, vegafusion seems incompatible
    # with `disable_max_rows()` and there is no guarantee that the vegafusion
    # transforms will get under the 5000 row default limit. So disabling
    # the row limit is the only option.
    default_altair_state = altair.data_transformers._get_state()
    altair.data_transformers.disable_max_rows()

    # Pandas:
    # Globally set the float format used in DataFrames and resulting HTML tables.
    # G indicates Python "general" format, which limits precision
    # (default: 6 significant digits), drops trailing zeros,
    # and uses scientific notation where appropriate.
    pandas.set_option('display.float_format', '{:G}'.format)

    # Geopandas:
    # Override the default pyogrio because it is less dependable than fiona
    # https://github.com/geopandas/pyogrio/issues/629
    geopandas.options.io_engine = 'fiona'

    # Matplotlib
    matplotlib.rcParams.update(MATPLOTLIB_PARAMS)
    # Debug-level is far too noisy and likely never useful in invest's context
    matplotlib_logger.setLevel(logging.INFO)

    try:
        yield
    finally:
        altair.data_transformers._set_state(default_altair_state)
        pandas.reset_option('display.float_format')
        geopandas.options.io_engine = None
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib_logger.setLevel(logging.NOTSET)
