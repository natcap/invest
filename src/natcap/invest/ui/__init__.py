
from ..utils import mock_import

with mock_import('natcap.invest.ui.model'):
    from .pollination import Pollination

    MODELS = {
        'pollination': Pollination,
    }
