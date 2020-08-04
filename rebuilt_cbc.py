import os
import logging

from natcap.invest.coastal_blue_carbon import coastal_blue_carbon2

logging.basicConfig(level=logging.INFO)
ARGS = {
    'workspace_dir': 'rebuilt_cbc',
    'results_suffix': '',
    'transitions_csv': 'rebuilt_cbc_transitions.csv',
    'landcover_transitions_table': os.path.join(
        'data', 'invest-sample-data', 'CoastalBlueCarbon',
        'outputs_preprocessor', 'transitions_sample.csv'),
    'baseline_lulc_path': os.path.join(
        'data', 'invest-sample-data', 'CoastalBlueCarbon', 'inputs',
        'GBJC_2010_mean_Resample.tif'),
    'baseline_lulc_year': 2010,
    #'analysis_year': 2060,
    'biophysical_table_path': 'rebuilt_cbc_biophysical_table.csv',
    'n_workers': 6,
    'do_economic_analysis': True,  # when True, remove analysis year.
    'do_price_table': True,
    'price_table':
        'data/invest-sample-data/CoastalBlueCarbon/inputs/Price_table_SCC3.csv',
    'discount_rate': 6.0,
}

if __name__ == '__main__':
    coastal_blue_carbon2.execute(ARGS)
