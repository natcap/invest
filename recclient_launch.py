"""profile code for recclient"""
import cProfile
import pstats
import shutil
import os

import natcap.invest.recreation.recmodel_client

def main():
    """entry point"""

    recclient_args = {
        'hostname': 'localhost',
        'port': 42342,
        'aoi_path': r"C:\Users\Rich\Documents\svn_repos\invest-sample-data\Recreation\input\initial\predictors\parks.shp",
        #'raw_csv_point_data_path': r"src\natcap\invest\recreation\foo.csv",
        'workspace_dir': r"./reclient_workspace",
    }

    prof = True
    if os.path.exists(recclient_args['workspace_dir']):
        shutil.rmtree(recclient_args['workspace_dir'])
    if prof:
        cProfile.runctx('natcap.invest.recreation.recmodel_client.execute(recclient_args)', locals(), globals(), 'rec_client_stats')
        profile = pstats.Stats('rec_client_stats')
        profile.sort_stats('cumulative').print_stats(10)
        profile.sort_stats('time').print_stats(10)
    else:
        natcap.invest.recreation.recmodel_client.execute(recclient_args)

if __name__ == '__main__':
    main()
