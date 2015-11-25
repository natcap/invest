"""profile code for recserver"""
import cProfile
import pstats
import shutil
import os


import natcap.invest.recreation.recmodel_server


def main():
    args = {
        'hostname': 'localhost',
        'port': 42342,
        'raw_csv_point_data_path': r"src\natcap\invest\recreation\photos_2005-2014_odlla.csv",
        #'raw_csv_point_data_path': r"src\natcap\invest\recreation\foo.csv",
        'cache_workspace': r"./qt_cache",
    }

    prof = True
    if os.path.exists(args['cache_workspace']):
        shutil.rmtree(args['cache_workspace'])
    if prof:
        cProfile.runctx('natcap.invest.recreation.recmodel_server.execute(args)', locals(), globals(), 'rec_stats')
        profile = pstats.Stats('rec_stats')
        profile.sort_stats('cumulative').print_stats(10)
        profile.sort_stats('time').print_stats(10)
    else:
        natcap.invest.recreation.recmodel_server.execute(args)

    #shutil.rmtree(args['cache_workspace'])


if __name__ == '__main__':
    main()
