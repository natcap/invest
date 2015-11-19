"""profile code for recserver"""
import cProfile
import pstats

import natcap.invest.recreation.recmodel_server


def main():
    args = {
        'hostname': 'localhost',
        'port': 42342,
        'raw_csv_point_data_path': r"src\natcap\invest\recreation\foo.csv"
    }

    prof = False
    if prof:
        cProfile.runctx('natcap.invest.recreation.recmodel_server.execute(args)', locals(), globals(), 'rec_stats')
        p = pstats.Stats('rec_stats')
        p.sort_stats('cumulative').print_stats(10)
        p.sort_stats('time').print_stats(10)
    else:
        natcap.invest.recreation.recmodel_server.execute(args)

if __name__ == '__main__':
    main()
