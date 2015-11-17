"""profile code for recserver"""
import cProfile
import pstats

import natcap.invest.recreation.recmodel_server


def main():
    args = {
        'hostname': 'localhost',
        'port': 42342,
        'raw_csv_point_data_path': r"C:\Users\Rich\Documents\bitbucket_repos\invest\src\natcap\invest\recreation\foo.csv"
    }

    cProfile.runctx('natcap.invest.recreation.recmodel_server.execute(args)', locals(), globals(), 'rec_stats')
    p = pstats.Stats('rec_stats')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('time').print_stats(10)

if __name__ == '__main__':
    main()
