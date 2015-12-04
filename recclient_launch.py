"""profile code for recclient"""
import time
import cProfile
import pstats
import shutil
import os
import zipfile
import glob

import natcap.invest.recreation.recmodel_client
import natcap.invest.recreation.recmodel_server

RECMODEL_ARGS = {
    'hostname': 'localhost',
    'port': 42342,
    'raw_csv_point_data_path': r"src\natcap\invest\recreation\photos_2005-2014_odlla.csv",
    #'raw_csv_point_data_path': r"src\natcap\invest\recreation\photos_2013-2014_odlla.csv",
    #'raw_csv_point_data_path': r"src\natcap\invest\recreation\foo.csv",
    'cache_workspace': r"./recserver_cache",
}

RECMODEL = natcap.invest.recreation.recmodel_server.RecModel(
    RECMODEL_ARGS['raw_csv_point_data_path'],
    RECMODEL_ARGS['cache_workspace'])


def main():
    """entry point"""

    recclient_args = {
        'hostname': 'localhost',
        'port': 42342,
        #'aoi_path': r"C:\Users\Rich\Documents\svn_repos\invest-sample-data\Recreation\input\initial\predictors\parks.shp",
        'aoi_path': r"C:\Users\Rich\Dropbox\globalrec_data\grid.shp",
        'workspace_dir': r"./reclient_workspace",
    }

    prof = True
    if os.path.exists(recclient_args['workspace_dir']):
        shutil.rmtree(recclient_args['workspace_dir'])

    if prof:
        cProfile.runctx('recmodel(recclient_args)', locals(), globals(), 'rec_client_stats')
        profile = pstats.Stats('rec_client_stats')
        profile.sort_stats('cumulative').print_stats(10)
        profile.sort_stats('time').print_stats(10)
    else:
        natcap.invest.recreation.recmodel_client.execute(recclient_args)


def recmodel(args):
    #gather shapefile components and zip them
    if not os.path.exists(args['workspace_dir']):
        os.makedirs(args['workspace_dir'])

    basename = os.path.splitext(args['aoi_path'])[0]
    aoi_archive_path = os.path.join(args['workspace_dir'], 'aoi_zipped.zip')
    with zipfile.ZipFile(aoi_archive_path, 'w') as myzip:
        for filename in glob.glob(basename + '.*'):
            print 'archiving ', filename
            myzip.write(filename, os.path.basename(filename))

    #convert shapefile to binary string for serialization
    zip_file_binary = open(aoi_archive_path, 'rb').read()

    #transfer zipped file to server
    start_time = time.time()
    print 'server version is ', RECMODEL.get_version()
    result_zip_file_binary = (
        RECMODEL.calc_user_days_binary(zip_file_binary))
    print 'received result, took %f seconds' % (time.time() - start_time)

    #unpack result
    result_zip_path = os.path.join(args['workspace_dir'], 'pud_result.zip')
    open(result_zip_path, 'wb').write(result_zip_file_binary)
    zipfile.ZipFile(result_zip_path, 'r').extractall(args['workspace_dir'])
    os.remove(result_zip_path)

if __name__ == '__main__':
    main()
