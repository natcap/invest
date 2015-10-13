"""recmodel client demo"""

import sys
import os
import glob
import zipfile
import time
import urllib

import Pyro4

#this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'

INVEST_RECREATION_REGISTRY_URL = ('http://data.naturalcapitalproject.org/'
                                  'server_registry/invest_recreation_model/')


def execute(args):
    """Execute InVEST recreation model.

    Parameters:
        args['workspace_dir'] (string): path to output directory
        args['aoi_path'] (string): path to an AOI polygon
        args['make_grid'] (boolean): if True, grids the AOI into grid cells,
            if False, result is aggregated by polygon in AOI.
        args['cell_size'] (float): if args['make_grid'] is True, AOI is
            divided into uniform grid cells of this size.  Ignored if
            args['make_grid'] is False.
        args['time_span'] (tuple): not sure yet, something like
            (start_date, end_date)
        args['fit_regression'] (boolean): if True regression is calculated
        args['reponse_variable'] (string): can be either 'pud' or 'ptd' for
            "photo user days" or "photo travel distances"
        args['predictor_dataset_path'] (string): path to a directory that
            contains predictor datasets

    Returns:
        None
    """

    # Connect to remote recreation server
    recmodel_server = Pyro4.Proxy(
        urllib.urlopen(INVEST_RECREATION_REGISTRY_URL).read().rstrip())

    tmp_dir = os.path.join(args['workspace_dir'], 'tmp')
    output_dir = os.path.join(args['workspace_dir'], 'output')

    for directory in [tmp_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #gather shapefile components and zip them
    aoi_shapefile_filename = os.path.basename(args['aoi_path'])
    basename = os.path.splitext(aoi_shapefile_filename)[0]
    aoi_archive_path = os.path.join(tmp_dir, 'to_server.zip')
    with zipfile.ZipFile(aoi_archive_path, 'w') as myzip:
        for filename in glob.glob(basename + '.*'):
            print 'archiving ', filename
            myzip.write(filename, os.path.basename(filename))

    #convert shapefile to binary string for serialization
    zip_file_binary = open(aoi_archive_path, 'rb').read()

    #transfer zipped file to server
    try:
        start_time = time.time()
        print 'server version is ', recmodel_server.get_version()
        result_zip_file_binary = (
            recmodel_server.calc_user_days_binary(zip_file_binary))
        print 'received result, took %f seconds' % (time.time() - start_time)

        #unpack result
        result_zip_uri = os.path.join(output_dir, 'pud_result.zip')
        open(result_zip_uri, 'wb').write(result_zip_file_binary)
        zipfile.ZipFile(result_zip_uri, 'r').extractall(output_dir)
        os.remove(result_zip_uri)
    except Exception:
        #This catches and prints Pyro tracebacks.
        print "Pyro traceback:"
        print "".join(Pyro4.util.getPyroTraceback())
