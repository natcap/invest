"""InVEST Recreation Client"""

import os
import glob
import zipfile
import time
import logging

import numpy
import Pyro4

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_client')

#this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'


def execute(args):
    """Connect to remote recreation server and pass AOI

    Parameters:
        args['workspace_dir'] (string): path to workspace directory
        args['aoi_path'] (string): path to AOI vector
        args['hostname'] (string): FQDN to recreation server
        args['port'] (string or int): port on hostname for recreation server
        args['start_date'] (string): start date in form YYYY-MM-DD this date
            is the inclusive lower bound to consider points in the PUD and
            regression
        args['end_date'] (string): end date in form YYYY-MM-DD this date
            is the inclusive upper bound to consider points in the PUD and
            regression

    Returns:
        None."""

    date_range = (
        numpy.datetime64(args['start_date']),
        numpy.datetime64(args['end_date']))

    if not os.path.exists(args['workspace_dir']):
        os.makedirs(args['workspace_dir'])

    recmodel_server = Pyro4.Proxy(
        "PYRO:natcap.invest.recreation@%s:%d" % (
            args['hostname'], int(args['port'])))

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
    LOGGER.info('server version is %s', recmodel_server.get_version())

    aggregating_metric = 'daily'
    result_zip_file_binary = (
        recmodel_server.calc_aggregated_points_in_aoi(
            zip_file_binary, date_range, aggregating_metric))
    LOGGER.info('received result, took %f seconds', time.time() - start_time)

    #unpack result
    result_zip_path = os.path.join(args['workspace_dir'], 'pud_result.zip')
    open(result_zip_path, 'wb').write(result_zip_file_binary)
    zipfile.ZipFile(result_zip_path, 'r').extractall(args['workspace_dir'])
    os.remove(result_zip_path)
