"""InVEST Recreation Server"""

import Queue
import os
import multiprocessing
import uuid
import csv
import zipfile
import glob
import datetime
import hashlib
import pickle
import time
import threading
import traceback
import collections
import file_hash
import logging

import Pyro4
from osgeo import ogr
from osgeo import osr
import shapely.ops
import shapely.wkt
import shapely.geometry
import shapely.prepared
import natcap.versioner
import numpy

__version__ = natcap.versioner.get_version('natcap.invest.recmodel_server')

import pyximport
pyximport.install(setup_args={'include_dirs': numpy.get_include()})
import natcap.invest.recreation.out_of_core_quadtree as out_of_core_quadtree

GLOBAL_MAX_POINTS_PER_NODE = 10000  # Default max points in quadtree to split
POINTS_TO_ADD_PER_STEP = GLOBAL_MAX_POINTS_PER_NODE / 2
GLOBAL_DEPTH = 10
LOCAL_MAX_POINTS_PER_NODE = 50
LOCAL_DEPTH = 8
CSV_ROWS_PER_PARSE = 2 ** 8

Pyro4.config.SERIALIZER = 'marshal'  # lets us pass null bytes in strings

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_server')


def _read_file(filename, file_buffer_queue, blocksize):
    """Reads one blocksize at a time and adds to the file buffer queue"""
    with open(filename, 'rb') as file_to_hash:
        buf = file_to_hash.read(blocksize)
        while len(buf) > 0:
            file_buffer_queue.put(buf)
            buf = file_to_hash.read(blocksize)
    file_buffer_queue.put('STOP')


def _hash_blocks(file_buffer_queue):
    """Processes the file_buffer_queue one buf at a time and adds to current
        hash"""
    hasher = hashlib.sha1()
    for row_buffer in iter(file_buffer_queue.get, "STOP"):
        hasher.update(row_buffer)
    file_buffer_queue.put(hasher.hexdigest()[:16])


def hashfile(filename, blocksize=2**20):
    """Memory efficient and threaded function to return a hash since this
        operation is IO bound"""

    file_buffer_queue = Queue.Queue(100)
    read_file_process = threading.Thread(
        target=_read_file, args=(filename, file_buffer_queue, blocksize))
    read_file_process.start()
    hash_blocks_process = threading.Thread(
        target=_hash_blocks, args=(file_buffer_queue,))
    hash_blocks_process.start()
    read_file_process.join()
    hash_blocks_process.join()
    return file_buffer_queue.get()


class RecModel(object):
    """Class that manages RPCs for calculating photo user days"""

    def __init__(self, raw_csv_filename, cache_workspace='./quadtree_cache'):

        initial_bounding_box = [-180, -90, 180, 90]
        try:
            self.qt_pickle_filename = construct_userday_quadtree(
                initial_bounding_box, raw_csv_filename, cache_workspace)
            self.cache_workspace = cache_workspace
        except:
            print "FATAL: construct_userday_quadtree exited while multiprocessing"
            traceback.print_exc()
            raise

    def get_version(self):  # not static so it can register in Pyro object
        """Returns the rec model server version"""
        return __version__

    def calc_user_days_binary(self, zip_file_binary):
        """A try/execept wrapper for _calc_user_days_binary so that RPCs will
            not only get the exception but the server will too"""
        try:
            return self._calc_user_days_binary(zip_file_binary)
        except:
            print 'exception in calc_user_days_binary'
            print '-' * 60
            traceback.print_exc()
            print '-' * 60
            raise

    def _calc_user_days_binary(self, zip_file_binary):
        """Takes an AOI passed in via a binary stream zipped shapefile and
            annotates the file with photo user days per polygon.  Returns the
            result as a binary stream zip with modified shapefile.

            zip_file_binary - a bytestream that can be saved to disk and treated
                as a zip of an AOI as a polygon ogr.Datasource.

            returns bytestring of zipped modified AOI file."""

        #make a random workspace name so we can work in parallel
        while True:
            workspace_uri = os.path.join(
                'rec_server_workspaces', str(uuid.uuid4()))
            if not os.path.exists(workspace_uri):
                os.makedirs(workspace_uri)
                break

        #decompress zip
        out_zip_file_filename = os.path.join(
            workspace_uri, str('server_in')+'.zip')

        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'decompress zip')
        with open(out_zip_file_filename, 'wb') as zip_file_disk:
            zip_file_disk.write(zip_file_binary)
        shapefile_archive = zipfile.ZipFile(out_zip_file_filename, 'r')
        shapefile_archive.extractall(workspace_uri)
        aoi_uri = os.path.join(
            workspace_uri, os.path.splitext(
                shapefile_archive.namelist()[0])[0]+'.shp')

        print (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
            ': running calc user days on ' + workspace_uri)
        base_pud_aoi_uri = self.calc_user_days(aoi_uri, workspace_uri)

        #ZIP and stream the result back
        print 'zipping result'
        aoi_pud_archive_uri = os.path.join(workspace_uri, 'aoi_pud_result.zip')
        with zipfile.ZipFile(aoi_pud_archive_uri, 'w') as myzip:
            for filename in glob.glob(
                    os.path.splitext(base_pud_aoi_uri)[0] + '.*'):
                myzip.write(filename, os.path.basename(filename))
        #return the binary stream
        print (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
            ': calc user days complete sending binary back on ' + workspace_uri)
        return open(aoi_pud_archive_uri, 'rb').read()

    def calc_user_days(self, aoi_filename, workspace_uri):
        """Function to calculate the photo user days given an AOI

            aoi_filename - ogr.Datasource of polygons of interest.
            workspace_uri - a directory in which to write temporary files

            returns a URI to a modified version of aoi_filename with a
                photouserday_field appended to it"""

        aoi_datasource = ogr.Open(aoi_filename)

        #append a _pud to the aoi filename
        out_aoi_pud_filename = os.path.join(
            os.path.dirname(aoi_filename),
            os.path.splitext(os.path.basename(aoi_filename))[0]+'_pud.shp')

        #start the workers now, because they have to load a quadtree and
        #it will take some time
        poly_test_queue = multiprocessing.Queue()
        pud_poly_feature_queue = multiprocessing.Queue(4)
        n_polytest_processes = multiprocessing.cpu_count()

        global_qt = pickle.load(open(self.qt_pickle_filename, 'rb'))
        aoi_layer = aoi_datasource.GetLayer()
        aoi_extent = aoi_layer.GetExtent()
        aoi_ref = aoi_layer.GetSpatialRef()

        # coordinate transformation to convert AOI points to and from lat/lng
        lat_lng_ref = osr.SpatialReference()
        lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng

        to_lat_trans = osr.CoordinateTransformation(aoi_ref, lat_lng_ref)
        from_lat_trans = osr.CoordinateTransformation(lat_lng_ref, aoi_ref)

        x_min, y_min, _ = to_lat_trans.TransformPoint(
            aoi_extent[0], aoi_extent[2])
        x_max, y_max, _ = to_lat_trans.TransformPoint(
            aoi_extent[1], aoi_extent[3])

        global_b_box = [x_min, y_min, x_max, y_max]
        local_b_box = [
            aoi_extent[0],
            aoi_extent[2],
            aoi_extent[1],
            aoi_extent[3]]

        print 'querying global quadtree against %s' % str(global_b_box)
        local_points = global_qt.get_intersecting_points_in_bounding_box(
            global_b_box)
        print 'found %d points' % len(local_points)

        local_qt_cache_dir = os.path.join(workspace_uri, 'local_qt')
        local_qt_pickle_filename = os.path.join(
            local_qt_cache_dir, 'local_qt.pickle')
        os.mkdir(local_qt_cache_dir)

        print 'building local quadtree in bounds %s' % str(local_b_box)
        local_qt = out_of_core_quadtree.OutOfCoreQuadTree(
            local_b_box, LOCAL_MAX_POINTS_PER_NODE, LOCAL_DEPTH,
            local_qt_cache_dir, pickle_filename=local_qt_pickle_filename)

        print 'building local quadtree with %d points' % len(local_points)
        last_time = time.time()
        for point_list_slice_index in xrange(
                0, len(local_points), POINTS_TO_ADD_PER_STEP):
            time_elapsed = time.time() - last_time
            if time_elapsed > 5.0:
                LOGGER.info(
                    '%d out of %d points added to local_qt so far, and n_nodes'
                    ' in qt %d in %.2fs', local_qt.n_points(),
                    len(local_points), local_qt.n_nodes(), time_elapsed)
                last_time = time.time()
            projected_point_list = local_points[
                point_list_slice_index:
                point_list_slice_index+POINTS_TO_ADD_PER_STEP]
            for point_index in xrange(len(projected_point_list)):
                current_point = projected_point_list[point_index]
                lng_coord = current_point[1]
                lat_coord = current_point[2]
                x_coord, y_coord, _ = from_lat_trans.TransformPoint(
                    lng_coord, lat_coord)
                projected_point_list[point_index] = (
                    current_point[0], x_coord, y_coord)

            local_qt.add_points(projected_point_list)
        print 'saving local qt to %s' % local_qt_pickle_filename
        local_qt.flush()

        local_quad_tree_shapefile_name = os.path.join(
            local_qt_cache_dir, 'local_qt.shp')

        build_quadtree_shape(
            local_quad_tree_shapefile_name, local_qt, aoi_ref)

        #Start several testing processes
        for _ in xrange(n_polytest_processes):
            polytest_process = multiprocessing.Process(
                target=_calc_poly_pud, args=(
                    local_qt_pickle_filename, aoi_filename, poly_test_queue,
                    pud_poly_feature_queue))
            polytest_process.start()

        #Copy the input shapefile into the designated output folder
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'Creating a copy of the input shapefile')
        esri_driver = ogr.GetDriverByName('ESRI Shapefile')
        pud_aoi_datasource = esri_driver.CopyDataSource(
            aoi_datasource, out_aoi_pud_filename)
        pud_aoi_layer = pud_aoi_datasource.GetLayer()
        photopud_field = ogr.FieldDefn('PUD', ogr.OFTInteger)
        pud_aoi_layer.CreateField(photopud_field)

        last_time = time.time()
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'testing polygons against quadtree')

        #Load up the test queue with polygons
        for poly_feat in pud_aoi_layer:
            poly_test_queue.put(poly_feat.GetFID())

        #Fill the queue with STOPs for each process
        for _ in xrange(n_polytest_processes):
            poly_test_queue.put('STOP')

        #Read the result until we've seen n_processes_alive
        n_processes_alive = n_polytest_processes
        n_poly_tested = 0
        while True:
            result_tuple = pud_poly_feature_queue.get()
            n_poly_tested += 1
            if result_tuple == 'STOP':
                n_processes_alive -= 1
                if n_processes_alive == 0:
                    break
                continue
            current_time = time.time()
            if current_time - last_time > 5.0:
                print '%.2f%% of polygons tested' % (
                    100 * float(n_poly_tested) / pud_aoi_layer.GetFeatureCount())
                last_time = current_time
            poly_id, poly_pud = result_tuple
            poly_feat = pud_aoi_layer.GetFeature(poly_id)
            poly_feat.SetField('PUD', poly_pud)
            pud_aoi_layer.SetFeature(poly_feat)

        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'done with polygon test, synching to disk')
        pud_aoi_layer = None
        pud_aoi_datasource.SyncToDisk()

        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'returning out shapefile uri')
        return out_aoi_pud_filename


def _read_from_disk_csv(infile_name, raw_file_lines_queue, n_readers):
    """Reads files from the CSV as fast as possible and pushes them down
        the queue

        infile_name - csv input file
        raw_file_lines_queue - will have deques of lines from the raw CSV file
            put in it
        n_readers - number of reader processes for inserting the sentinel

        returns nothing"""


    original_size = os.path.getsize(infile_name)
    bytes_left = original_size
    last_time = time.time()

    with open(infile_name, 'rb') as infile:
        csvfile_reader = csv.reader(infile)
        csvfile_reader.next() #skip the header

        row_deque = collections.deque()
        for row in csvfile_reader:
            bytes_left -= len(','.join(row))
            current_time = time.time()
            if current_time - last_time > 5.0:
                print '%.2f%% of %s read, text row queue size %d' % (
                    (100.0 * (1.0 - (float(bytes_left) / original_size)),
                     infile_name, raw_file_lines_queue.qsize()))
                last_time = current_time
            row_deque.append(row)

            if len(row_deque) > CSV_ROWS_PER_PARSE:
                #print row_deque
                raw_file_lines_queue.put(row_deque)
                row_deque = collections.deque()
    if len(row_deque) > 0:
        raw_file_lines_queue.put(row_deque)
    for _ in xrange(n_readers):
        raw_file_lines_queue.put('STOP')


def _parse_input_csv(raw_file_lines_queue, user_day_tuple_queue):
    """Takes a CSV file lines and dump lists of (userdayhash, lat, lng) tuples

        raw_file_lines_queue - contains lines from the raw CSV file
        user_day_tuple_queue - will have deques of lines from the raw CSV file
            put in it

        returns nothing"""

    result_list = []
    while True:
        row_deque = raw_file_lines_queue.get()
        if row_deque == 'STOP':
            break

        for row in row_deque:
            user_string = row[1]

            #determine the day of the year from the 2006-05-21 00:05:58 format
            try:
                date = datetime.date(
                    int(row[2][:4]), int(row[2][5:7]), int(row[2][8:10]))
                year_day = date.timetuple().tm_yday
            except ValueError:
                # there are malformed dates, which are okay to skip
                continue

            #creates a 4 bit hash of user id and year day.  this identifies a
            #unique user day point in space with only 4 bytes
            user_day_hash = hashlib.md5(user_string+str(year_day)).digest()[-4:]

            #Get the lat/lng
            lat = float(row[3])
            lng = float(row[4])

            result_list.append((user_day_hash, lng, lat))
            if len(result_list) > POINTS_TO_ADD_PER_STEP:
                user_day_tuple_queue.put(result_list)
                result_list = []

    user_day_tuple_queue.put(result_list)
    user_day_tuple_queue.put('STOP')


def construct_userday_quadtree(
        initial_bounding_box, raw_photo_csv_table, cache_dir,
        max_points_per_node=GLOBAL_MAX_POINTS_PER_NODE):

    #see if we've already created the quadtree
    print 'hashing input file'
    start_time = time.time()
    csv_hash = file_hash.hashfile(raw_photo_csv_table, fast_hash=True)

    ooc_qt_picklefilename = os.path.join(cache_dir, csv_hash + '.pickle')
    if os.path.isfile(ooc_qt_picklefilename):
        #that's an out of core quadtree
        return ooc_qt_picklefilename
    else:
        print '%s not found, constructing quadtree' % ooc_qt_picklefilename
        #construct a new quadtree
        ooc_qt = out_of_core_quadtree.OutOfCoreQuadTree(
            initial_bounding_box, max_points_per_node, GLOBAL_DEPTH,
            cache_dir, pickle_filename=ooc_qt_picklefilename)

        n_parse_processes = (multiprocessing.cpu_count() - 2)
        if n_parse_processes < 1:
            n_parse_processes = 1

        raw_file_lines_queue = multiprocessing.Queue(1000)
        read_from_disk_csv_process = multiprocessing.Process(
            target=_read_from_disk_csv, args=(
                raw_photo_csv_table, raw_file_lines_queue,
                n_parse_processes))
        read_from_disk_csv_process.start()

        user_day_tuple_queue = multiprocessing.Queue(4)

        for _ in xrange(n_parse_processes):
            parse_input_csv_process = multiprocessing.Process(
                target=_parse_input_csv, args=(
                    raw_file_lines_queue, user_day_tuple_queue))
            parse_input_csv_process.start()

        #add deques of points to the quadtree as they are ready
        n_points = 0
        last_time = time.time()
        while True:
            userday_tuple_list = user_day_tuple_queue.get()
            if userday_tuple_list == 'STOP':  # count 'n cpu' STOPs
                n_parse_processes -= 1
                if n_parse_processes == 0:
                    break
                continue

            n_points += len(userday_tuple_list)
            time_elapsed = time.time() - last_time
            if time_elapsed > 5.0:
                LOGGER.info(
                    '%d points added to ooc_qt so far, %d points in qt, and '
                    'n_nodes in qt %d in %.2fs', n_points, ooc_qt.n_points(),
                    ooc_qt.n_nodes(), time_elapsed)
                last_time = time.time()
            ooc_qt.add_points(userday_tuple_list)

        #save it to disk
        LOGGER.info(
            '%d points added to ooc_qt final, %d points in qt, and n_nodes in '
            'qt %d', n_points, ooc_qt.n_points(), ooc_qt.n_nodes())
        ooc_qt.flush()

        quad_tree_shapefile_name = os.path.join(
            cache_dir, 'quad_tree_shape.shp')

        lat_lng_ref = osr.SpatialReference()
        lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
        build_quadtree_shape(quad_tree_shapefile_name, ooc_qt, lat_lng_ref)

    print 'took %f seconds' % (time.time() - start_time)
    return ooc_qt_picklefilename


def build_quadtree_shape(quad_tree_shapefile_name, quadtree, spatial_reference):
    """make a shapefile that's some geometry of the quadtree in its cache dir

        quad_tree_shapefile_name - location to save the shapefile
        quadtree - quadtree datastructure
        spatial_reference - spatial reference to make the geometry output to

        returns nothing"""

    print 'updating quadtree shape at %s' % quad_tree_shapefile_name
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if os.path.isfile(quad_tree_shapefile_name):
        os.remove(quad_tree_shapefile_name)
    datasource = driver.CreateDataSource(quad_tree_shapefile_name)

    polygon_layer = datasource.CreateLayer(
        'quad_tree_shape', spatial_reference, ogr.wkbPolygon)

    # Add a field to identify how deep the node is
    polygon_layer.CreateField(ogr.FieldDefn('n_points', ogr.OFTInteger))
    polygon_layer.CreateField(ogr.FieldDefn('bb_box', ogr.OFTString))
    quadtree.build_node_shapes(polygon_layer)


def _calc_poly_pud(
        local_qt_pickle_filename, aoi_filename, poly_test_queue,
        pud_poly_feature_queue):
    """Loads a pre-calculated quadtree and tests incoming polygons against it
        updates those polygons with a PUD and sends back out on the queue

        local_qt_pickle_filename - pickled local quadtree
        poly_test_queue - queue with incoming ogr.Features
        pud_poly_feature_queue - queue to put outgoing (fid, pud) tuple

        returns nothing"""

    start_time = time.time()
    print 'in a _calc_poly_process, loading %s' % local_qt_pickle_filename
    local_qt = pickle.load(open(local_qt_pickle_filename, 'rb'))
    print 'local qt load took %.2fs' % (time.time() - start_time)

    aoi_datasource = ogr.Open(aoi_filename)
    aoi_layer = aoi_datasource.GetLayer()

    for poly_id in iter(poly_test_queue.get, 'STOP'):
        poly_feat = aoi_layer.GetFeature(poly_id)
        poly_geom = poly_feat.GetGeometryRef()
        poly_wkt = poly_geom.ExportToWkt()
        try:
            shapely_polygon = shapely.wkt.loads(poly_wkt)
        except Exception:
            # We often get weird corrupt data, this lets us tolerate it
            LOGGER.warn('error parsing poly, skipping')
            continue

        poly_points = local_qt.get_intersecting_points_in_polygon(
            shapely_polygon)
        pud_set = set()
        for pud_hash, _, _ in poly_points:
            pud_set.add(pud_hash)
        pud_poly_feature_queue.put((poly_id, len(pud_set)))
    pud_poly_feature_queue.put('STOP')


def execute(args):
    """Launch recreation server and parse/generate point lookup structure if
    necessary.  Function registers a Pyro RPC RecModel entry point given the
    configuration input parameters described below.

    Parameters:
        args['raw_csv_point_data_path'] (string): path to a csv file of the
            format
        args['hostname'] (string): hostname to host Pyro server.
        args['port'] (int/or string representation of int): port number to host
            Pyro entry point.

    Returns:
        Never returns"""

    daemon = Pyro4.Daemon(args['hostname'], int(args['port']))
    uri = daemon.register(
        RecModel(args['raw_csv_point_data_path']), 'natcap.invest.recreation')
    LOGGER.info("natcap.invest.recreation ready. Object uri = %s", uri)
    #daemon.requestLoop()
