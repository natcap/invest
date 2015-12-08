"""InVEST Recreation Server"""

import subprocess
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
import StringIO

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

import natcap.invest.recreation.out_of_core_quadtree as out_of_core_quadtree

BLOCKSIZE = 2 ** 21
GLOBAL_MAX_POINTS_PER_NODE = 10000  # Default max points in quadtree to split
POINTS_TO_ADD_PER_STEP = 2 ** 8
GLOBAL_DEPTH = 10
LOCAL_MAX_POINTS_PER_NODE = 50
LOCAL_DEPTH = 8
CSV_ROWS_PER_PARSE = 2 ** 10

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

    def calc_aggregated_points_in_aoi(
            self, zip_file_binary, date_range, aggregate_metric):
        """Aggregate the number of unique points in the AOI given a date range
        and temporal metric.

        Parameters:
            zip_file_binary (string): a bytestring that is a zip file of an
                OGR compatable vector.
            date_range (string 2-tuple): a tuple that contains the inclusive
                start and end date in text form as YYYY-MM-DD
            aggregate_metric (string): one of "yearly", "monthly" or "daily"

        Returns:
            a bytestring of a zipped copy of `zip_file_binary` with a "PUD"
            field which contains the metric per polygon."""

        # try/except block so Pyro4 can recieve an exception if there is one
        try:
            allowed_aggregate_metrics = ['yearly', 'monthly', 'daily']
            if aggregate_metric not in allowed_aggregate_metrics:
                raise ValueError(
                    "Unknown aggregate type: '%s', expected one of %s",
                    aggregate_metric, allowed_aggregate_metrics)

            #make a random workspace name so we can work in parallel
            while True:
                workspace_path = os.path.join(
                    'rec_server_workspaces', str(uuid.uuid4()))
                if not os.path.exists(workspace_path):
                    os.makedirs(workspace_path)
                    break

            #decompress zip
            out_zip_file_filename = os.path.join(
                workspace_path, str('server_in')+'.zip')

            LOGGER.info('decompress zip file AOI')
            with open(out_zip_file_filename, 'wb') as zip_file_disk:
                zip_file_disk.write(zip_file_binary)
            shapefile_archive = zipfile.ZipFile(out_zip_file_filename, 'r')
            shapefile_archive.extractall(workspace_path)
            aoi_path = os.path.join(
                workspace_path, os.path.splitext(
                    shapefile_archive.namelist()[0])[0]+'.shp')

            LOGGER.info('running calc user days on %s', workspace_path)
            numpy_date_range = (
                numpy.datetime64(date_range[0]),
                numpy.datetime64(date_range[1]))
            base_pud_aoi_path = self._calc_aggregated_points_in_aoi(
                aoi_path, workspace_path, numpy_date_range, aggregate_metric)

            #ZIP and stream the result back
            print 'zipping result'
            aoi_pud_archive_path = os.path.join(
                workspace_path, 'aoi_pud_result.zip')
            with zipfile.ZipFile(aoi_pud_archive_path, 'w') as myzip:
                for filename in glob.glob(
                        os.path.splitext(base_pud_aoi_path)[0] + '.*'):
                    myzip.write(filename, os.path.basename(filename))
            #return the binary stream
            print (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
                ': calc user days complete sending binary back on ' +
                workspace_path)
            return open(aoi_pud_archive_path, 'rb').read()
        except:
            print 'exception in calc_aggregated_points_in_aoi'
            print '-' * 60
            traceback.print_exc()
            print '-' * 60
            raise

    def _calc_aggregated_points_in_aoi(
            self, aoi_path, workspace_path, date_range, aggregate_metric):
        """Aggregate the number of unique points in the AOI given a date range
        and temporal metric.

        Parameters:
            aoi_path (string): a path to an OGR compatable vector.
            workspace_path(string): path to a directory where working files
                can be created
            date_range (datetime 2-tuple): a tuple that contains the inclusive
                start and end date
            aggregate_metric (string): one of "yearly", "monthly" or "daily"

        Returns:
            a path to an ESRI shapefile copy of `aoi_path` updated with a
            "PUD" field which contains the metric per polygon."""

        aoi_vector = ogr.Open(aoi_path)

        #append a _pud to the aoi filename
        out_aoi_pud_path = os.path.join(
            os.path.dirname(aoi_path),
            os.path.splitext(os.path.basename(aoi_path))[0]+'_pud.shp')

        #start the workers now, because they have to load a quadtree and
        #it will take some time
        poly_test_queue = multiprocessing.Queue()
        pud_poly_feature_queue = multiprocessing.Queue(4)
        n_polytest_processes = multiprocessing.cpu_count()

        global_qt = pickle.load(open(self.qt_pickle_filename, 'rb'))
        aoi_layer = aoi_vector.GetLayer()
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

        local_qt_cache_dir = os.path.join(workspace_path, 'local_qt')
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
            for point_index in xrange(
                    min(len(projected_point_list), POINTS_TO_ADD_PER_STEP)):
                current_point = projected_point_list[point_index]
                # convert to python float types rather than numpy.float32
                lng_coord = float(current_point[2])
                lat_coord = float(current_point[3])
                x_coord, y_coord, _ = from_lat_trans.TransformPoint(
                    lng_coord, lat_coord)
                projected_point_list[point_index] = (
                    current_point[0], current_point[1], x_coord, y_coord)

            local_qt.add_points(
                projected_point_list, 0, len(projected_point_list))
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
                    local_qt_pickle_filename, aoi_path, date_range,
                    aggregate_metric, poly_test_queue,
                    pud_poly_feature_queue))
            polytest_process.start()

        #Copy the input shapefile into the designated output folder
        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'Creating a copy of the input shapefile')
        esri_driver = ogr.GetDriverByName('ESRI Shapefile')
        pud_aoi_vector = esri_driver.CopyDataSource(
            aoi_vector, out_aoi_pud_path)
        pud_aoi_layer = pud_aoi_vector.GetLayer()
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
                LOGGER.info(
                    '%.2f%% of polygons tested', 100 * float(n_poly_tested) /
                    pud_aoi_layer.GetFeatureCount())
                last_time = current_time
            poly_id, poly_pud = result_tuple
            poly_feat = pud_aoi_layer.GetFeature(poly_id)
            poly_feat.SetField('PUD', poly_pud)
            pud_aoi_layer.SetFeature(poly_feat)

        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'done with polygon test, synching to disk')
        pud_aoi_layer = None
        pud_aoi_vector.SyncToDisk()

        print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") +
               'returning out shapefile path')
        return out_aoi_pud_path


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
        csvfile_reader.next()  # skip the header

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
    LOGGER.info('done reading csv from disk')


def _parse_input_csv(
        block_offset_size_queue, csv_filepath, numpy_array_queue):
    """Parse CSV file lines to (datetime64[d], userhash, lat, lng) tuples.

    Parameters:

        block_offset_size_queue (multiprocessing.Queue): contains tuples of
            the form (offset, chunk size) to direct where the file should be
            read from
        numpy_array_queue (multiprocessing.Queue): output queue will have
            paths to files that can be opened with numpy.load and contain
            structured arrays of (datetime, userid, lat, lng) parsed from the
            raw CSV file
        csv_filepath (string): path to csv file to parse from

    Returns:
        None."""

    for file_offset, chunk_size in iter(block_offset_size_queue.get, 'STOP'):
        csv_file = open(csv_filepath, 'rb')
        csv_file.seek(file_offset, 0)
        chunk_string = csv_file.read(chunk_size)
        csv_file.close()

        # sample line:
        # 8568090486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
        # this pattern matches the above style of line and only parses valid
        # dates to handle some cases where there are weird dates in the input
        pattern = r"[^,]+,([^,]+),(19|20\d\d-(?:0[1-9]|1[012])-(?:0[1-9]|[12][0-9]|3[01])) [^,]+,([^,]+),([^,]+),[^\n]"
        result = numpy.fromregex(
            StringIO.StringIO(chunk_string), pattern,
            [('user', 'S40'), ('date', 'datetime64[D]'), ('lat', 'f4'),
             ('lng', 'f4')])

        #year_day = result['date'].astype(int)
        def md5hash(user_string):
            """md5hash userid"""
            return hashlib.md5(user_string).digest()[-4:]

        md5hash_v = numpy.vectorize(md5hash, otypes=['S4'])
        hashes = md5hash_v(result['user'])

        user_day_lng_lat = numpy.empty(
            hashes.size, dtype='datetime64[D],a4,f4,f4')
        user_day_lng_lat['f0'] = result['date']
        user_day_lng_lat['f1'] = hashes
        user_day_lng_lat['f2'] = result['lng']
        user_day_lng_lat['f3'] = result['lat']
        numpy_array_queue.put(user_day_lng_lat)
    numpy_array_queue.put('STOP')


def file_len(file_path):
    """Count lines in file, return -1 if not supported"""
    wc_process = subprocess.Popen(
        ['wc', '-l', file_path], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    result, err = wc_process.communicate()
    if wc_process.returncode != 0:
        LOGGER.warn(err)
        return -1
    return int(result.strip().split()[0])


def construct_userday_quadtree(
        initial_bounding_box, raw_photo_csv_table, cache_dir,
        max_points_per_node=GLOBAL_MAX_POINTS_PER_NODE):

    #see if we've already created the quadtree
    LOGGER.info('hashing input file')
    start_time = time.time()
    LOGGER.info(raw_photo_csv_table)
    csv_hash = file_hash.hashfile(raw_photo_csv_table, fast_hash=True)

    ooc_qt_picklefilename = os.path.join(cache_dir, csv_hash + '.pickle')
    if os.path.isfile(ooc_qt_picklefilename):
        #that's an out of core quadtree
        return ooc_qt_picklefilename
    else:
        LOGGER.info('%s not found, constructing quadtree', ooc_qt_picklefilename)

        LOGGER.info('counting lines in input file')
        total_lines = file_len(raw_photo_csv_table)
        LOGGER.info('%d lines', total_lines)
        #construct a new quadtree
        ooc_qt = out_of_core_quadtree.OutOfCoreQuadTree(
            initial_bounding_box, max_points_per_node, GLOBAL_DEPTH,
            cache_dir, pickle_filename=ooc_qt_picklefilename)

        n_parse_processes = multiprocessing.cpu_count() - 1
        if n_parse_processes < 1:
            n_parse_processes = 1
        #n_parse_processes = 1

        block_offset_size_queue = multiprocessing.Queue(n_parse_processes * 2)
        numpy_array_queue = multiprocessing.Queue(n_parse_processes * 2)

        LOGGER.info('starting parsing processes')
        for _ in xrange(n_parse_processes):
            parse_input_csv_process = multiprocessing.Process(
                target=_parse_input_csv, args=(
                    block_offset_size_queue, raw_photo_csv_table,
                    numpy_array_queue))
            parse_input_csv_process.start()

        # rush through file and determine reasonable offsets and blocks
        def _populate_offset_queue(block_offset_size_queue):
            csv_file = open(raw_photo_csv_table, 'rb')
            csv_file.readline()  # skip the csv header
            while True:
                start = csv_file.tell()
                csv_file.seek(BLOCKSIZE, 1)
                line = csv_file.readline()  # skip to end of line
                bounds = (start, csv_file.tell() - start)
                block_offset_size_queue.put(bounds)
                if not line:
                    break
            csv_file.close()
            for _ in xrange(n_parse_processes):
                block_offset_size_queue.put('STOP')

        LOGGER.info('starting offset queue population thread')
        populate_thread = threading.Thread(
            target=_populate_offset_queue, args=(block_offset_size_queue,))
        populate_thread.start()

        LOGGER.info("add points to the quadtree as they are ready")
        last_time = time.time()
        start_time = last_time
        n_points = 0

        while True:
            point_array = numpy_array_queue.get()
            if (isinstance(point_array, basestring) and
                    point_array == 'STOP'):  # count 'n cpu' STOPs
                n_parse_processes -= 1
                if n_parse_processes == 0:
                    break
                continue

            n_points += len(point_array)
            ooc_qt.add_points(point_array, 0, point_array.size)
            current_time = time.time()
            time_elapsed = current_time - last_time
            if time_elapsed > 5.0:
                LOGGER.info(
                    '%.2f%% complete, %d points skipped, %d nodes in qt in '
                    'only %.2fs', n_points * 100.0 / total_lines,
                    n_points - ooc_qt.n_points(), ooc_qt.n_nodes(),
                    current_time-start_time)
                last_time = time.time()

        #save quadtree to disk
        ooc_qt.flush()
        LOGGER.info(
            '100.00%% complete, %d points skipped, %d nodes in qt in '
            'only %.2fs', n_points - ooc_qt.n_points(), ooc_qt.n_nodes(),
            time.time()-start_time)

        quad_tree_shapefile_name = os.path.join(
            cache_dir, 'quad_tree_shape.shp')

        lat_lng_ref = osr.SpatialReference()
        lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
        LOGGER.info("building quadtree shapefile overview")
        build_quadtree_shape(quad_tree_shapefile_name, ooc_qt, lat_lng_ref)

    print 'took %f seconds' % (time.time() - start_time)
    return ooc_qt_picklefilename


def build_quadtree_shape(
        quad_tree_shapefile_name, quadtree, spatial_reference):
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
        local_qt_pickle_path, aoi_path, date_range, aggregate_metric,
        poly_test_queue, pud_poly_feature_queue):
    """Load a pre-calculated quadtree and test incoming polygons against it.
    Updates polygons with a PUD and send back out on the queue.

    Parameters:
        local_qt_pickle_path (string): path to pickled local quadtree
        aoi_path (string): path to AOI that contains polygon features
        date_range (tuple): numpy.datetime64 tuple indicating inclusive start
            and stop dates
        aggregate_metric (string): one of 'yearly', 'monthly', or 'daily' to
            aggregate multiple points against.
        poly_test_queue (multiprocessing.Queue): queue with incoming
            ogr.Features
        pud_poly_feature_queue (multiprocessing.Queue): queue to put outgoing
            (fid, pud) tuple

        returns nothing"""

    start_time = time.time()
    print 'in a _calc_poly_process, loading %s' % local_qt_pickle_path
    local_qt = pickle.load(open(local_qt_pickle_path, 'rb'))
    print 'local qt load took %.2fs' % (time.time() - start_time)

    aoi_vector = ogr.Open(aoi_path)
    aoi_layer = aoi_vector.GetLayer()

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

        if aggregate_metric == 'yearly':
            agg_fn = lambda x: x.tolist().timetuple().tm_yday
        elif aggregate_metric == 'monthly':
            agg_fn = lambda x: x.tolist().timetuple().tm_mon
        elif aggregate_metric == 'daily':
            agg_fn = lambda x: x.tolist().timetuple().tm_mday

        for point_datetime, user_hash, _, _ in poly_points:
            if date_range[0] <= point_datetime <= date_range[1]:
                pud_hash = user_hash + str(agg_fn(point_datetime))
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
        RecModel(args['raw_csv_point_data_path'], args['cache_workspace']),
        'natcap.invest.recreation')
    LOGGER.info("natcap.invest.recreation ready. Object uri = %s", uri)
    daemon.requestLoop()
