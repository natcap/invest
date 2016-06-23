"""InVEST Recreation Server."""

import subprocess
import Queue
import os
import multiprocessing
import uuid
import zipfile
import glob
import hashlib
import pickle
import time
import threading
import collections
import logging
import StringIO

import Pyro4
import numpy
from osgeo import ogr
from osgeo import osr
import shapely.ops
import shapely.wkt
import shapely.geometry
import shapely.prepared

from ... import invest
from natcap.invest.recreation import out_of_core_quadtree  # pylint: disable=import-error,no-name-in-module
from . import recmodel_client

BLOCKSIZE = 2 ** 21
GLOBAL_MAX_POINTS_PER_NODE = 10000  # Default max points in quadtree to split
POINTS_TO_ADD_PER_STEP = 2 ** 8
GLOBAL_DEPTH = 10
LOCAL_MAX_POINTS_PER_NODE = 50
LOCAL_DEPTH = 8
CSV_ROWS_PER_PARSE = 2 ** 10
LOGGER_TIME_DELAY = 5.0

Pyro4.config.SERIALIZER = 'marshal'  # lets us pass null bytes in strings

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recreation.recmodel_server')


def _try_except_wrapper(mesg):
    """Wrap the function in a try/except to log exception before failing.

    This can be useful in places where multiprocessing crashes for some reason
    or Pyro4 calls crash and need to report back over stdout.

    Parameters:
        mesg (string): printed to log before the exception object

    Returns:
        None
    """
    def try_except_decorator(func):
        """Raw decorator function."""
        def try_except_wrapper(*args, **kwargs):
            """General purpose try/except wrapper."""
            func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except Exception as exc_obj:
                LOGGER.exception("%s\n%s", mesg, str(exc_obj))
                raise
        return try_except_wrapper
    return try_except_decorator


class RecModel(object):
    """Class that manages RPCs for calculating photo user days."""

    @_try_except_wrapper("RecModel construction exited while multiprocessing.")
    def __init__(
            self, raw_csv_filename, min_year, max_year, cache_workspace,
            max_points_per_node=GLOBAL_MAX_POINTS_PER_NODE):
        """Initialize RecModel object.

        Parameters:
            raw_csv_filename (string): path to csv file that contains lines
                with the following pattern:

                id,userid,date/time,lat,lng,err

                example:

                0486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
            min_year (int): minimum year allowed to be queried by user
            max_year (int): maximum year allowed to be queried by user
            cache_workspace (string): path to a writable directory where the
                object can write quadtree data to disk and search for
                pre-computed quadtrees based on the hash of the file at
                `raw_csv_filename`

        Returns:
            None
        """
        initial_bounding_box = [-180, -90, 180, 90]
        if max_year < min_year:
            raise ValueError(
                "max_year is less than min_year, must be greater or "
                "equal to")
        self.qt_pickle_filename = construct_userday_quadtree(
            initial_bounding_box, raw_csv_filename, cache_workspace,
            max_points_per_node)
        self.cache_workspace = cache_workspace
        self.min_year = min_year
        self.max_year = max_year

    def get_valid_year_range(self):
        """Return the min and max year queriable.

        Returns:
            (min_year, max_year)
        """
        return (self.min_year, self.max_year)

    # not static so it can register in Pyro object
    def get_version(self):  # pylint: disable=no-self-use
        """Return the rec model server version.

        This string can be used to uniquely identify the PUD database and
        algorithm for publication in terms of reproducibility.
        """
        return '%s:%s' % (invest.__version__, self.qt_pickle_filename)

    # not static so it can register in Pyro object
    @_try_except_wrapper("exception in fetch_workspace_aoi")
    def fetch_workspace_aoi(self, workspace_id):  # pylint: disable=no-self-use
        """Download the AOI of the workspace specified by workspace_id.

        Searches self.cache_workspace for the workspace specified, zips the
        contents, then returns the result as a binary string.

        Parameters:
            workspace_id (string): unique workspace ID on server to query.

        Returns:
            zip file as a binary string of workspace.
        """
        # make a random workspace name so we can work in parallel
        workspace_path = os.path.join(self.cache_workspace, workspace_id)
        out_zip_file_path = os.path.join(
            workspace_path, str('server_in')+'.zip')
        return open(out_zip_file_path, 'rb').read()

    @_try_except_wrapper("exception in calc_photo_user_days_in_aoi")
    def calc_photo_user_days_in_aoi(
            self, zip_file_binary, date_range, out_vector_filename):
        """Calculate annual average and per monthly average photo user days.

        Parameters:
            zip_file_binary (string): a bytestring that is a zip file of an
                ESRI shapefile.
            date_range (string 2-tuple): a tuple that contains the inclusive
                start and end date formatted as 'YYYY-MM-DD'
            out_vector_filename (string): base filename of output vector

        Returns:
            zip_result: a bytestring of a zipped copy of `zip_file_binary`
                with a "PUD_YR_AVG", and a "PUD_{MON}_AVG" for {MON} in the
                calendar months.
            workspace_id: a string that can be used to uniquely identify this
                run on the server
        """
        # make a random workspace name so we can work in parallel
        while True:
            # although there should never be a uuid4 collision, this loop
            # makes me feel better
            workspace_id = str(uuid.uuid4())
            workspace_path = os.path.join(self.cache_workspace, workspace_id)
            if not os.path.exists(workspace_path):
                os.makedirs(workspace_path)
                break

        # decompress zip
        out_zip_file_filename = os.path.join(
            workspace_path, str('server_in')+'.zip')

        LOGGER.info('decompress zip file AOI')
        with open(out_zip_file_filename, 'wb') as zip_file_disk:
            zip_file_disk.write(zip_file_binary)
        shapefile_archive = zipfile.ZipFile(out_zip_file_filename, 'r')
        shapefile_archive.extractall(workspace_path)
        aoi_path = glob.glob(os.path.join(workspace_path, '*.shp'))[0]

        LOGGER.info('running calc user days on %s', workspace_path)
        numpy_date_range = (
            numpy.datetime64(date_range[0]),
            numpy.datetime64(date_range[1]))
        base_pud_aoi_path, monthly_table_path = (
            self._calc_aggregated_points_in_aoi(
                aoi_path, workspace_path, numpy_date_range,
                out_vector_filename))

        # ZIP and stream the result back
        LOGGER.info('zipping result')
        aoi_pud_archive_path = os.path.join(
            workspace_path, 'aoi_pud_result.zip')
        with zipfile.ZipFile(aoi_pud_archive_path, 'w') as myzip:
            for filename in glob.glob(
                    os.path.splitext(base_pud_aoi_path)[0] + '.*'):
                myzip.write(filename, os.path.basename(filename))
            myzip.write(
                monthly_table_path, os.path.basename(monthly_table_path))
        # return the binary stream
        LOGGER.info(
            'calc user days complete sending binary back on %s',
            workspace_path)
        return open(aoi_pud_archive_path, 'rb').read(), workspace_id

    def _calc_aggregated_points_in_aoi(
            self, aoi_path, workspace_path, date_range, out_vector_filename):
        """Aggregate the PUD in the AOI.

        Parameters:
            aoi_path (string): a path to an OGR compatible vector.
            workspace_path(string): path to a directory where working files
                can be created
            date_range (datetime 2-tuple): a tuple that contains the inclusive
                start and end date
            out_vector_filename (string): base filename of output vector

        Returns:
            a path to an ESRI shapefile copy of `aoi_path` updated with a
            "PUD" field which contains the metric per polygon.
        """
        aoi_vector = ogr.Open(aoi_path)
        # append a _pud to the aoi filename
        out_aoi_pud_path = os.path.join(workspace_path, out_vector_filename)

        # start the workers now, because they have to load a quadtree and
        # it will take some time
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

        # calculate x_min transformed by comparing the x coordinate at both
        # the top and bottom of the aoi extent and taking the minimum
        x_min_y_min, _, _ = to_lat_trans.TransformPoint(
            aoi_extent[0], aoi_extent[2])
        x_min_y_max, _, _ = to_lat_trans.TransformPoint(
            aoi_extent[0], aoi_extent[3])
        x_min = min(x_min_y_min, x_min_y_max)

        # calculate x_max transformed by comparing the x coordinate at both
        # the top and bottom of the aoi extent and taking the maximum
        x_max_y_min, _, _ = to_lat_trans.TransformPoint(
            aoi_extent[1], aoi_extent[2])
        x_max_y_max, _, _ = to_lat_trans.TransformPoint(
            aoi_extent[1], aoi_extent[3])
        x_max = max(x_max_y_min, x_max_y_max)

        # calculate y_min transformed by comparing the y coordinate at both
        # the top and bottom of the aoi extent and taking the minimum
        _, y_min_x_min, _ = to_lat_trans.TransformPoint(
            aoi_extent[0], aoi_extent[2])
        _, y_min_x_max, _ = to_lat_trans.TransformPoint(
            aoi_extent[1], aoi_extent[2])
        y_min = min(y_min_x_min, y_min_x_max)

        # calculate y_max transformed by comparing the y coordinate at both
        # the top and bottom of the aoi extent and taking the maximum
        _, y_max_x_min, _ = to_lat_trans.TransformPoint(
            aoi_extent[0], aoi_extent[3])
        _, y_max_x_max, _ = to_lat_trans.TransformPoint(
            aoi_extent[1], aoi_extent[3])
        y_max = max(y_max_x_min, y_max_x_max)

        global_b_box = [x_min, y_min, x_max, y_max]

        local_b_box = [
            aoi_extent[0], aoi_extent[2], aoi_extent[1], aoi_extent[3]]

        LOGGER.info(
            'querying global quadtree against %s', str(global_b_box))
        local_points = global_qt.get_intersecting_points_in_bounding_box(
            global_b_box)
        LOGGER.info('found %d points', len(local_points))

        local_qt_cache_dir = os.path.join(workspace_path, 'local_qt')
        local_qt_pickle_filename = os.path.join(
            local_qt_cache_dir, 'local_qt.pickle')
        os.mkdir(local_qt_cache_dir)

        LOGGER.info('building local quadtree in bounds %s', str(local_b_box))
        local_qt = out_of_core_quadtree.OutOfCoreQuadTree(
            local_b_box, LOCAL_MAX_POINTS_PER_NODE, LOCAL_DEPTH,
            local_qt_cache_dir, pickle_filename=local_qt_pickle_filename)

        LOGGER.info(
            'building local quadtree with %d points', len(local_points))
        last_time = time.time()
        time_elapsed = None
        for point_list_slice_index in xrange(
                0, len(local_points), POINTS_TO_ADD_PER_STEP):
            time_elapsed = time.time() - last_time
            last_time = recmodel_client.delay_op(
                last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                    '%d out of %d points added to local_qt so far, and '
                    ' n_nodes in qt %d in %.2fs', local_qt.n_points(),
                    len(local_points), local_qt.n_nodes(), time_elapsed))

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
        LOGGER.info('saving local qt to %s', local_qt_pickle_filename)
        local_qt.flush()

        local_quad_tree_shapefile_name = os.path.join(
            local_qt_cache_dir, 'local_qt.shp')

        build_quadtree_shape(
            local_quad_tree_shapefile_name, local_qt, aoi_ref)

        # Start several testing processes
        for _ in xrange(n_polytest_processes):
            polytest_process = multiprocessing.Process(
                target=_calc_poly_pud, args=(
                    local_qt_pickle_filename, aoi_path, date_range,
                    poly_test_queue, pud_poly_feature_queue))
            polytest_process.start()

        # Copy the input shapefile into the designated output folder
        LOGGER.info('Creating a copy of the input shapefile')
        esri_driver = ogr.GetDriverByName('ESRI Shapefile')
        LOGGER.debug(out_aoi_pud_path)
        pud_aoi_vector = esri_driver.CopyDataSource(
            aoi_vector, out_aoi_pud_path)
        pud_aoi_layer = pud_aoi_vector.GetLayer()
        pud_id_suffix_list = [
            'YR_AVG', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
            'SEP', 'OCT', 'NOV', 'DEC']
        for field_suffix in pud_id_suffix_list:
            field_id = 'PUD_%s' % field_suffix
            # delete the field if it already exists
            field_index = pud_aoi_layer.FindFieldIndex(str(field_id), 1)
            if field_index >= 0:
                pud_aoi_layer.DeleteField(field_index)
            pud_aoi_layer.CreateField(ogr.FieldDefn(field_id, ogr.OFTReal))

        last_time = time.time()
        LOGGER.info('testing polygons against quadtree')

        # Load up the test queue with polygons
        for poly_feat in pud_aoi_layer:
            poly_test_queue.put(poly_feat.GetFID())

        # Fill the queue with STOPs for each process
        for _ in xrange(n_polytest_processes):
            poly_test_queue.put('STOP')

        # Read the result until we've seen n_processes_alive
        n_processes_alive = n_polytest_processes
        n_poly_tested = 0

        monthly_table_path = os.path.join(workspace_path, 'monthly_table.csv')
        monthly_table = open(monthly_table_path, 'wb')
        date_range_year = [
            date.tolist().timetuple().tm_year for date in date_range]
        table_headers = [
            '%s-%s' % (year, month) for year in xrange(
                int(date_range_year[0]), int(date_range_year[1])+1)
            for month in xrange(1, 13)]
        monthly_table.write('poly_id,' + ','.join(table_headers) + '\n')

        while True:
            result_tuple = pud_poly_feature_queue.get()
            n_poly_tested += 1
            if result_tuple == 'STOP':
                n_processes_alive -= 1
                if n_processes_alive == 0:
                    break
                continue
            last_time = recmodel_client.delay_op(
                last_time, LOGGER_TIME_DELAY, lambda: LOGGER.info(
                    '%.2f%% of polygons tested', 100 * float(n_poly_tested) /
                    pud_aoi_layer.GetFeatureCount()))
            poly_id, pud_list, pud_monthly_set = result_tuple
            poly_feat = pud_aoi_layer.GetFeature(poly_id)
            for pud_index, pud_id in enumerate(pud_id_suffix_list):
                poly_feat.SetField('PUD_%s' % pud_id, pud_list[pud_index])
            pud_aoi_layer.SetFeature(poly_feat)

            line = '%s,' % poly_id
            line += (
                ",".join(['%s' % len(pud_monthly_set[header])
                          for header in table_headers]))
            line += '\n'  # final newline
            monthly_table.write(line)

        LOGGER.info('done with polygon test, syncing to disk')
        pud_aoi_layer = None
        pud_aoi_vector.SyncToDisk()

        LOGGER.info('returning out shapefile path')
        return out_aoi_pud_path, monthly_table_path


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
        None
    """
    for file_offset, chunk_size in iter(block_offset_size_queue.get, 'STOP'):
        csv_file = open(csv_filepath, 'rb')
        csv_file.seek(file_offset, 0)
        chunk_string = csv_file.read(chunk_size)
        csv_file.close()

        # sample line:
        # 8568090486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
        # this pattern matches the above style of line and only parses valid
        # dates to handle some cases where there are weird dates in the input
        pattern = r"[^,]+,([^,]+),(19|20\d\d-(?:0[1-9]|1[012])-(?:0[1-9]|[12][0-9]|3[01])) [^,]+,([^,]+),([^,]+),[^\n]"  # pylint: disable=line-too-long
        result = numpy.fromregex(
            StringIO.StringIO(chunk_string), pattern,
            [('user', 'S40'), ('date', 'datetime64[D]'), ('lat', 'f4'),
             ('lng', 'f4')])

        def md5hash(user_string):
            """md5hash userid."""
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


def _file_len(file_path):
    """Count lines in file, return -1 if not supported."""
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
        max_points_per_node):
    """Construct a spatial quadtree for fast querying of userday points.

    Parameters:
        initial_bounding_box (list of int):
        raw_photo_csv_table ():
        cache_dir (string): path to a directory that can be used to cache
            the quadtree files on disk
        max_points_per_node(int): maximum number of points to allow per node
            of the quadree.  A larger amount will cause the quadtree to
            subdivide.

    Returns:
        None
    """
    LOGGER.info('hashing input file')
    start_time = time.time()
    LOGGER.info(raw_photo_csv_table)
    csv_hash = _hashfile(raw_photo_csv_table, fast_hash=True)

    ooc_qt_picklefilename = os.path.join(cache_dir, csv_hash + '.pickle')
    if os.path.isfile(ooc_qt_picklefilename):
        return ooc_qt_picklefilename
    else:
        LOGGER.info(
            '%s not found, constructing quadtree', ooc_qt_picklefilename)
        LOGGER.info('counting lines in input file')
        total_lines = _file_len(raw_photo_csv_table)
        LOGGER.info('%d lines', total_lines)
        ooc_qt = out_of_core_quadtree.OutOfCoreQuadTree(
            initial_bounding_box, max_points_per_node, GLOBAL_DEPTH,
            cache_dir, pickle_filename=ooc_qt_picklefilename)

        n_parse_processes = multiprocessing.cpu_count() - 1
        if n_parse_processes < 1:
            n_parse_processes = 1

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

        # save quadtree to disk
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

    LOGGER.info('took %f seconds', (time.time() - start_time))
    return ooc_qt_picklefilename


def build_quadtree_shape(
        quad_tree_shapefile_path, quadtree, spatial_reference):
    """Generate a vector of the quadtree geometry.

    Parameters:
        quad_tree_shapefile_path (string): path to save the vector
        quadtree (out_of_core_quadtree.OutOfCoreQuadTree): quadtree
            data structure
        spatial_reference (osr.SpatialReference): spatial reference for the
            output vector

    Returns:
        None
    """
    LOGGER.info('updating quadtree shape at %s', quad_tree_shapefile_path)
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if os.path.isfile(quad_tree_shapefile_path):
        os.remove(quad_tree_shapefile_path)
    datasource = driver.CreateDataSource(quad_tree_shapefile_path)

    polygon_layer = datasource.CreateLayer(
        'quad_tree_shape', spatial_reference, ogr.wkbPolygon)

    # Add a field to identify how deep the node is
    polygon_layer.CreateField(ogr.FieldDefn('n_points', ogr.OFTInteger))
    polygon_layer.CreateField(ogr.FieldDefn('bb_box', ogr.OFTString))
    quadtree.build_node_shapes(polygon_layer)


def _calc_poly_pud(
        local_qt_pickle_path, aoi_path, date_range, poly_test_queue,
        pud_poly_feature_queue):
    """Load a pre-calculated quadtree and test incoming polygons against it.

    Updates polygons with a PUD and send back out on the queue.

    Parameters:
        local_qt_pickle_path (string): path to pickled local quadtree
        aoi_path (string): path to AOI that contains polygon features
        date_range (tuple): numpy.datetime64 tuple indicating inclusive start
            and stop dates
        poly_test_queue (multiprocessing.Queue): queue with incoming
            ogr.Features
        pud_poly_feature_queue (multiprocessing.Queue): queue to put outgoing
            (fid, pud) tuple

    Returns:
        None
    """
    start_time = time.time()
    LOGGER.info('in a _calc_poly_process, loading %s', local_qt_pickle_path)
    local_qt = pickle.load(open(local_qt_pickle_path, 'rb'))
    LOGGER.info('local qt load took %.2fs', time.time() - start_time)

    aoi_vector = ogr.Open(aoi_path)
    aoi_layer = aoi_vector.GetLayer()

    for poly_id in iter(poly_test_queue.get, 'STOP'):
        poly_feat = aoi_layer.GetFeature(poly_id)
        poly_geom = poly_feat.GetGeometryRef()
        poly_wkt = poly_geom.ExportToWkt()
        try:
            shapely_polygon = shapely.wkt.loads(poly_wkt)
        except Exception:  # pylint: disable=broad-except
            # We often get weird corrupt data, this lets us tolerate it
            LOGGER.warn('error parsing poly, skipping')
            continue

        poly_points = local_qt.get_intersecting_points_in_polygon(
            shapely_polygon)
        pud_set = set()
        pud_monthly_set = collections.defaultdict(set)

        for point_datetime, user_hash, _, _ in poly_points:
            if date_range[0] <= point_datetime <= date_range[1]:
                timetuple = point_datetime.tolist().timetuple()

                year = str(timetuple.tm_year)
                month = str(timetuple.tm_mon)
                day = str(timetuple.tm_mday)
                pud_hash = user_hash + '%s-%s-%s' % (year, month, day)
                pud_set.add(pud_hash)
                pud_monthly_set[month].add(pud_hash)
                pud_monthly_set["%s-%s" % (year, month)].add(pud_hash)

        # calculate the number of years and months between the max/min dates
        # index 0 is annual and 1-12 are the months
        pud_averages = [0.0] * 13
        n_years = (
            date_range[1].tolist().timetuple().tm_year -
            date_range[0].tolist().timetuple().tm_year + 1)
        pud_averages[0] = len(pud_set) / float(n_years)
        for month_id in xrange(1, 13):
            monthly_pud_set = pud_monthly_set[str(month_id)]
            pud_averages[month_id] = (
                len(monthly_pud_set) / float(n_years))

        pud_poly_feature_queue.put((poly_id, pud_averages, pud_monthly_set))
    pud_poly_feature_queue.put('STOP')


def execute(args):
    """Launch recreation server and parse/generate quadtree if necessary.

    A call to this function registers a Pyro RPC RecModel entry point given
    the configuration input parameters described below.

    There are many methods to launch a server, including at a Linux command
    line as shown:

    nohup python -u -c "import natcap.invest.recreation.recmodel_server;
        args={'hostname':'$LOCALIP',
              'port':$REC_SERVER_PORT,
              'raw_csv_point_data_path': $POINT_DATA_PATH,
              'max_year': $MAX_YEAR,
              'min_year': $MIN_YEAR,
              'cache_workspace': $CACHE_WORKSPACE_PATH'};
        natcap.invest.recreation.recmodel_server.execute(args)"

    Parameters:
        args['raw_csv_point_data_path'] (string): path to a csv file of the
            format
        args['hostname'] (string): hostname to host Pyro server.
        args['port'] (int/or string representation of int): port number to host
            Pyro entry point.
        args['max_year'] (int): maximum year allowed to be queries by user
        args['min_year'] (int): minimum valid year allowed to be queried by
            user

    Returns:
        Never returns
    """
    daemon = Pyro4.Daemon(args['hostname'], int(args['port']))
    max_points_per_node = GLOBAL_MAX_POINTS_PER_NODE
    if 'max_points_per_node' in args:
        max_points_per_node = args['max_points_per_node']

    uri = daemon.register(
        RecModel(args['raw_csv_point_data_path'], args['min_year'],
                 args['max_year'], args['cache_workspace'],
                 max_points_per_node=max_points_per_node),
        'natcap.invest.recreation')
    LOGGER.info("natcap.invest.recreation ready. Object uri = %s", uri)
    daemon.requestLoop()


def _hashfile(file_path, blocksize=2**20, fast_hash=False):
    """Hash file with memory efficiency as a priority.

    Parameters:
        file_path (string): path to file to hash
        blocksize (int): largest memory block to hold in memory at once in
            bytes
        fast_hash (boolean): if True, hashes the first and last `blocksize` of
            `file_path`, the file_size, file_name, and file_path which takes
            less time on large files for a full hash.  Full hash is done if
            this parameter is true

    Returns:
        sha1 hash of `file_path` if fast_hash is False, otherwise sha1 hash of
        first and last memory blocks, file size, file modified, file name, and
        appends "_fast_hash" to result.
    """
    def _read_file(file_path, file_buffer_queue, blocksize, fast_hash=False):
        """Read one blocksize at a time and adds to the file buffer queue."""
        with open(file_path, 'rb') as file_to_hash:
            if fast_hash:
                # fast hash reads the first and last blocks and uses the
                # modified stamp and filesize
                buf = file_to_hash.read(blocksize)
                file_buffer_queue.put(buf)
                file_size = os.path.getsize(file_path)
                if file_size - blocksize > 0:
                    file_to_hash.seek(file_size - blocksize)
                    buf = file_to_hash.read(blocksize)
                file_buffer_queue.put(buf)
                file_buffer_queue.put(file_path)
                file_buffer_queue.put(str(file_size))
                file_buffer_queue.put(time.ctime(os.path.getmtime(file_path)))
            else:
                buf = file_to_hash.read(blocksize)
                while len(buf) > 0:
                    file_buffer_queue.put(buf)
                    buf = file_to_hash.read(blocksize)
        file_buffer_queue.put('STOP')

    def _hash_blocks(file_buffer_queue):
        """Process file_buffer_queue one buf at a time."""
        hasher = hashlib.sha1()
        for row_buffer in iter(file_buffer_queue.get, "STOP"):
            hasher.update(row_buffer)
        file_buffer_queue.put(hasher.hexdigest()[:16])

    file_buffer_queue = Queue.Queue(100)
    read_file_process = threading.Thread(
        target=_read_file, args=(
            file_path, file_buffer_queue, blocksize, fast_hash))
    read_file_process.start()
    hash_blocks_process = threading.Thread(
        target=_hash_blocks, args=(file_buffer_queue,))
    hash_blocks_process.start()
    read_file_process.join()
    hash_blocks_process.join()
    file_hash = file_buffer_queue.get()
    if fast_hash:
        file_hash += '_fast_hash'
    return file_hash
