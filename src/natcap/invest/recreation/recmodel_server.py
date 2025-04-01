"""InVEST Recreation Server."""

import collections
import concurrent.futures
import glob
import hashlib
import logging
import multiprocessing
import os
import pickle
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from io import StringIO

import numpy
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
import pygeoprocessing
import Pyro5.api
import shapely.ops
import shapely.wkt
import shapely.geometry
import shapely.prepared

from ... import invest
from .. import utils
from natcap.invest.recreation import out_of_core_quadtree
from . import recmodel_client
from ._utils import _numpy_dumps, _numpy_loads


BLOCKSIZE = 2 ** 21
GLOBAL_MAX_POINTS_PER_NODE = 10000  # Default max points in quadtree to split
POINTS_TO_ADD_PER_STEP = 2 ** 8
GLOBAL_DEPTH = 10
LOCAL_MAX_POINTS_PER_NODE = 50
LOCAL_DEPTH = 8
CSV_ROWS_PER_PARSE = 2 ** 10
LOGGER_TIME_DELAY = 5.0
INITIAL_BOUNDING_BOX = [-180, -90, 180, 90]

# Max points within an AOI bounding box before rejecting the AOI.
# This is configureable in the execute args, but here is a conservative
# default. For a 2-CPU VM, an AOI capturing ~20 million points
# takes ~30 minutes to build the local quadtree and uses ~2GB RAM.
MAX_ALLOWABLE_QUERY = 30_000_000

Pyro5.config.SERIALIZER = 'marshal'  # lets us pass null bytes in strings

LOGGER = logging.getLogger('natcap.invest.recreation.recmodel_server')

# sample line from flickr:
# 8568090486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
# sample line from twitter:
# 1117195232,2023-01-01,-22.908,-43.1975
# this pattern matches the above style of line and only parses valid
# dates to handle some cases where there are weird dates in the input
flickr_pattern = r"[^,]+,([^,]+),(19|20\d\d-(?:0[1-9]|1[012])-(?:0[1-9]|[12][0-9]|3[01])) [^,]+,([^,]+),([^,]+),[^\n]"  # pylint: disable=line-too-long
twittr_pattern = r"([^,]+),(19|20\d\d-(?:0[1-9]|1[012])-(?:0[1-9]|[12][0-9]|3[01])),([^,]+),([^,]+)\n"  # pylint: disable=line-too-long
CSV_PATTERNS = {
    'flickr': flickr_pattern,
    'twitter': twittr_pattern
}


def _try_except_wrapper(mesg):
    """Wrap the function in a try/except to log exception before failing.

    This can be useful in places where multiprocessing crashes for some reason
    or Pyro4 calls crash and need to report back over stdout.

    Args:
        mesg (string): printed to log before the exception object

    Returns:
        None

    """
    def try_except_decorator(func):
        """Raw decorator function."""
        def try_except_wrapper(*args, **kwargs):
            """General purpose try/except wrapper."""
            try:
                return func(*args, **kwargs)
            except Exception as exc_obj:
                LOGGER.exception("%s\n%s", mesg, str(exc_obj))
                raise
        return try_except_wrapper
    return try_except_decorator


@Pyro5.api.expose
class RecManager(object):
    """A class to manage incoming Pyro requests.

    This class's methods will typically be called by a remote client.

    """

    def __init__(self, servers_dict, max_allowable_query):
        """Initialize the manager with references to servers.

        In this context, a "server" is a ``RecModel`` instance.

        Args:
            servers_dict (dict): mapping names to ``RecModel`` instances.
                e.g. {'flickr': flickr_model, 'twitter': twitter_model }
            max_allowable_query (int): the maximum number of points allowed
                within the bounding box of a query.

        """
        self.servers = servers_dict
        self.max_allowable_query = max_allowable_query
        self.client_log_queues = {}

    def get_valid_year_range(self, dataset):
        """Return the min and max year supported for dataset queries.

        Args:
            dataset (str): one of 'flickr' or 'twitter'.

        Returns:
            (min_year, max_year)

        """
        server = self.servers[dataset]
        return server.get_valid_year_range()

    def estimate_aoi_query_size(self, bounding_box, dataset):
        """Count points in quadtree nodes that intersect a bounding box.

        This allows for a quick upper-limit estimate of the number
        of points found within an AOI extent.

        Args:
            bounding_box (list): of the form [xmin, ymin, xmax, ymax]
                where coordinates are WGS84 decimal degrees.
            dataset (str): one of 'flickr' or 'twitter'

        Returns:
            (int, int): (n points, max number of points allowed by this manager)

        """
        LOGGER.info(f'Validating AOI extent: {bounding_box} against {dataset}')
        server = self.servers[dataset]
        n_points = server.n_points_in_intersecting_nodes(bounding_box)
        LOGGER.info(
            f'{n_points} found; max allowed: {self.max_allowable_query}')
        return (n_points, self.max_allowable_query)

    @_try_except_wrapper("calculate_userdays exited while multiprocessing.")
    def calculate_userdays(self, zip_file_binary, aoi_filename, start_year,
                           end_year, dataset_list, client_id):
        """Calculate userdays as requested by a client.

        Submit concurrent.futures jobs to ``RecModel`` servers and wait for
        them to complete. Also manage a logging Queue that is shared by the
        servers.

        Args:
            zip_file_binary (string): a bytestring that is a zip file of a
                GDAL vector.
            aoi_filename (string): the name of the AOI file extracted from
                ``zip_file_binary``
            start_year (int or string): formatted as 'YYYY' or YYYY
            end_year (int or string): formatted as 'YYYY' or YYYY
            dataset_list (list): listing the names of RecModel servers to query
            client_id (string): a unique id sent by the Pyro client.

        Returns:
            (dict): an tuple item for each in ``dataset_list``
                (result_zip_file_binary, workspace_id, server_version)

        """
        log_queue = multiprocessing.Manager().Queue()
        self.client_log_queues[client_id] = log_queue
        results = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_to_label = {}
            for dataset in dataset_list:
                server = self.servers[dataset]
                server.log_queue_map[client_id] = log_queue

                results_filename = f'{server.acronym}_results.gpkg'
                fut = executor.submit(
                    server.calc_user_days_in_aoi,
                    zip_file_binary, aoi_filename,
                    start_year, end_year, results_filename, client_id)
                future_to_label[fut] = server.acronym

            for future in concurrent.futures.as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    # If an exception occurred in the worker, do not raise it
                    # here in the process running the Pyro daemon.
                    results[label] = future.result()
                except Exception:
                    # Exceptions are not pickle-able so return this instead:
                    trace_str = '.'.join(traceback.format_exception(
                        *sys.exc_info()))
                    results[label] = ('ERROR', trace_str)

        LOGGER.info('all user-day calculations complete; sending binary back')

        # With the futures complete, we know that no more messages will be sent
        # to the logging queue. A STOP sentinel conveys that.
        self.client_log_queues[client_id].put('STOP')
        # Also clean up the references held by each server.
        for dataset in dataset_list:
            server = self.servers[dataset]
            del server.log_queue_map[client_id]
        return results

    def log_to_client(self, client_id):
        """Get queued log messages and return them to client.

        Args:
            client_id (string): a unique id sent by the Pyro client.

        Returns:
            (dict): LogRecords cannot be returned with the Pyro5 serializer,
                so they are returned as dicts to be reconstructed.

        """
        if client_id in self.client_log_queues:
            record = self.client_log_queues[client_id].get()
            if record == 'STOP':
                del self.client_log_queues[client_id]
                return None
            return record.__dict__

    @_try_except_wrapper("fetch workspaces exited.")
    def fetch_aoi_workspaces(self, workspace_id, server_id):
        """Download the AOI in the workspace specified by workspace_id.

        Constructs the path using the server's self.local_cache_workspace.

        Args:
            workspace_id (string): identifier of the workspace
            server_id (string): one of ('flickr', 'twitter')

        Returns:
            binary string of a zipfile containing the AOI.

        """
        server = self.servers[server_id]
        zipfile_path = server.find_workspace(workspace_id)
        with open(zipfile_path, 'rb') as out_zipfile:
            zip_binary = out_zipfile.read()
        return zip_binary


class RecModel(object):
    """Class that manages quadtree construction and queries."""

    def __init__(
            self, min_year, max_year, cache_workspace,
            raw_csv_filename=None,
            quadtree_pickle_filename=None,
            max_points_per_node=GLOBAL_MAX_POINTS_PER_NODE,
            max_depth=GLOBAL_DEPTH, dataset_name='flickr'):
        """Initialize RecModel object.

        The object can be initialized either with a path to a CSV file
        containing the raw point data with which to construct a quadtree,
        or with a path to an existing quadtree in the cache_workspace.

        Args:
            raw_csv_filename (string): path to csv file that contains points
                for indexing into a quadtree.
                Must be given if ``quadtree_pickle_filename`` is None.
            quadtree_pickle_filename (string): path to pickle file containing
                a pre-existing quadtree index.
                Must be given if ``raw_csv_filename`` is None.
            min_year (int): minimum year allowed to be queried by user
            max_year (int): maximum year allowed to be queried by user
            cache_workspace (string): path to a writable directory where the
                object can write quadtree data to disk or search for
                pre-computed quadtrees based on the hash of the file at
                `raw_csv_filename`.
            max_points_per_node(int): maximum number of points to allow per
                node of the quadtree. Exceeding this will cause the node to
                subdivide.
            max_depth (int): maximum depth of nodes in the quadtree.
                Once reached, the leaf nodes will not subdivide,
                even if max_points_per_node is exceeded.
            dataset_name (string): one of 'flickr', 'twitter', indicating
                the expected structure of data in the raw csv.

        Returns:
            None

        """
        if max_year < min_year:
            raise ValueError(
                "max_year is less than min_year, must be greater or "
                "equal to")

        if raw_csv_filename:
            LOGGER.info('hashing input file')
            LOGGER.info(raw_csv_filename)
            csv_hash = _hashfile(raw_csv_filename, fast_hash=True)
            ooc_qt_picklefilename = os.path.join(
                cache_workspace, csv_hash + '.pickle')
        elif quadtree_pickle_filename:
            ooc_qt_picklefilename = quadtree_pickle_filename
        else:
            raise ValueError(
                'Both raw_csv_filename and quadtree_pickle_filename'
                'are None. One of these kwargs must be given a value.')

        if os.path.isfile(ooc_qt_picklefilename):
            LOGGER.info(f'{ooc_qt_picklefilename} quadtree already exists')
            if os.path.dirname(ooc_qt_picklefilename) != cache_workspace:
                if not os.path.exists(cache_workspace):
                    os.mkdir(cache_workspace)
                ooc_qt_picklefilename = transplant_quadtree(
                    ooc_qt_picklefilename, cache_workspace)
        else:
            LOGGER.info(
                f'Pickle file {ooc_qt_picklefilename} does not exist, '
                f'constructing quadtree from {raw_csv_filename}')
            if not os.path.exists(raw_csv_filename):
                raise ValueError(f'{raw_csv_filename} does not exist.')
            construct_userday_quadtree(
                INITIAL_BOUNDING_BOX, [raw_csv_filename], dataset_name,
                cache_workspace, ooc_qt_picklefilename,
                max_points_per_node, max_depth)
        self.qt_pickle_filename = ooc_qt_picklefilename
        self.local_cache_workspace = os.path.join(cache_workspace, 'local')
        self.min_year = min_year
        self.max_year = max_year
        self.acronym = 'PUD' if dataset_name == 'flickr' else 'TUD'
        self.log_queue_map = {}

    def get_valid_year_range(self):
        """Return the min and max year queriable.

        Returns:
            (min_year, max_year)

        """
        return (self.min_year, self.max_year)

    def get_version(self):
        """Return the rec model server version.

        This string can be used to uniquely identify the userday database and
        algorithm for publication in terms of reproducibility.
        """
        return '%s:%s' % (invest.__version__, self.qt_pickle_filename)

    def find_workspace(self, workspace_id):
        """Find the AOI of the workspace specified by workspace_id.

        Args:
            workspace_id (string): unique workspace ID on server to query.

        Returns:
            string: path to a zip file

        """
        workspace_path = os.path.join(self.local_cache_workspace, workspace_id)
        out_zip_file_path = os.path.join(
            workspace_path, 'server_in.zip')
        return out_zip_file_path

    def n_points_in_intersecting_nodes(self, bounding_box):
        """Count points in quadtree nodes that intersect a bounding box.

        This allows for a quick upper-limit estimate of the number
        of points found within an AOI extent.

        Args:
            bounding_box (list): of the form [xmin, ymin, xmax, ymax]
                where coordinates are WGS84 decimal degrees.

        Returns:
            int: the number of points in the intersecting nodes.

        """
        with open(self.qt_pickle_filename, 'rb') as qt_pickle:
            global_qt = pickle.load(qt_pickle)
        return global_qt.estimate_points_in_bounding_box(bounding_box)

    def calc_user_days_in_aoi(
            self, zip_file_binary, aoi_filename, start_year, end_year,
            out_vector_filename, client_id=None):
        """Calculate annual average and per monthly average user days.

        Args:
            zip_file_binary (string): a bytestring that is a zip file of a
                GDAL vector.
            aoi_filename (string): the filename for the AOI expected to be
                extracted from ``zip_file_binary``.
            start_year (string | int): formatted as 'YYYY' or YYYY
            end_year (string | int): formatted as 'YYYY' or YYYY
            out_vector_filename (string): base filename of output vector
            client_id (string): a unique id sent by the Pyro client.

        Returns:
            (tuple):
                - bytestring of a zipped copy of `zip_file_binary`
                    with a "PUD_YR_AVG", and a "PUD_{MON}_AVG" for {MON} in the
                    calendar months.
                - string that can be used to uniquely identify this workspace
                    on the server
                - string representing the server version

        """
        # make a random workspace name so we can work in parallel
        workspace_id = str(uuid.uuid4())
        workspace_path = os.path.join(self.local_cache_workspace, workspace_id)
        os.makedirs(workspace_path)

        if client_id:
            # If a Pyro client is calling this function, setup a logger that
            # queues messages for retrieval by the client.
            # Client-relevant logging should use this logger. Other messages
            # can use this module's global LOGGER.
            handler = logging.handlers.QueueHandler(
                self.log_queue_map[client_id])
            logger = logging.getLogger(f'{self.acronym}_{workspace_id}')
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            # Formatting is handled by the client-side logger because
            # log records are passed over the network as dicts rather
            # than LogRecords.
        else:
            logger = LOGGER

        logger.info('decompress zip file AOI')
        out_zip_file_filename = os.path.join(
            workspace_path, str('server_in')+'.zip')
        with open(out_zip_file_filename, 'wb') as zip_file_disk:
            zip_file_disk.write(zip_file_binary)
        aoi_archive = zipfile.ZipFile(out_zip_file_filename, 'r')
        aoi_archive.extractall(workspace_path)
        aoi_archive.close()
        aoi_archive = None
        aoi_path = os.path.join(workspace_path, aoi_filename)

        logger.info('running calc user days on %s', workspace_path)
        base_ud_aoi_path, monthly_table_path = (
            self._calc_aggregated_points_in_aoi(
                aoi_path, workspace_path, start_year, end_year,
                out_vector_filename, logger))

        # ZIP and stream the result back
        logger.info(f'finished {self.acronym}; zipping result')
        aoi_ud_archive_path = os.path.join(
            workspace_path, 'aoi_ud_result.zip')
        with zipfile.ZipFile(aoi_ud_archive_path, 'w') as myzip:
            for filename in glob.glob(
                    os.path.splitext(base_ud_aoi_path)[0] + '.*'):
                myzip.write(filename, os.path.basename(filename))
            myzip.write(
                monthly_table_path, os.path.basename(monthly_table_path))

        # return the binary stream
        with open(aoi_ud_archive_path, 'rb') as aoi_ud_archive:
            return aoi_ud_archive.read(), workspace_id, self.get_version()

    def _calc_aggregated_points_in_aoi(
            self, aoi_path, workspace_path, start_year, end_year,
            out_vector_filename, logger=None):
        """Aggregate the userdays in the AOI.

        If a user wishes to query a RecModel quadtree locally, rather than
        through a Pyro-connected client, this function would be the right
        one to use.

        Args:
            aoi_path (string): a path to an OGR compatible vector.
                It must have a unique ID integer field named 'poly_id'.
            workspace_path(string): path to a directory where working files
                can be created
            start_year (string | int): formatted as 'YYYY' or YYYY
            end_year (string | int): formatted as 'YYYY' or YYYY
            date_range (datetime 2-tuple): a tuple that contains the inclusive
                start and end date
            out_vector_filename (string): base filename of output vector
            logger (logging.Logger): a logger with a QueueHandler for messages
                that are relevant to the client. Only use this if queries
                are being made from a Pyro-connected client.

        Returns:
            - a path to a GDAL vector copy of `aoi_path` updated with annual
              userday counts per polygon, indexed by 'poly_id'
            - a path to a CSV table containing monthly counts of userdays
              per polygon, indexed by 'poly_id'

        """
        # A field expected to be in the AOI vector sent by a client
        poly_id_field = 'poly_id'
        if logger is None:
            logger = LOGGER

        if int(end_year) < int(start_year):
            raise ValueError(
                "Start year must be less than or equal to end year.\n"
                f"start_year: {start_year}\nend_year: {end_year}")

        min_year, max_year = self.get_valid_year_range()
        if not min_year <= int(start_year) <= max_year:
            raise ValueError(
                f"Start year must be between {min_year} and {max_year}.\n"
                f" User input: ({start_year})")
        if not min_year <= int(end_year) <= max_year:
            raise ValueError(
                f"End year must be between {min_year} and {max_year}.\n"
                f" User input: ({end_year})")
        # append jan 1 to start and dec 31 to end
        start_date = numpy.datetime64(str(start_year)+'-01-01')
        end_date = numpy.datetime64(str(end_year)+'-12-31')
        date_range = (start_date, end_date)

        aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)
        out_aoi_ud_path = os.path.join(workspace_path, out_vector_filename)

        # start the workers now, because they have to load a quadtree and
        # it will take some time
        poly_test_queue = multiprocessing.Queue()
        ud_poly_feature_queue = multiprocessing.Queue(4)
        n_processes = multiprocessing.cpu_count()

        LOGGER.info(f'OPENING {self.qt_pickle_filename}')
        with open(self.qt_pickle_filename, 'rb') as qt_pickle:
            global_qt = pickle.load(qt_pickle)

        aoi_layer = aoi_vector.GetLayer()
        # aoi_extent = aoi_layer.GetExtent()
        aoi_ref = aoi_layer.GetSpatialRef()

        # coordinate transformation to convert AOI points to and from lat/lng
        lat_lng_ref = osr.SpatialReference()
        lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
        aoi_info = pygeoprocessing.get_vector_info(aoi_path)
        local_b_box = aoi_info['bounding_box']
        global_b_box = pygeoprocessing.transform_bounding_box(
            local_b_box, aoi_info['projection_wkt'], lat_lng_ref.ExportToWkt())

        from_lat_trans = utils.create_coordinate_transformer(lat_lng_ref, aoi_ref)

        logger.info(
            'querying global quadtree against %s', str(global_b_box))
        local_points = global_qt.get_intersecting_points_in_bounding_box(
            global_b_box)
        logger.info('found %d points', len(local_points))

        local_qt_cache_dir = os.path.join(workspace_path, 'local_qt')
        local_qt_pickle_filename = os.path.join(
            local_qt_cache_dir, 'local_qt.pickle')
        os.mkdir(local_qt_cache_dir)

        logger.info('building local quadtree in bounds %s', str(local_b_box))
        local_qt = out_of_core_quadtree.OutOfCoreQuadTree(
            local_b_box, LOCAL_MAX_POINTS_PER_NODE, LOCAL_DEPTH,
            local_qt_cache_dir, pickle_filename=local_qt_pickle_filename,
            n_workers=n_processes)

        logger.info(
            'building local quadtree with %d points', len(local_points))
        last_time = time.time()
        time_elapsed = None
        for point_list_slice_index in range(
                0, len(local_points), POINTS_TO_ADD_PER_STEP):
            time_elapsed = time.time() - last_time
            last_time = recmodel_client.delay_op(
                last_time, LOGGER_TIME_DELAY, lambda: logger.info(
                    '%d out of %d points added to local_qt so far, and '
                    ' n_nodes in qt %d in %.2fs', local_qt.n_points(),
                    len(local_points), local_qt.n_nodes(), time_elapsed))

            projected_point_list = local_points[
                point_list_slice_index:
                point_list_slice_index+POINTS_TO_ADD_PER_STEP]
            for point_index in range(
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
        polytest_process_list = []
        for _ in range(n_processes):
            polytest_process = multiprocessing.Process(
                target=_calc_poly_ud, args=(
                    local_qt_pickle_filename, aoi_path, date_range,
                    poly_test_queue, ud_poly_feature_queue))
            polytest_process.daemon = True
            polytest_process.start()
            polytest_process_list.append(polytest_process)

        # Copy the input shapefile into the designated output folder
        LOGGER.info('Creating a copy of the input AOI')
        driver = gdal.GetDriverByName('GPKG')
        ud_aoi_vector = driver.CreateCopy(out_aoi_ud_path, aoi_vector)
        ud_aoi_layer = ud_aoi_vector.GetLayer()

        aoi_layer = None
        aoi_vector = None

        ud_id_suffix_list = [
            'YR_AVG', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
            'SEP', 'OCT', 'NOV', 'DEC']
        for field_suffix in ud_id_suffix_list:
            field_id = f'{self.acronym}_{field_suffix}'
            # delete the field if it already exists
            field_index = ud_aoi_layer.FindFieldIndex(str(field_id), 1)
            if field_index >= 0:
                ud_aoi_layer.DeleteField(field_index)
            field_defn = ogr.FieldDefn(field_id, ogr.OFTReal)
            ud_aoi_layer.CreateField(field_defn)

        last_time = time.time()
        logger.info('testing polygons against quadtree')

        # Load up the test queue with polygons
        for poly_feat in ud_aoi_layer:
            poly_test_queue.put(poly_feat.GetFID())

        # Fill the queue with STOPs for each process
        for _ in range(n_processes):
            poly_test_queue.put('STOP')

        # Read the result until we've seen n_processes_alive
        n_processes_alive = n_processes
        n_poly_tested = 0

        monthly_table_path = os.path.join(
            workspace_path, f'{self.acronym}_monthly_table.csv')
        date_range_year = [
            date.tolist().timetuple().tm_year for date in date_range]
        table_headers = [
            '%s-%s' % (year, month) for year in range(
                int(date_range_year[0]), int(date_range_year[1])+1)
            for month in range(1, 13)]
        with open(monthly_table_path, 'w') as monthly_table:
            monthly_table.write('poly_id,' + ','.join(table_headers) + '\n')

            while True:
                result_tuple = ud_poly_feature_queue.get()
                n_poly_tested += 1
                if result_tuple == 'STOP':
                    n_processes_alive -= 1
                    if n_processes_alive == 0:
                        break
                    continue
                last_time = recmodel_client.delay_op(
                    last_time, LOGGER_TIME_DELAY, lambda: logger.info(
                        '%.2f%% of polygons tested', 100 * float(n_poly_tested) /
                        ud_aoi_layer.GetFeatureCount()))
                fid, ud_list, ud_monthly_set = result_tuple
                poly_feat = ud_aoi_layer.GetFeature(fid)
                for ud_index, ud_id in enumerate(ud_id_suffix_list):
                    poly_feat.SetField(f'{self.acronym}_{ud_id}', ud_list[ud_index])
                ud_aoi_layer.SetFeature(poly_feat)

                line = '%s,' % poly_feat.GetField(poly_id_field)
                line += (
                    ",".join(['%s' % len(ud_monthly_set[header])
                              for header in table_headers]))
                line += '\n'  # final newline
                monthly_table.write(line)

        logger.info('done with polygon tests')
        ud_aoi_layer = None
        ud_aoi_vector.FlushCache()
        ud_aoi_vector = None

        for polytest_process in polytest_process_list:
            polytest_process.join()

        return out_aoi_ud_path, monthly_table_path


def _parse_big_input_csv(
        block_offset_size_queue, numpy_array_queue, csv_filepath, dataset_name):
    """Parse CSV file lines to (datetime64[d], userhash, lat, lng) tuples.

    Args:

        block_offset_size_queue (multiprocessing.Queue): contains tuples of
            the form (offset, chunk size) to direct where the file should be
            read from
        numpy_array_queue (multiprocessing.Queue): output queue will have
            paths to files that can be opened with numpy.load and contain
            structured arrays of (datetime, userid, lat, lng) parsed from the
            raw CSV file
        csv_filepath (string): path to csv file to parse from
        dataset_name (string): one of 'flickr', 'twitter', to indicate the
            expected structure of lines in the csv.

    Returns:
        None
    """
    for file_offset, chunk_size in iter(block_offset_size_queue.get, 'STOP'):
        csv_file = open(csv_filepath, 'r')
        csv_file.seek(file_offset, 0)
        chunk_string = csv_file.read(chunk_size)
        csv_file.close()

        result = numpy.fromregex(
            StringIO(chunk_string), CSV_PATTERNS[dataset_name],
            [('user', 'S40'), ('date', 'datetime64[D]'), ('lat', 'f4'),
             ('lng', 'f4')])

        def md5hash(user_string):
            """md5hash userid."""
            return hashlib.md5(user_string).digest()[-4:]

        md5hash_v = numpy.vectorize(md5hash, otypes=['S4'])
        hashes = md5hash_v(result['user'])

        user_day_lng_lat = numpy.empty(
            hashes.size, dtype='datetime64[D],S4,f4,f4')
        user_day_lng_lat['f0'] = result['date']
        user_day_lng_lat['f1'] = hashes
        user_day_lng_lat['f2'] = result['lng']
        user_day_lng_lat['f3'] = result['lat']
        # multiprocessing.Queue pickles the array. Pickling isn't perfect and
        # it modifies the `datetime64` dtype metadata, causing a warning later.
        # To avoid this we dump the array to a string before adding to queue.
        numpy_array_queue.put(_numpy_dumps(user_day_lng_lat))
    numpy_array_queue.put('STOP')


def _parse_small_input_csv_list(
        csv_file_list, numpy_array_queue, dataset_name):
    """Parse CSV file lines to (datetime64[d], userhash, lat, lng) tuples.

    Args:

        csv_file_list (string): list of csv file paths to parse from
        numpy_array_queue (multiprocessing.Queue): output queue will have
            paths to files that can be opened with numpy.load and contain
            structured arrays of (datetime, userid, lat, lng) parsed from the
            raw CSV file
        dataset_name (string): one of 'flickr', 'twitter', to indicate the
            expected structure of lines in the csv.

    Returns:
        None
    """
    for csv_filepath in csv_file_list:
        LOGGER.info(f'parsing {csv_filepath}')
        csv_file = open(csv_filepath, 'r')
        csv_file.readline()  # skip the csv header
        chunk_string = csv_file.read()
        csv_file.close()

        def md5hash(user_string):
            """md5hash userid."""
            return hashlib.md5(user_string).digest()[-4:]

        if chunk_string:
            result = numpy.fromregex(
                StringIO(chunk_string), CSV_PATTERNS[dataset_name],
                [('user', 'S40'), ('date', 'datetime64[D]'), ('lat', 'f4'),
                 ('lng', 'f4')])

            md5hash_v = numpy.vectorize(md5hash, otypes=['S4'])
            hashes = md5hash_v(result['user'])

            user_day_lng_lat = numpy.empty(
                hashes.size, dtype='datetime64[D],a4,f4,f4')
            user_day_lng_lat['f0'] = result['date']
            user_day_lng_lat['f1'] = hashes
            user_day_lng_lat['f2'] = result['lng']
            user_day_lng_lat['f3'] = result['lat']
            # multiprocessing.Queue pickles the array. Pickling isn't perfect
            # and it modifies the `datetime64` dtype metadata, causing a
            # UserWarning later, on save. To avoid this we dump the array
            # to a string before adding to queue.
            numpy_array_queue.put(_numpy_dumps(user_day_lng_lat))
    numpy_array_queue.put('STOP')


def _file_len(file_path_list, estimate=False):
    """Count lines in file, return -1 if not supported."""
    file_list = file_path_list
    multiplier = 1
    if estimate:
        multiplier = 100
        n = int(len(file_path_list) / multiplier)
        file_list = random.sample(file_path_list, n)
    cmdlist = ['wc', '-l'] + file_list
    try:
        # If wc isn't found, Popen raises an exception here
        wc_process = subprocess.Popen(
            cmdlist, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except OSError as e:
        LOGGER.warning(repr(e))
        return -1
    result, err = wc_process.communicate()
    if wc_process.returncode != 0:
        LOGGER.warning(err)
        return -1
    return int(result.strip().split()[-2]) * multiplier


def construct_userday_quadtree(
        initial_bounding_box, raw_csv_file_list, dataset_name, cache_dir,
        ooc_qt_picklefilename, max_points_per_node, max_depth,
        n_workers=None, build_shapefile=True, fast_point_count=False):
    """Construct a spatial quadtree for fast querying of userday points.

    Args:
        initial_bounding_box (list of int):
        raw_csv_file_list (list): list of filepaths of point CSVs
        dataset_name (string): one of 'flickr', 'twitter', indicating the
            expected structure of the csv.
        cache_dir (string): path to a directory that can be used to cache
            the quadtree files on disk
        ooc_qt_picklefilename (string): name for the pickle file quadtree index
            created in the cache_dir.
        max_points_per_node(int): maximum number of points to allow per node
            of the quadtree.  A larger amount will cause the quadtree to
            subdivide.
        max_depth (int): maximum depth of nodes in the quadtree.
            Once reached, the leaf nodes will not subdivide,
            even if max_points_per_node is exceeded.
        n_workers (int): number of cores for multiprocessing.
        build_shapefile (boolean): whether or not to create vector geometries
            representing nodes of the quadtree.
        fast_point_count (boolean): If False, count the number of lines in all
            the csv files. If True, estimate total number of points by counting
            a random sample of files.

    Returns:
        None
    """
    LOGGER.info('counting lines in input file')
    total_lines = _file_len(raw_csv_file_list, estimate=fast_point_count)
    LOGGER.info('%d lines', total_lines)

    # On a single CPU, flushing to disk is the main bottleneck.
    # For large trees (i.e. twitter) more than 80% of wall-time
    # is spent in the flush, while the main process is idle.
    # Devoting 75% of CPUs to flush, setting 1 aside for the
    # main process, leaving the rest for parsing input tables.
    # When only 50% of CPUs devoted to flush, it is still a
    # bottleneck and overall CPU efficiency according to SLURM
    # is 25%, given 8 CPU. Many parser processes are idle most
    # of the time also, as the numpy_array_queue is often full
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    n_flush_processes = int(n_workers * 0.75)
    n_parse_processes = n_workers - n_flush_processes - 1
    if n_parse_processes < 1:
        n_parse_processes = 1
    if n_flush_processes < 1:
        n_flush_processes = 1

    ooc_qt = out_of_core_quadtree.OutOfCoreQuadTree(
        initial_bounding_box, max_points_per_node, max_depth,
        cache_dir, pickle_filename=ooc_qt_picklefilename,
        n_workers=n_flush_processes)

    numpy_array_queue = multiprocessing.Queue(n_parse_processes * 2)
    populate_thread = None

    parse_process_list = []
    if len(raw_csv_file_list) > 1:
        LOGGER.info('starting parsing processes by file')
        if len(raw_csv_file_list) < n_parse_processes:
            n_parse_processes = len(raw_csv_file_list)
        for i in range(n_parse_processes):
            csv_file_list = raw_csv_file_list[i::n_parse_processes]
            parse_input_csv_process = multiprocessing.Process(
                target=_parse_small_input_csv_list,
                args=(csv_file_list, numpy_array_queue, dataset_name))
            parse_input_csv_process.deamon = True
            parse_input_csv_process.start()
            parse_process_list.append(parse_input_csv_process)
    else:
        raw_photo_csv_table = raw_csv_file_list[0]
        block_offset_size_queue = multiprocessing.Queue(n_parse_processes * 2)

        LOGGER.info('starting parsing processes by chunks')
        for _ in range(n_parse_processes):
            parse_input_csv_process = multiprocessing.Process(
                target=_parse_big_input_csv, args=(
                    block_offset_size_queue, numpy_array_queue,
                    raw_photo_csv_table, dataset_name))
            parse_input_csv_process.deamon = True
            parse_input_csv_process.start()
            parse_process_list.append(parse_input_csv_process)

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
            for _ in range(n_parse_processes):
                block_offset_size_queue.put('STOP')

        LOGGER.info('starting offset queue population thread')
        populate_thread = threading.Thread(
            target=_populate_offset_queue, args=(block_offset_size_queue,))
        populate_thread.start()

    LOGGER.info("add points to the quadtree as they are ready")
    last_time = time.time()
    start_time = last_time
    n_points = 0

    LOGGER.info(f'process counter: {n_parse_processes}')
    while True:
        payload = numpy_array_queue.get()
        # if the item is a 'STOP' sentinel, don't load as an array
        if payload == 'STOP':
            n_parse_processes -= 1
            if n_parse_processes == 0:
                break
            continue
        else:
            point_array = _numpy_loads(payload)

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

    if build_shapefile:
        quad_tree_shapefile_name = os.path.join(
            cache_dir, 'quad_tree_shape.shp')

        lat_lng_ref = osr.SpatialReference()
        lat_lng_ref.ImportFromEPSG(4326)  # EPSG 4326 is lat/lng
        LOGGER.info("building quadtree shapefile overview")
        build_quadtree_shape(quad_tree_shapefile_name, ooc_qt, lat_lng_ref)

    if populate_thread:
        populate_thread.join()
    for proc in parse_process_list:
        proc.join()

    LOGGER.info('took %f seconds', (time.time() - start_time))


def build_quadtree_shape(
        quad_tree_shapefile_path, quadtree, spatial_reference):
    """Generate a vector of the quadtree geometry.

    Args:
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


def _calc_poly_ud(
        local_qt_pickle_path, aoi_path, date_range, poly_test_queue,
        ud_poly_feature_queue):
    """Load a pre-calculated quadtree and test incoming polygons against it.

    Updates polygons with a userday count and send back out on the queue.

    Args:
        local_qt_pickle_path (string): path to pickled local quadtree
        aoi_path (string): path to AOI that contains polygon features
        date_range (tuple): numpy.datetime64 tuple indicating inclusive start
            and stop dates
        poly_test_queue (multiprocessing.Queue): queue with incoming
            ogr.Features
        ud_poly_feature_queue (multiprocessing.Queue): queue to put outgoing
            (fid, ud) tuple

    Returns:
        None
    """
    start_time = time.time()
    LOGGER.info('in a _calc_poly_process, loading %s', local_qt_pickle_path)
    with open(local_qt_pickle_path, 'rb') as qt_pickle:
        local_qt = pickle.load(qt_pickle)
    LOGGER.info('local qt load took %.2fs', time.time() - start_time)

    aoi_vector = gdal.OpenEx(aoi_path, gdal.OF_VECTOR)
    if aoi_vector:
        aoi_layer = aoi_vector.GetLayer()
        for poly_id in iter(poly_test_queue.get, 'STOP'):
            try:
                poly_feat = aoi_layer.GetFeature(poly_id)
                poly_geom = poly_feat.GetGeometryRef()
                poly_wkt = poly_geom.ExportToWkt()
            except AttributeError as error:
                LOGGER.warning('skipping feature that raised: %s', str(error))
                continue
            try:
                shapely_polygon = shapely.wkt.loads(poly_wkt)
            except Exception:  # pylint: disable=broad-except
                # We often get weird corrupt data, this lets us tolerate it
                LOGGER.warning('error parsing poly, skipping')
                continue
            poly_points = local_qt.get_intersecting_points_in_polygon(
                shapely_polygon)
            ud_set = set()
            ud_monthly_set = collections.defaultdict(set)
            for point_datetime, user_hash, _, _ in poly_points:
                if date_range[0] <= point_datetime <= date_range[1]:
                    timetuple = point_datetime.tolist().timetuple()

                    year = str(timetuple.tm_year)
                    month = str(timetuple.tm_mon)
                    day = str(timetuple.tm_mday)
                    ud_hash = str(user_hash) + '%s-%s-%s' % (year, month, day)
                    ud_set.add(ud_hash)
                    ud_monthly_set[month].add(ud_hash)
                    ud_monthly_set["%s-%s" % (year, month)].add(ud_hash)

            # calculate the number of years and months between the max/min dates
            # index 0 is annual and 1-12 are the months
            ud_averages = [0.0] * 13
            n_years = (
                date_range[1].tolist().timetuple().tm_year -
                date_range[0].tolist().timetuple().tm_year + 1)
            ud_averages[0] = len(ud_set) / float(n_years)
            for month_id in range(1, 13):
                monthly_ud_set = ud_monthly_set[str(month_id)]
                ud_averages[month_id] = (
                    len(monthly_ud_set) / float(n_years))
            ud_poly_feature_queue.put((poly_id, ud_averages, ud_monthly_set))
    ud_poly_feature_queue.put('STOP')
    aoi_layer = None
    aoi_vector = None


def _hashfile(file_path, blocksize=2**20, fast_hash=False):
    """Hash file with memory efficiency as a priority.

    Args:
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
        with open(file_path, 'r') as file_to_hash:
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
            hasher.update(row_buffer.encode('utf-8'))
        file_buffer_queue.put(hasher.hexdigest()[:16])

    file_buffer_queue = queue.Queue(100)
    read_file_process = threading.Thread(
        target=_read_file, args=(
            file_path, file_buffer_queue, blocksize, fast_hash))
    read_file_process.daemon = True
    read_file_process.start()
    hash_blocks_process = threading.Thread(
        target=_hash_blocks, args=(file_buffer_queue,))
    hash_blocks_process.daemon = True
    hash_blocks_process.start()
    read_file_process.join()
    hash_blocks_process.join()
    file_hash = str(file_buffer_queue.get())
    if fast_hash:
        file_hash += '_fast_hash'
    return file_hash


def transplant_quadtree(qt_pickle_filepath, workspace):
    """Move quadtree filepath references to a local filesystem.

    The quadtree index contains paths that are remnants of the
    filesystem where it was created. Since we're serving it from a
    different filesystem, we overwrite those paths and write a new
    quadtree index file.

    Args:
        qt_pickle_filepath (string): path to a quadtree pickle file
        workspace (string): path to a local directory to write the
            modified quadtree index.

    Returns:
        None
    """
    storage_dir = os.path.dirname(qt_pickle_filepath)
    pickle_filepath = qt_pickle_filepath

    def rename_managers(qt):
        if qt.is_leaf:
            # re-writing from relative to absolute paths
            qt.node_data_manager.manager_filename = f'{qt_pickle_filepath}.db'
            qt.node_data_manager.manager_directory = os.path.dirname(qt_pickle_filepath)
            qt.quad_tree_storage_dir = storage_dir  # this one still relative
        else:
            [rename_managers(qt.nodes[index]) for index in range(4)]
        return qt

    with open(qt_pickle_filepath, 'rb') as qt_pickle:
        global_qt = pickle.load(qt_pickle)

    if global_qt.quad_tree_storage_dir != storage_dir:
        LOGGER.info(
            f'setting quadtree node references to the local filesystem '
            f'{storage_dir}')
        new_qt = rename_managers(global_qt)
        pickle_filepath = os.path.join(
            workspace, f'transplant_{os.path.basename(qt_pickle_filepath)}')
        LOGGER.info(
            f'writing new quadtree index to {pickle_filepath}')
        with open(pickle_filepath, 'wb') as qt_pickle:
            pickle.dump(new_qt, qt_pickle)
    return pickle_filepath


def execute(args):
    """Launch recreation manager, initializing RecModel servers.

    A call to this function registers a Pyro RPC RecManager entry point given
    the configuration input parameters described below.

    The RecManager instantiates RecModel servers, which parse input data
    and construct quadtrees if necessary.

    For a usage example,
    see invest/scripts/recreation_server/launch_recserver.sh
    and invest/scripts/recreation_server/execute_recmodel_server.py

    Example::

        args = {
            'hostname': '',
            'port': 54322,
            'max_allowable_query': 40_000_000,
            'datasets': {
                'flickr': {
                    'raw_csv_point_data_path': 'photos_2005-2017_odlla.csv',
                    'min_year': 2005,
                    'max_year': 2017
                },
                'twitter': {
                    'quadtree_pickle_filename': 'global_twitter_qt.pickle',
                    'min_year': 2012,
                    'max_year': 2022
                }
            }
        }

    Args:
        args['hostname'] (string): hostname to host Pyro server.
        args['port'] (int/or string representation of int): port number to host
            Pyro entry point.
        args['cache_workspace'] (string): Path to a local, writeable, directory.
            Avoid network-mounted volumes.
        args['max_allowable_query'] (int): the maximum number of points allowed
            within the bounding box of a query.
        args['datasets'] (dict): args for instantiating each RecModel server.
            Keys should include 'flickr', 'twitter', or both.

    Returns:
        Never returns

    """
    max_points_per_node = GLOBAL_MAX_POINTS_PER_NODE
    if 'max_points_per_node' in args:
        max_points_per_node = args['max_points_per_node']

    max_allowable_query = MAX_ALLOWABLE_QUERY
    if 'max_allowable_query' in args:
        max_allowable_query = args['max_allowable_query']

    servers = {}
    for dataset, ds_args in args['datasets'].items():
        cache_workspace = os.path.join(args['cache_workspace'], dataset)
        if 'raw_csv_point_data_path' in ds_args and ds_args['raw_csv_point_data_path']:
            servers[dataset] = RecModel(
                ds_args['min_year'], ds_args['max_year'], cache_workspace,
                raw_csv_filename=ds_args['raw_csv_point_data_path'],
                max_points_per_node=max_points_per_node,
                dataset_name=dataset)
        elif 'quadtree_pickle_filename' in ds_args and ds_args['quadtree_pickle_filename']:
            servers[dataset] = RecModel(
                ds_args['min_year'], ds_args['max_year'], cache_workspace,
                quadtree_pickle_filename=ds_args['quadtree_pickle_filename'],
                dataset_name=dataset)
        else:
            raise ValueError(
                f'Either `raw_csv_point_data_path` or `quadtree_pickle_filename`'
                f'must be present in `args[datasets][{dataset}]`')
    if len(servers) == 0:
        raise ValueError('No valid RecModel servers configured in `args`')

    daemon = Pyro5.api.Daemon(args['hostname'], int(args['port']))
    manager = RecManager(servers, max_allowable_query)
    uri = daemon.register(manager, 'natcap.invest.recreation')
    LOGGER.info("natcap.invest.recreation ready. Object uri = %s", uri)
    LOGGER.info(f'accepting queries up to {max_allowable_query} points')
    daemon.requestLoop()
