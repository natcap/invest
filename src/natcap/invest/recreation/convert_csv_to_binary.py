"""To convert large .csv inputs to binary"""

import os
import io
import multiprocessing
import datetime
import struct
import sys
import csv
import time
import hashlib
import zlib
import cProfile
import pstats

import ogr
import osr
import shapely
import shapely.wkt
import shapely.ops
import shapely.prepared
import shapely.speedups

if shapely.speedups.available:
    shapely.speedups.enable()

def lowpriority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)

def process_csv_to_binary(numprocs, infile_name, outfile_name):
    """Sets up a parallel pipeline for reading and parsing the CSV,
        converting to binary data, then writing those data out.

        numprocs - how many processors to allocate to the workers
        infile_name - name of .csv input file

            expects format to be:
                [header line]
                [userid,YYYMMDD,lat,lng]
                ...

        outfile_name - name of .bin output file
            output will be
                [iffiffiff..] where i is the user hash, ff is lat/lng

        returns nothing"""
    #holds the rows of csv parsed text
    #put an upper limit on the queue so we don't run out of memory
    csv_text_row_queue = multiprocessing.Queue(2**10)

    #holds binary output buffers to dump directly to a file
    binary_row_queue = multiprocessing.Queue()

    parse_input_csv_process = multiprocessing.Process(
        target=_parse_input_csv, args=(
            infile_name, csv_text_row_queue, numprocs))
    write_output_binary_process = multiprocessing.Process(
        target=_write_output_binary, args=(
            outfile_name, binary_row_queue, numprocs))
    p_workers = [
        multiprocessing.Process(target=_parse_row_to_binary_row, args=(
            csv_text_row_queue, binary_row_queue))
        for _ in range(numprocs)]

    parse_input_csv_process.start()
    write_output_binary_process.start()
    for p_worker in p_workers:
        p_worker.start()

    parse_input_csv_process.join()
    for index, p_worker in enumerate(p_workers):
        p_worker.join()

    write_output_binary_process.join()

def _parse_input_csv(infile_name, csv_text_row_queue, numprocs):
    """
    The data is then sent over csv_text_row_queue for the workers to do their
    thing.  At the end the input process sends a 'STOP' message for each
    worker.

        infile_name - csv input file
        csv_text_row_queue - output queue to write rows of csv text to
        numprocs - how many worker processes there are

        returns nothing"""
    lowpriority()
    original_size = os.path.getsize(infile_name)
    bytes_left = original_size

    last_time = time.time()
    with open(infile_name, 'rb') as infile:
        csvfile_reader = csv.reader(infile)
        csvfile_reader.next() #skip the header
        row_buffer = []
        for row in csvfile_reader:
            bytes_left -= len(','.join(row))
            current_time = time.time()
            if current_time - last_time > 5.0:
                print '%.2f%% of %s read' % (
                    (100.0 * (1.0 - (float(bytes_left) / original_size)), infile_name))
                last_time = current_time

            row_buffer.append(row)
            #after some experimenting, 2**8 seemed to be the optimal queue size
            #for ramping up the pipeline and minimizing queuing overhead
            if len(row_buffer) > 2**7:
                csv_text_row_queue.put(row_buffer)
                row_buffer = []
        if len(row_buffer) > 0:
            csv_text_row_queue.put(row_buffer)

    for _ in range(numprocs):
        csv_text_row_queue.put("STOP")

def _parse_row_to_binary_row(csv_text_row_queue, binary_row_queue):
    """
    Workers. Consume csv_text_row_queue and hash and pack to tight format

        csv_text_row_queue - a queue of a list of lists, each sublist is
            a parsed row from the original csv
        binary_row_queue - the raw binary output to write

        returns nothing
    """
    lowpriority()
    out_line = ''
    for row_buffer in iter(csv_text_row_queue.get, "STOP"):
        for row in row_buffer:
            #row format is
            #[photo_id,owner_name,date_taken,latitude,longitude,accuracy]
            user_string = row[1]

            #determine the day of the year from the 2006-05-21 00:05:58 format
            try:
                year_day = datetime.datetime.strptime(
                    row[2], '%Y-%m-%d %H:%M:%S').timetuple().tm_yday
            except ValueError as exception:
                # there are malformed dates, which are okay to skip
                continue

            #creates a 4 bit hash of user id and year day.  this identifies a
            #unique user day point in space with only 4 bytes
            user_day_hash = hashlib.md5(user_string+str(year_day)).digest()[-4:]

            #Get the lat/lng
            lat = float(row[3])
            lng = float(row[4])

            #pack in efficient binary format
            out_line += (struct.pack('4c', *user_day_hash) +
                         struct.pack('ff', lat, lng))

            #if the outsize is a healthy size bigger than the IO buffer, put
            #in the queue.  This saves us from having expensive multiprocessing
            #queue overhead. The * 4 was selected after trying a handful of
            #various values
            if len(out_line) > io.DEFAULT_BUFFER_SIZE * 4:
                compressed_line = zlib.compress(out_line)
                compressed_line = (
                    struct.pack('i', len(compressed_line)) + compressed_line)
                binary_row_queue.put(compressed_line)
                out_line = ''

    #if there's any left, flush it
    if len(out_line) > 0:
        compressed_line = zlib.compress(out_line, 9)
        compressed_line = (
            struct.pack('i', len(compressed_line)) + compressed_line)
        binary_row_queue.put(compressed_line)
    binary_row_queue.put("STOP")

def _write_output_binary(outfile_name, binary_row_queue, numprocs):
    """
    Open outgoing file and dump binary output to the file

        outfile_name - filename to write to
        binary_row_queue - a multiprocessing queue that contains streams
            of raw binary data to be blindly written to file. Marked with STOP.
        numprocs - necessary to know how many STOPs to expect.
    """
    lowpriority()
    outfile = open(outfile_name, "wb")

    #Keep running until we see numprocs STOP messages
    for _ in range(numprocs):
        for out_line in iter(binary_row_queue.get, "STOP"):

            outfile.write(out_line)
    outfile.close()

def calc_user_days(in_filename, aoi_filename, out_aoi_userday_filename):
    """Read in binary format and an AOI and output modified AOI with userday
        information.

        """

    aoi_datasource = ogr.Open(aoi_filename)
    #If there is already an existing shapefile with the same name and path,
    # delete it
    if os.path.isfile(out_aoi_userday_filename):
        os.remove(out_aoi_userday_filename)
    #Copy the input shapefile into the designated output folder
    esri_driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource_copy = esri_driver.CopyDataSource(
        aoi_datasource, out_aoi_userday_filename)
    layer = datasource_copy.GetLayer()
    photouserday_field = ogr.FieldDefn('PUD', ogr.OFTInteger)
    layer.CreateField(photouserday_field)

    print 'Loading the polygons into Shapely'
    poly_list = []
    for poly_index, poly_feat in enumerate(layer):
        poly_wkt = poly_feat.GetGeometryRef().ExportToWkt()
        shapely_polygon = shapely.wkt.loads(poly_wkt)
        poly_list.append(shapely_polygon)

    #Get the bounding box of polygon geometries by taking the union
    polygon_collection = shapely.ops.unary_union(poly_list)
    polygon_bounds = polygon_collection.bounds

    #list to keep track of the unique photo user day ids.
    pud_list = [set() for _ in xrange(len(poly_list))]
    with open(in_filename, 'rb') as fin:
        while True:
            #read the number of bytes to read in the compressed block
            compressed_block_size_buf = fin.read(4)
            if not compressed_block_size_buf:
                # if None, must be at the end of the file
                break

            #read that many bytes and decompress
            compressed_block_size = int(
                struct.unpack('i', compressed_block_size_buf)[0])
            compressed_buf = fin.read(compressed_block_size)
            buf = zlib.decompress(compressed_buf)

            #loop over number of blocks
            for block_index in xrange(len(buf) / 12):
                pud_hash, lat, lng = struct.unpack(
                    'iff', buf[block_index*12:(block_index+1)*12])
                #quick check to see if it's in the global bounding volume
                if (lng >= polygon_bounds[0] and lat >= polygon_bounds[1] and
                        lng <= polygon_bounds[2] and lat <= polygon_bounds[3]):
                    #If it is, make an expensive shapely point and test
                    #for contianment on given polygons
                    point = shapely.geometry.Point(lng, lat)
                    for poly_index, poly in enumerate(poly_list):
                        if poly.contains(point):
                            #adding to a set, so if the pud_hash is already in
                            #there it's not counted twice
                            pud_list[poly_index].add(pud_hash)

    #visit each AOI polygon and update its PUD field with the size of the
    #pud_hash for each polygon.
    layer.ResetReading()
    for poly_index, poly_feat in enumerate(layer):
        poly_feat.SetField('PUD', len(pud_list[poly_index]))
        layer.SetFeature(poly_feat)

def main():
    """Entry point that makes profiling easier"""

    #parse expected inputs
    raw_csv_filename = sys.argv[1]
    output_binary_lookup_filename = sys.argv[2]
    aoi_shapefile_filename = sys.argv[3]
    userday_update_aoi_filename = sys.argv[2]+'.userday.shp'

    #convert the CSV file into a binary file.  In deployment we'd likely
    #do this once and ship the binary along with the model.
    print 'converting csv to compressed binary'
    process_csv_to_binary(
        multiprocessing.cpu_count(), raw_csv_filename,
        output_binary_lookup_filename)

    print 'creating user days for polygon'
    calc_user_days(
        output_binary_lookup_filename, aoi_shapefile_filename,
        userday_update_aoi_filename)

    print 'result is in: %s' % userday_update_aoi_filename

if __name__ == '__main__':
    PROFILE = False
    if len(sys.argv) == 4:
        if not PROFILE:
            main()
        else:
            #Code to profile and list top 10 times by cumulative and total time
            cProfile.run('main()', 'pud_stats')
            PSTAT_RESULT = pstats.Stats('pud_stats')
            PSTAT_RESULT.sort_stats('cumulative', 'time').print_stats(10)
            PSTAT_RESULT.sort_stats('time', 'cumulative').print_stats(10)
    else:
        print "Error, usage:"
        print "%s input.csv ouput.bin userdaypoly.shp" % sys.argv[0]
        sys.exit(-1)
