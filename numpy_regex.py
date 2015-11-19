"""numpy regex tracer"""
import hashlib
import numpy
import time
import re
import numpy.lib.recfunctions

RAW_CSV_POINT_DATA_PATH = r"src\natcap\invest\recreation\foo.csv"

def join_struct_arrays(arrays):
    sizes = numpy.array([a.itemsize for a in arrays])
    offsets = numpy.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    dtype = 'S4, f4, f4'
    joint = numpy.empty((n, offsets[-1]), dtype=dtype)
    joint[:, 0:4] = arrays[0].view('S4').reshape(n, 4)
    joint[:, 4:12] = arrays[1].view('f4,f4').reshape(n, 8)
    #for a, size, offset in zip(arrays, sizes, offsets):
    #    joint[:,offset:offset+size] = a.view(numpy.uint8).reshape(n,size)
    #dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def main():
    """entry point"""
    regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]

    #8568090486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
    #pattern = "[^,]+,([^,]+),([^-]{4}-[^-]{2}-[^-]{2}) [^,]+,([^,]+),([^,]+),[^\n]"
    pattern = r"[^,]+,([^,]+),(19|20\d\d-(?:0[1-9]|1[012])-(?:0[1-9]|[12][0-9]|3[01])) [^,]+,([^,]+),([^,]+),[^\n]"
    regexp = re.compile(pattern)
    raw_csv_point_data_file = open(RAW_CSV_POINT_DATA_PATH)
    raw_csv_point_data_file.readline()  # skip first line
    start = time.time()
    result = numpy.fromregex(
        raw_csv_point_data_file, regexp,
        [('user', 'S40'), ('date', 'datetime64[D]'),
         ('lat', numpy.float32), ('lng', numpy.float32)])
    #valid_date_pattern = r'19|20\d\d-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])'
    #regexp_valid_date = re.compile(valid_date_pattern)
    #vmatch = numpy.vectorize(lambda x: bool(regexp_valid_date.match(x)))
    #sel = vmatch(result['date'])
    #print result
    # filter out bad dates
    #result = result[sel]
    #print sel.shape
    year_day = result['date'].astype(int)
    print year_day

    def md5hash(user_string, year_day):
        """md5hash user yearday"""
        return hashlib.md5(user_string+str(year_day)).digest()[-4:]

    md5hash_v = numpy.vectorize(md5hash)
    hashes = md5hash_v(result['user'], year_day)
    print hashes.size

    #result['user'] = numpy.apply_along_axis(md5hash, 1, result[['user', 'date']])

    user_day_lat_lng = numpy.empty(hashes.size, dtype='S4,f4,f4')
    user_day_lat_lng['f0'] = hashes
    user_day_lat_lng['f1'] = result['lat']
    user_day_lat_lng['f2'] = result['lng']

    print user_day_lat_lng
    print time.time() - start


def prof_raw_read():
    raw_csv_point_data_file = open(RAW_CSV_POINT_DATA_PATH)
    count = 0
    start_time = time.time()
    for _ in raw_csv_point_data_file:
        count += 1
    print "%d lines in %.2fs" % (count, time.time() - start_time)


if __name__ == '__main__':
    main()
    prof_raw_read()
