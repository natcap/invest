"""numpy regex tracer"""
import numpy
import time
import re

RAW_CSV_POINT_DATA_PATH = r"src\natcap\invest\recreation\foo.csv"


def main():
    """entry point"""
    regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]

    #8568090486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
    pattern = "[^,]+,([^,]+),([^-]{4}-[^-]{2}-[^-]{2}) [^,]+,([^,]+),([^,]+),[^\n]"
    regexp = re.compile(pattern)
    raw_csv_point_data_file = open(RAW_CSV_POINT_DATA_PATH)
    raw_csv_point_data_file.readline()  # skip first line
    start = time.time()
    result = numpy.fromregex(
        raw_csv_point_data_file, regexp,
        [('user', 'S40'), ('date', 'S10'),
         ('lat', numpy.float32), ('lng', numpy.float32)])
    valid_date_pattern = r'19|20\d\d-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])'
    regexp_valid_date = re.compile(valid_date_pattern)
    vmatch = numpy.vectorize(lambda x: bool(regexp_valid_date.match(x)))
    sel = vmatch(result['date'])
    print time.time() - start
    print result
    print sel.shape
    print result[~sel].shape


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
