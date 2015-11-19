"""numpy regex tracer"""
import numpy


def main():
    """entry point"""
    raw_csv_point_data_path = r"src\natcap\invest\recreation\foo.csv"
    regexp = r"(\d+)\s+(...)"  # match [digits, whitespace, anything]

    #8568090486,48344648@N00,2013-03-17 16:27:27,42.383841,-71.138378,16
    regexp = r"[^,]+,([^,]+),(\d+)-(\d\d)-(\d\d) [^,]+,([^,]+),([^,]+),"
    raw_csv_point_data_file = open(raw_csv_point_data_path)
    raw_csv_point_data_file.readline() # skip first line
    result = numpy.fromregex(
        raw_csv_point_data_file, regexp,
        [('user', 'S40'), ('year', numpy.int32), ('month', numpy.int32), ('day', numpy.int32), ('lat', numpy.float32), ('lng', numpy.float32)])
    print result['lat']

if __name__ == '__main__':
    main()
