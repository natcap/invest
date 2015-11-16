"""script to simplify sample data to the bounds of a given shapefile"""
import time

import sys
import shapely.geometry
from osgeo import ogr


def main():
    """entry point"""
    csv_path = sys.argv[1]
    vector_path = sys.argv[2]
    clipped_data_path = sys.argv[3]

    vector = ogr.Open(vector_path)
    layer = vector.GetLayer()
    extent = layer.GetExtent()
    print extent
    ext = [
        (extent[0], extent[2]),
        (extent[1], extent[2]),
        (extent[1], extent[3]),
        (extent[0], extent[3]),
        (extent[0], extent[2])]
    poly = shapely.geometry.Polygon(ext)

    with open(csv_path, 'rb') as in_file, open(clipped_data_path, 'wb') as out_file:
        header = in_file.next()
        del header
        count = 0
        last_time = time.time()
        n_lines = 55090216
        for line in in_file:
            count += 1
            current_time = time.time()
            if current_time - last_time > 5.0:
                last_time = current_time
                print '%.2f%% complete' % (100.0 * count / float(n_lines))
            photo_id, owner_name, date_taken, latitude, longitude, accuracy = (
                line.split(','))
            point = shapely.geometry.Point(float(longitude), float(latitude))
            if poly.contains(point):
                out_file.write(
                    '%s,%s,%s,%s,%s,%s' % (
                        photo_id, owner_name, date_taken, latitude, longitude,
                        accuracy))
                print point
        print count


if __name__ == '__main__':
    main()