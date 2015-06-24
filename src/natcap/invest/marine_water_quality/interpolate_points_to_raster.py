import sys

from osgeo import gdal
from osgeo import ogr

#This craziness is just for development to import a relative link so I can
#run in the current directory that has a soft link.
try:
    import pygeoprocessing.geoprocessing
except:
    import pygeoprocessing.geoprocessing

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'usage:\n %s AOIPolygonFile PointFile PointAttribute' \
            % sys.argv[0]
        sys.exit(-1)

    aoi_ds = ogr.Open(sys.argv[1])
    point_ds = ogr.Open(sys.argv[2])
    attribute = sys.argv[3]

    CELL_SIZE = 30

    output_dataset = \
        pygeoprocessing.geoprocessing.create_raster_from_vector_extents(CELL_SIZE, CELL_SIZE, 
        gdal.GDT_Float32, -1e10, 'interpolated.tif', aoi_ds)
 
    aoi_layer = aoi_ds.GetLayer()
    aoi_feature = aoi_layer.GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()

    for layer in point_ds:
        for point_feature in layer:
            point_geometry = point_feature.GetGeometryRef()
            point = point_geometry.GetPoint()
            print point, aoi_geometry.Contains(point_geometry)

