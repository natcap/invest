import sys
import math

from osgeo import ogr
from osgeo import osr
from osgeo import gdal

def createRasterFromVectorExtents(xRes, yRes, format, nodata, rasterFile, shp):
    """Create a blank raster based on a vector file extent.  This code is
        adapted from http://trac.osgeo.org/gdal/wiki/FAQRaster#HowcanIcreateablankrasterbasedonavectorfilesextentsforusewithgdal_rasterizeGDAL1.8.0
    
        xRes - the x size of a pixel in the output dataset must be a positive 
            value
        yRes - the y size of a pixel in the output dataset must be a positive 
            value
        format - gdal GDT pixel type
        nodata - the output nodata value
        rasterFile - URI to file location for raster
        shp - vector shapefile to base extent of output raster on
        
        returns a blank raster whose bounds fit within `shp`s bounding box
            and features are equivalent to the passed in data"""

    #Determine the width and height of the tiff in pixels based on desired
    #x and y resolution
    shpExtent = shp.GetLayer(0).GetExtent()
    tiff_width = int(math.ceil(abs(shpExtent[1] - shpExtent[0]) / xRes))
    tiff_height = int(math.ceil(abs(shpExtent[3] - shpExtent[2]) / yRes))

    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(rasterFile, tiff_width, tiff_height, 1, format)
    raster.GetRasterBand(1).SetNoDataValue(nodata)

    #Set the transform based on the upper left corner and given pixel
    #dimensions
    raster_transform = [shpExtent[0], xRes, 0.0, shpExtent[3], 0.0, -yRes]
    raster.SetGeoTransform(raster_transform)

    #Use the same projection on the raster as the shapefile
    srs = osr.SpatialReference()
    srs.ImportFromWkt(shp.GetLayer(0).GetSpatialRef().__str__())
    raster.SetProjection(srs.ExportToWkt())

    #Initialize everything to nodata
    raster.GetRasterBand(1).Fill(nodata)
    raster.GetRasterBand(1).FlushCache()

    return raster

try:
    land_poly_file = sys.argv[1]
    aoi_poly_file = sys.argv[2]
    cell_size = int(sys.argv[3])
    outfile_name = sys.argv[4]

except:
    print "Usage create_grid.py land_poly_file aoi_poly_file cell_size"


land_ds = ogr.Open(land_poly_file)
aoi_ds = ogr.Open(aoi_poly_file)

    #format of aoi_extent [xleft, xright, ybot, ytop]
aoi_extent = aoi_ds.GetLayer(0).GetExtent()
xleft,xright,ybot,ytop = aoi_extent

srs = osr.SpatialReference()
srs.ImportFromWkt(land_ds.GetLayer(0).GetSpatialRef().__str__())
linear_units = srs.GetLinearUnits()

print xright-xleft

print linear_units

x_ticks = int((xright-xleft)/float(cell_size))
y_ticks = int((ytop-ybot)/float(cell_size))
    
print aoi_extent
print x_ticks, y_ticks

    #Get the land layer
land_layer = land_ds.GetLayer(0)

output_dataset = createRasterFromVectorExtents(cell_size, cell_size, gdal.GDT_Byte, 255, 'grid.tif', aoi_ds)
output_band = output_dataset.GetRasterBand(1)

#First fill it up with water (bit == 1)
output_dataset.GetRasterBand(1).Fill(1)

#Then fill it up with land (bit == 0)
gdal.RasterizeLayer(output_dataset, [1], land_layer, burn_values=[0])

land_array = output_dataset.GetRasterBand(1).ReadAsArray()

f=open(outfile_name,'w')

for y in range(land_array.shape[0]):
    for x in range(land_array.shape[1]):
        f.write(str(land_array[land_array.shape[0]-y-1][x]))
    f.write('\n')

#    for x_index in range(x_ticks):
#        for y_index in range(y_ticks):
#            x_coord = xleft+x_index*cell_size
#            y_coord = ytop-y_index*cell_size

