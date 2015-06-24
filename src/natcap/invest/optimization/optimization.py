from osgeo import gdal
import numpy
import bisect
import scipy.ndimage.filters

def new_raster_from_base(base, output_uri, gdal_format, nodata, datatype, fill_value=None, ):
    """Create a new, empty GDAL raster dataset with the spatial references,
        dimensions and geotranforms of the base GDAL raster dataset.
        
        base - a the GDAL raster dataset to base output size, and transforms on
        output_uri - a string URI to the new output raster dataset.
        gdal_format - a string representing the GDAL file format of the 
            output raster.  See http://gdal.org/formats_list.html for a list
            of available formats.  This parameter expects the format code, such
            as 'GTiff' or 'MEM'
        nodata - a value that will be set as the nodata value for the 
            output raster.  Should be the same type as 'datatype'
        datatype - the pixel datatype of the output raster, for example 
            gdal.GDT_Float32.  See the following header file for supported 
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        fill_value - (optional) the value to fill in the raster on creation
                
        returns a new GDAL raster dataset."""

    n_cols = base.RasterXSize
    n_rows = base.RasterYSize
    projection = base.GetProjection()
    geotransform = base.GetGeoTransform()
    driver = gdal.GetDriverByName(gdal_format)
    new_raster = driver.Create(output_uri.encode('utf-8'), n_cols, n_rows, 1, datatype)
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)
    band = new_raster.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    if fill_value != None:
        band.SetNoDataValue(fill_value)
    else:
        band.SetNoDataValue(nodata)
    band = None

    return new_raster


def static_max_marginal_gain(
    score_dataset_uri, budget, output_datset_uri, sigma=0.0, aoi_uri=None):
    """This funciton calculates the maximum marginal gain by selecting pixels
        in a greedy fashion until the entire budget is spent.
        
        score_dataset_uri - gdal dataset to a float raster
        budget - number of pixels to select
        output_dataset_uri - the uri to an output gdal dataset of type gdal.Byte
            values are 0 if not selected, 1 if selected, and nodata if the original
            was nodata.
        sigma - a "clumping factor" parameter that biases the selection of maximum
            pixels to be close to other selected pixels.  The higher the value
            the higher the clumps.  Formally this is the sigma paramter on
            a gaussian filter that operates on the original score_dataset_uri
            the default is 0.0 which does not bias selection toward clumping.
        aoi_uri - an area to consider selection
    
    returns nothing"""
    
    #TODO: mask aoi_uri here
    
    dataset = gdal.Open(score_dataset_uri)
    band = dataset.GetRasterBand(1)
    
    #TODO: use memmapped or hd5 arrays here
    array = band.ReadAsArray()
    in_nodata = band.GetNoDataValue()
    array[numpy.isnan(array)] = in_nodata
    nodata_mask = array == in_nodata
    
    #Gaussian smooth the array but treating nodata as 0.0
    if sigma != 0.0:
        array[array==in_nodata] = 0.0
        array = scipy.ndimage.filters.gaussian_filter(array, sigma, mode='constant', cval=0.0)
        #put the nodata values back
        array[nodata_mask] = in_nodata
        nodata_mask = None
        
    #This sets any nans to nodata values, an issue with the MN data
    flat_array = array.flat
    ordered_indexes = numpy.argsort(flat_array)
    ordered_array = flat_array[ordered_indexes]
    
    #TODO: use memmapped or hd5 arrays here
    #This is the array that will record what features are selected
    out_nodata = 255
    selection_array = numpy.empty_like(flat_array, dtype=numpy.ubyte)
    selection_array[:] = out_nodata
    
    #This finds the range of nodata values in the sorted array
    left_nodata_index = bisect.bisect_left(ordered_array, in_nodata)
    right_nodata_index = bisect.bisect_right(ordered_array, in_nodata)
    
    n_elements = len(ordered_indexes)
    right_length = n_elements-right_nodata_index
    
    if budget < right_length:
        #We can spend it all on the right side of the array
        selection_array[ordered_indexes[(n_elements-budget):n_elements]] = 1
        budget = 0
    else:
        #Spend what we can on the right and figure out the left
        selection_array[ordered_indexes[right_nodata_index:n_elements]] = 1
        budget -= right_length
        if budget < left_nodata_index:
            #we can spend the remaining on the left side of the array
            selection_array[ordered_indexes[(left_nodata_index-budget):left_nodata_index]] = 1
            budget = 0
        else:
            #Otherwise select the rest of the array and have some remaining budget
            selection_array[ordered_indexes[0:left_nodata_index]] = 1
            budget -= left_nodata_index
    
    print 'remaining budget', budget
    
    #Write output result
    out_dataset = new_raster_from_base(dataset, output_datset_uri, 'GTiff', out_nodata, gdal.GDT_Byte)
    out_band = out_dataset.GetRasterBand(1)
    selection_array.shape = array.shape
    out_band.WriteArray(selection_array)
    
