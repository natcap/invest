import os
import tempfile
import logging
import time
import sys
import traceback

cimport numpy
import numpy
cimport cython
from libcpp.map cimport map

from libc.math cimport sqrt
from libc.math cimport exp
from libc.math cimport ceil

from osgeo import gdal

LOGGER = logging.getLogger('natcap.invest.pygeoprocessing_0_3_3.geoprocessing_cython')


@cython.boundscheck(False)
def reclassify_by_dictionary(dataset, rules, output_uri, format,
    float default_value, datatype, output_dataset):
    """Convert all the non-default values in dataset to the values mapped to
        by rules.  If there is no rule for an input value it is replaced by
        the default output value (which may or may not be the raster's nodata
        value ... it could just be any default value).

        dataset - GDAL raster dataset
        rules - a dictionary of the form:
            {'dataset_value1' : 'output_value1', ...
             'dataset_valuen' : 'output_valuen'}
             used to map dataset input types to output
        output_uri - The location to hold the output raster on disk
        format - either 'MEM' or 'GTiff'
        default_value - output raster dataset default value (may be nodata)
        datatype - a GDAL output type

        return the mapped raster as a GDAL dataset"""

    dataset_band = dataset.GetRasterBand(1)

    cdef map[float,float] lookup
    for key in rules.keys():
        lookup[float(key)] = rules[key]

    output_band = output_dataset.GetRasterBand(1)

    cdef int n_rows = output_band.YSize
    cdef int n_cols = output_band.XSize
    cdef numpy.ndarray[numpy.float_t, ndim=2] dataset_array = numpy.empty((1, n_cols))
    cdef float value = 0.0

    for row in range(n_rows):
        dataset_band.ReadAsArray(0,row,output_band.XSize,1, buf_obj = dataset_array)
        for col in range(n_cols):
            value = dataset_array[0,col]
            if lookup.count(value) == 1:
                dataset_array[0,col] = lookup[value]
            else:
                dataset_array[0,col] = default_value
        output_band.WriteArray(dataset_array, 0, row)

    output_band = None
    output_dataset.FlushCache()

    return output_dataset


def _cython_calculate_slope(dem_dataset_uri, slope_uri):
    """Generates raster maps of slope.  Follows the algorithm described here:
        http://webhelp.esri.com/arcgiSDEsktop/9.3/index.cfm?TopicName=How%20Slope%20works
        and generates a slope dataset as a percent

        dem_dataset_uri - (input) a URI to a  single band raster of z values.
        slope_uri - (input) a path to the output slope uri in percent.

        returns nothing"""

    #Read the DEM directly into an array
    cdef float a,b,c,d,e,f,g,h,i,dem_nodata,z
    cdef int row_index, col_index, n_rows, n_cols

    dem_dataset = gdal.OpenEx(dem_dataset_uri)
    dem_band = dem_dataset.GetRasterBand(1)
    dem_nodata = dem_band.GetNoDataValue()

    slope_dataset = gdal.OpenEx(slope_uri, gdal.GA_Update)
    slope_band = slope_dataset.GetRasterBand(1)
    slope_nodata = slope_band.GetNoDataValue()

    gt = dem_dataset.GetGeoTransform()
    cdef float cell_size_times_8 = gt[1] * 8

    n_rows = dem_band.YSize
    n_cols = dem_band.XSize

    cdef numpy.ndarray[numpy.float_t, ndim=2] dem_array = numpy.empty((3, n_cols))
    cdef numpy.ndarray[numpy.float_t, ndim=2] slope_array = numpy.empty((1, n_cols))

    #Fill the top and bottom row of the slope since we won't touch it in this loop
    slope_array[0, :] = slope_nodata
    slope_band.WriteArray(slope_array, 0, 0)
    slope_band.WriteArray(slope_array, 0, n_rows - 1)

    cdef numpy.ndarray[numpy.float_t, ndim=2] dzdx = numpy.empty((1, n_cols))
    cdef numpy.ndarray[numpy.float_t, ndim=2] dzdy = numpy.empty((1, n_cols))

    for row_index in xrange(n_rows):
        #Loop through the dataset 3 rows at a time
        start_row_index = row_index - 1
        n_rows_to_read = 3
        # see if we need to loose a row on the top
        if start_row_index < 0:
            n_rows_to_read -= 1
            start_row_index = 0
        # see if we need to lose a row on the bottom
        if start_row_index + 2 >= n_rows:
            # -= 1 allows us to handle single row DEMs
            n_rows_to_read -= 1

        dem_array = dem_band.ReadAsArray(0, start_row_index, n_cols, n_rows_to_read, buf_obj=dem_array)
        slope_array[0, :] = slope_nodata
        dzdx[:] = slope_nodata
        dzdy[:] = slope_nodata
        for col_index in xrange(n_cols):
            # abc
            # def
            # ghi

            # e will be the value of any out of bound or nodata pixels
            e = dem_array[1, col_index]
            if e == dem_nodata:
                continue

            if row_index > 0:  # top bounds check
                if col_index > 0:  # left bounds check
                    a = dem_array[0, col_index - 1]
                    if a == dem_nodata:
                        a = e
                else:
                    a = e

                b = dem_array[0, col_index]
                if b == dem_nodata:
                    b = e

                if col_index < n_cols - 1:  # right bounds check
                    c = dem_array[0, col_index + 1]
                if c == dem_nodata:
                    c = e
            else:
                # entire top row is out of bounds
                a = e
                b = e
                c = e

            if col_index > 0:  # left bounds check
                d = dem_array[1, col_index - 1]
                if d == dem_nodata:
                    d = e
            else:
                d = e

            if col_index < n_cols - 1:  # right bounds check
                f = dem_array[1, col_index + 1]
                if f == dem_nodata:
                    f = e
            else:
                f = e

            if row_index < n_rows - 1:  # bottom bounds check
                if col_index > 0:  # left bounds check
                    g = dem_array[2, col_index - 1]
                    if g == dem_nodata:
                        g = e
                else:
                    g = e

                h = dem_array[2, col_index]
                if h == dem_nodata:
                    h = e

                if col_index < n_cols - 1:  # right bounds check
                    i = dem_array[2, col_index + 1]
                    if i == dem_nodata:
                        i = e
                else:
                    i = e
            else:
                # entire bottom row is out of bounds
                g = e
                h = e
                i = e

            dzdx[0, col_index] = ((c+2*f+i) - (a+2*d+g)) / (cell_size_times_8)
            dzdy[0, col_index] = ((g+2*h+i) - (a+2*b+c)) / (cell_size_times_8)
            #output in terms of percent

        slope_array[:] = numpy.where(dzdx != slope_nodata, numpy.tan(numpy.arctan(numpy.sqrt(dzdx**2 + dzdy**2))) * 100, slope_nodata)
        slope_band.WriteArray(slope_array, 0, row_index)

    dem_band = None
    slope_band = None
    gdal.Dataset.__swig_destroy__(dem_dataset)
    gdal.Dataset.__swig_destroy__(slope_dataset)
    dem_dataset = None
    slope_dataset = None


cdef long long _f(long long x, long long i, long long gi):
    return (x-i)*(x-i)+ gi*gi


@cython.cdivision(True)
cdef long long _sep(long long i, long long u, long long gu, long long gi):
    return (u*u - i*i + gu*gu - gi*gi) / (2*(u-i))


#@cython.boundscheck(False)
def distance_transform_edt(input_mask_uri, output_distance_uri):
    """Calculate the Euclidean distance transform on input_mask_uri and output
        the result into an output raster

        input_mask_uri - a gdal raster to calculate distance from the 0 value
            pixels

        output_distance_uri - will make a float raster w/ same dimensions and
            projection as input_mask_uri where all non-zero values of
            input_mask_uri are equal to the euclidean distance to the closest
            0 pixel.

        returns nothing"""

    input_mask_ds = gdal.OpenEx(input_mask_uri)
    input_mask_band = input_mask_ds.GetRasterBand(1)
    cdef int n_cols = input_mask_ds.RasterXSize
    cdef int n_rows = input_mask_ds.RasterYSize
    cdef int block_size = input_mask_band.GetBlockSize()[0]
    cdef int input_nodata = input_mask_band.GetNoDataValue()

    #create a transposed g function
    file_handle, g_dataset_uri = tempfile.mkstemp()
    os.close(file_handle)
    cdef int g_nodata = -1

    input_projection = input_mask_ds.GetProjection()
    input_geotransform = input_mask_ds.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    #invert the rows and columns since it's a transpose
    g_dataset = driver.Create(
        g_dataset_uri.encode('utf-8'), n_cols, n_rows, 1, gdal.GDT_Int32,
        options=['TILED=YES', 'BLOCKXSIZE=%d' % block_size, 'BLOCKYSIZE=%d' % block_size])

    g_dataset.SetProjection(input_projection)
    g_dataset.SetGeoTransform(input_geotransform)
    g_band = g_dataset.GetRasterBand(1)
    g_band.SetNoDataValue(g_nodata)

    cdef float output_nodata = -1.0
    output_dataset = driver.Create(
        output_distance_uri.encode('utf-8'), n_cols, n_rows, 1,
        gdal.GDT_Float64, options=['TILED=YES', 'BLOCKXSIZE=%d' % block_size,
        'BLOCKYSIZE=%d' % block_size])
    output_dataset.SetProjection(input_projection)
    output_dataset.SetGeoTransform(input_geotransform)
    output_band = output_dataset.GetRasterBand(1)
    output_band.SetNoDataValue(output_nodata)

    #the euclidan distance will be less than this
    cdef int numerical_inf = n_cols + n_rows

    LOGGER.info('Distance Transform Phase 1')
    output_blocksize = output_band.GetBlockSize()
    if output_blocksize[0] != block_size or output_blocksize[1] != block_size:
        raise Exception(
            "Output blocksize should be %d,%d, instead it's %d,%d" % (
                block_size, block_size, output_blocksize[0], output_blocksize[1]))

    #phase one, calculate column G(x,y)

    cdef numpy.ndarray[numpy.int32_t, ndim=2] g_array
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] b_array

    cdef int col_index, row_index, q_index, u_index
    cdef long long w
    cdef int n_col_blocks = int(numpy.ceil(n_cols/float(block_size)))
    cdef int col_block_index, local_col_index, win_xsize
    cdef double current_time, last_time
    last_time = time.time()
    for col_block_index in xrange(n_col_blocks):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                'Distance transform phase 1 %.2f%% complete' %
                (col_block_index/float(n_col_blocks)*100.0))
            last_time = current_time
        local_col_index = col_block_index * block_size
        if n_cols - local_col_index < block_size:
            win_xsize = n_cols - local_col_index
        else:
            win_xsize = block_size
        b_array = input_mask_band.ReadAsArray(
            xoff=local_col_index, yoff=0, win_xsize=win_xsize,
            win_ysize=n_rows)
        g_array = numpy.empty((n_rows, win_xsize), dtype=numpy.int32)

        #initalize the first element to either be infinate distance, or zero if it's a blob
        for col_index in xrange(win_xsize):
            if b_array[0, col_index] and b_array[0, col_index] != input_nodata:
                g_array[0, col_index] = 0
            else:
                g_array[0, col_index] = numerical_inf

            #pass 1 go down
            for row_index in xrange(1, n_rows):
                if b_array[row_index, col_index] and b_array[row_index, col_index] != input_nodata:
                    g_array[row_index, col_index] = 0
                else:
                    g_array[row_index, col_index] = (
                        1 + g_array[row_index - 1, col_index])

            #pass 2 come back up
            for row_index in xrange(n_rows-2, -1, -1):
                if (g_array[row_index + 1, col_index] <
                    g_array[row_index, col_index]):
                    g_array[row_index, col_index] = (
                        1 + g_array[row_index + 1, col_index])
        g_band.WriteArray(
            g_array, xoff=local_col_index, yoff=0)

    g_band.FlushCache()
    LOGGER.info('Distance Transform Phase 2')
    cdef numpy.ndarray[numpy.int64_t, ndim=2] s_array
    cdef numpy.ndarray[numpy.int64_t, ndim=2] t_array
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dt


    cdef int n_row_blocks = int(numpy.ceil(n_rows/float(block_size)))
    cdef int row_block_index, local_row_index, win_ysize

    for row_block_index in xrange(n_row_blocks):
        current_time = time.time()
        if current_time - last_time > 5.0:
            LOGGER.info(
                'Distance transform phase 2 %.2f%% complete' %
                (row_block_index/float(n_row_blocks)*100.0))
            last_time = current_time

        local_row_index = row_block_index * block_size
        if n_rows - local_row_index < block_size:
            win_ysize = n_rows - local_row_index
        else:
            win_ysize = block_size

        g_array = g_band.ReadAsArray(
            xoff=0, yoff=local_row_index, win_xsize=n_cols,
            win_ysize=win_ysize)

        s_array = numpy.zeros((win_ysize, n_cols), dtype=numpy.int64)
        t_array = numpy.zeros((win_ysize, n_cols), dtype=numpy.int64)
        dt = numpy.empty((win_ysize, n_cols), dtype=numpy.float64)

        for row_index in xrange(win_ysize):
            q_index = 0
            s_array[row_index, 0] = 0
            t_array[row_index, 0] = 0
            for u_index in xrange(1, n_cols):
                while (q_index >= 0 and
                    _f(t_array[row_index, q_index], s_array[row_index, q_index],
                        g_array[row_index, s_array[row_index, q_index]]) >
                    _f(t_array[row_index, q_index], u_index, g_array[row_index, u_index])):
                    q_index -= 1
                if q_index < 0:
                   q_index = 0
                   s_array[row_index, 0] = u_index
                else:
                    w = 1 + _sep(
                        s_array[row_index, q_index], u_index, g_array[row_index, u_index],
                        g_array[row_index, s_array[row_index, q_index]])
                    if w < n_cols:
                        q_index += 1
                        s_array[row_index, q_index] = u_index
                        t_array[row_index, q_index] = w

            for u_index in xrange(n_cols-1, -1, -1):
                dt[row_index, u_index] = _f(
                    u_index, s_array[row_index, q_index],
                    g_array[row_index, s_array[row_index, q_index]])
                if u_index == t_array[row_index, q_index]:
                    q_index -= 1

        b_array = input_mask_band.ReadAsArray(
            xoff=0, yoff=local_row_index, win_xsize=n_cols,
            win_ysize=win_ysize)

        dt = numpy.sqrt(dt)
        dt[b_array == input_nodata] = output_nodata
        output_band.WriteArray(dt, xoff=0, yoff=local_row_index)

    output_band.FlushCache()
    output_band = None
    gdal.Dataset.__swig_destroy__(output_dataset)
    output_dataset = None
    input_mask_band = None
    gdal.Dataset.__swig_destroy__(input_mask_ds)
    input_mask_ds = None
    g_band = None
    gdal.Dataset.__swig_destroy__(g_dataset)
    g_dataset = None
    try:
        os.remove(g_dataset_uri)
    except OSError:
        LOGGER.warn("couldn't remove file %s" % g_dataset_uri)


def new_raster_from_base_uri(base_uri, *args, **kwargs):
    """A wrapper for the function new_raster_from_base that opens up
        the base_uri before passing it to new_raster_from_base.

        base_uri - a URI to a GDAL dataset on disk.

        All other arguments to new_raster_from_base are passed in.

        Returns nothing.
        """
    base_raster = gdal.OpenEx(base_uri)
    if base_raster is None:
        raise IOError("%s not found when opening GDAL raster")
    new_raster = new_raster_from_base(base_raster, *args, **kwargs)

    gdal.Dataset.__swig_destroy__(new_raster)
    gdal.Dataset.__swig_destroy__(base_raster)
    new_raster = None
    base_raster = None


def new_raster_from_base(
    base, output_uri, gdal_format, nodata, datatype, fill_value=None,
    n_rows=None, n_cols=None, dataset_options=None):
    """Create a new, empty GDAL raster dataset with the spatial references,
        geotranforms of the base GDAL raster dataset.

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
        n_rows - (optional) if set makes the resulting raster have n_rows in it
            if not, the number of rows of the outgoing dataset are equal to
            the base.
        n_cols - (optional) similar to n_rows, but for the columns.
        dataset_options - (optional) a list of dataset options that gets
            passed to the gdal creation driver, overrides defaults

        returns a new GDAL raster dataset."""

    #This might be a numpy type coming in, set it to native python type
    try:
        nodata = nodata.item()
    except AttributeError:
        pass

    if n_rows is None:
        n_rows = base.RasterYSize
    if n_cols is None:
        n_cols = base.RasterXSize
    projection = base.GetProjection()
    geotransform = base.GetGeoTransform()
    driver = gdal.GetDriverByName(gdal_format)

    base_band = base.GetRasterBand(1)
    block_size = base_band.GetBlockSize()
    metadata = base_band.GetMetadata('IMAGE_STRUCTURE')
    base_band = None

    if dataset_options == None:
        #make a new list to make sure we aren't ailiasing one passed in
        dataset_options = []
        #first, should it be tiled?  yes if it's not striped
        if block_size[0] != n_cols:
            #just do 256x256 blocks
            dataset_options = [
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=IF_SAFER']
        if 'PIXELTYPE' in metadata:
            dataset_options.append('PIXELTYPE=' + metadata['PIXELTYPE'])

    new_raster = driver.Create(
        output_uri.encode('utf-8'), n_cols, n_rows, 1, datatype,
        options=dataset_options)
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)
    band = new_raster.GetRasterBand(1)

    if nodata is not None:
        band.SetNoDataValue(nodata)
    else:
        LOGGER.warn(
            "None is passed in for the nodata value, failed to set any nodata "
            "value for new raster.")

    if fill_value != None:
        band.Fill(fill_value)
    elif nodata is not None:
        band.Fill(nodata)
    band = None

    return new_raster
