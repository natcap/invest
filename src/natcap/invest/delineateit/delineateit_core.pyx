# cython: language_level=3
import numpy
import pygeoprocessing
cimport numpy
cimport cython
from libcpp.set cimport set as cset
from libcpp.pair cimport pair as cpair


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef cset[cpair[double, double]] calculate_pour_point_array(
    int[:, :] flow_dir_array, 
    int[:] edges, 
    int nodata,
    cpair[int, int] offset,
    cpair[double, double] origin,
    cpair[double, double] pixel_size):
    """
    Return a binary array indicating which pixels are pour points.

    Args:
        flow_dir_array (numpy.ndarray): a 2D array of D8 flow direction 
            values (0 - 7) and possiby the nodata value.
        edges (list): a binary list of length 4 indicating whether or not each 
            edge is an edge of the raster. Order: [top, left, bottom, right]
        nodata (int): the nodata value of the flow direction array
        offset (cpair[int, int]): the input flow_dir_array is a block taken 
            from a larger raster. Offset is the (x, y) coordinate of the
            upper-left corner of this block relative to the whole raster as a
            numpy array (the raster starts at (0, 0) and each pixel is 1x1).
        origin (cpair[double, double]): the (x, y) origin of the raster from 
            which this block was taken, in its original coordinate system. 
            This is equivalent to elements (0, 3) of:
            ``pygeoprocessing.get_raster_info(flow_dir_raster)['geotransform']``.
        pixel_size (cpair[double, double]): the (x, y) dimensions of a pixel in 
            the raster from which this block was taken, in its original 
            coordinate system. This is equivalent to:
            ``pygeoprocessing.get_raster_info(flow_dir_raster)['pixel_size']``.

    Returns:
        set of (x, y) coordinates representing pour points in the coordinate
        system of the original raster. C type cset[cpair[double, double]] is 
        automatically cast to python type set(tuple(float, float)) when this
        function is called from a python function.
    """   

    # flow direction: (delta rows, delta columns)
    # for a pixel at (i, j) with flow direction x,
    # the pixel it flows into is located at:
    # (i + directions[x][0], j + directions[x][1])
    cdef int *row_offsets = [0, -1, -1, -1,  0,  1, 1, 1]
    cdef int *col_offsets = [1,  1,  0, -1, -1, -1, 0, 1]

    cdef int height, width
    cdef int row, col
    cdef int flow_dir
    cdef int sink_row, sink_col

    # tuple unpacking doesn't work in cython
    height, width = flow_dir_array.shape[0], flow_dir_array.shape[1]

    cdef cset[cpair[double, double]] pour_points

    # iterate over each pixel
    for row in range(height):
        for col in range(width):
            
            # get the flow direction [0-7] for this pixel
            flow_dir = flow_dir_array[row, col]
            if flow_dir != nodata:
                # get the location of the pixel it flows into
                sink_row = row + row_offsets[flow_dir]
                sink_col = col + col_offsets[flow_dir]

                # if the row index is -1, the pixel is flowing off of the block
                if sink_row == -1:
                    # if this edge of the block is an edge of the raster, 
                    # this is a pour point. Otherwise, we can't say whether it is
                    # because the necessary information isn't in this block.
                    if not edges[0]:  # top edge
                        continue
                elif sink_col == -1:
                    if not edges[1]:  # left edge
                        continue
                elif sink_row == height:
                    if not edges[2]:  # bottom edge
                        continue
                elif sink_col == width:
                    if not edges[3]:  # right edge
                        continue

                # if we get to here without having continued, the point 
                # (sink_row, sink_col) is known to be within the bounds of 
                # the array, so it's safe to index 
                elif flow_dir_array[sink_row, sink_col] != nodata:
                    continue
                
                # if none of the above conditions passed, this is a pour point
                pour_points.insert(cpair[double, double](
                    # +0.5 so that the point is centered in the pixel
                    (col + offset.first + 0.5) * pixel_size.first + origin.first,
                    (row + offset.second + 0.5) * pixel_size.second + origin.second))

    # return set of (x, y) coordinates referenced to the same coordinate system
    # as the original raster
    return pour_points


