# cython: language_level=2
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
import numpy
import pygeoprocessing
cimport numpy
cimport cython
from libcpp.list cimport list as clist
from libcpp.pair cimport pair as cpair

# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()

# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef _calculate_pour_point_array2(int[:, :] flow_dir_array, 
                                   int[:] edges, 
                                   int nodata):
    """
    Return a binary array indicating which pixels are pour points.

    Args:
        flow_dir_array (numpy.ndarray): a 2D array of D8 flow direction 
            values (0 - 7) and possiby the nodata value.
        edges (dict): has boolean keys 'top', 'left', 'bottom', 'right'
            indicating whether or not each edge is an edge of the raster.
the 
    Returns:
        numpy.ndarray of the same shape as ``flow_direction_array``.
        Each element is 1 if that pixel is a pour point, 0 if not,
        nodata if it cannot be determined.
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

    pour_point_array = numpy.empty((height, width), dtype=numpy.intc)
    cdef int[:, :] pour_point_array_view = pour_point_array

    # iterate over each pixel
    for row in range(height):
        for col in range(width):

            # get the flow direction [0-7] for this pixel
            flow_dir = flow_dir_array[row, col]
            if flow_dir == nodata:
                pour_point_array_view[row, col] = nodata
            else:
                # get the location of the pixel it flows into
                sink_row = row + row_offsets[flow_dir]
                sink_col = col + col_offsets[flow_dir]
                # if the sink pixel value is nodata, it's either flowing
                # into a nodata area, or off the edge of the whole raster
                # either case means this is a pour point

                # edges order: top, left, bottom, right

                if sink_row == -1:
                    if edges[0]:  # top edge
                        pour_point_array_view[row, col] = 1
                    else:
                        pour_point_array_view[row, col] = nodata
                    continue
                if sink_col == -1:
                    if edges[1]:  # left edge
                        pour_point_array_view[row, col] = 1
                    else:
                        pour_point_array_view[row, col] = nodata
                    continue
                if sink_row == height:
                    if edges[2]:  # bottom edge
                        pour_point_array_view[row, col] = 1
                    else:
                        pour_point_array_view[row, col] = nodata
                    continue
                if sink_col == width:
                    if edges[3]:  # right edge
                        pour_point_array_view[row, col] = 1
                    else:
                        pour_point_array_view[row, col] = nodata
                    continue

                else:
                    if flow_dir_array[sink_row, sink_col] == nodata:
                        pour_point_array_view[row, col] = 1
                    else:
                        pour_point_array_view[row, col] = 0


    return pour_point_array

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef clist[cpair[int, int]] _calculate_pour_point_array3(int[:, :] flow_dir_array, 
                                   int[:] edges, 
                                   int nodata):
    """
    Return a binary array indicating which pixels are pour points.

    Args:
        flow_dir_array (numpy.ndarray): a 2D array of D8 flow direction 
            values (0 - 7) and possiby the nodata value.
        edges (dict): has boolean keys 'top', 'left', 'bottom', 'right'
            indicating whether or not each edge is an edge of the raster.
the 
    Returns:
        numpy.ndarray of the same shape as ``flow_direction_array``.
        Each element is 1 if that pixel is a pour point, 0 if not,
        nodata if it cannot be determined.
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

    cdef clist[cpair[int, int]] pour_points

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
                    if edges[0]:  # top edge
                        pour_points.push_back(cpair[int, int](row, col))
                elif sink_col == -1:
                    if edges[1]:  # left edge
                        pour_points.push_back(cpair[int, int](row, col))
                elif sink_row == height:
                    if edges[2]:  # bottom edge
                        pour_points.push_back(cpair[int, int](row, col))
                elif sink_col == width:
                    if edges[3]:  # right edge
                        pour_points.push_back(cpair[int, int](row, col))

                # if we get to here, the point (sink_row, sink_col)  
                # is known to be within the bounds of the array,
                # so it's safe to index 
                elif flow_dir_array[sink_row, sink_col] == nodata:
                    pour_points.push_back(cpair[int, int](row, col))

    return pour_points

# def func():
#     return 1
def python_wrapper3(flow_dir_array, edges, nodata):
    return _calculate_pour_point_array3(flow_dir_array, edges, nodata)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef clist[cpair[int, int]] _calculate_pour_point_array4(int[:, :] flow_dir_array, 
                                   int[:] edges, 
                                   int nodata):
    """
    Return a binary array indicating which pixels are pour points.

    Args:
        flow_dir_array (numpy.ndarray): a 2D array of D8 flow direction 
            values (0 - 7) and possiby the nodata value.
        edges (dict): has boolean keys 'top', 'left', 'bottom', 'right'
            indicating whether or not each edge is an edge of the raster.
the 
    Returns:
        numpy.ndarray of the same shape as ``flow_direction_array``.
        Each element is 1 if that pixel is a pour point, 0 if not,
        nodata if it cannot be determined.
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

    cdef clist[cpair[int, int]] pour_points

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
                    if edges[0]:  # top edge
                        pour_points.push_back(cpair[int, int](row, col))
                elif sink_col == -1:
                    if edges[1]:  # left edge
                        pour_points.push_back(cpair[int, int](row, col))
                elif sink_row == height:
                    if edges[2]:  # bottom edge
                        pour_points.push_back(cpair[int, int](row, col))
                elif sink_col == width:
                    if edges[3]:  # right edge
                        pour_points.push_back(cpair[int, int](row, col))

                # if we get to here, the point (sink_row, sink_col)  
                # is known to be within the bounds of the array,
                # so it's safe to index 
                elif flow_dir_array[sink_row, sink_col] == nodata:
                    pour_points.push_back(cpair[int, int](row, col))

    return pour_points


def profile_test():
    flow_dir_array = numpy.zeros((128, 128), dtype=numpy.intc)
    edges = numpy.array([1, 1, 0, 0], dtype=numpy.intc)
    nodata = 100
    for i in range(10000):
        _calculate_pour_point_array2(flow_dir_array, edges, nodata)
    for i in range(10000):
        python_wrapper3(flow_dir_array, edges, nodata)
    for i in range(10000):
        _calculate_pour_point_array4(flow_dir_array, edges, nodata)



