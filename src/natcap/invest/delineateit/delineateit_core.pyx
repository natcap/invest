# cython: profile=False
# cython: language_level=2

cimport numpy
cimport cython
from libcpp.list cimport list as clist

cdef int is_pour_point(clist[int] kernel, int nodata, int fill_value):
    
    if fill_value in kernel:
        return nodata
    if kernel[4] == nodata:
        return nodata
    else:
        return kernel[convert[kernel[4]]] == nodata

