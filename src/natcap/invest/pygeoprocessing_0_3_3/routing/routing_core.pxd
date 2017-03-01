from libcpp.deque cimport deque
from libcpp.map cimport map
cimport numpy

cdef class BlockCache:
    cdef numpy.int32_t[:,:] row_tag_cache
    cdef numpy.int32_t[:,:] col_tag_cache
    cdef numpy.int8_t[:,:] cache_dirty
    cdef int n_block_rows
    cdef int n_block_cols
    cdef int block_col_size
    cdef int block_row_size
    cdef int n_rows
    cdef int n_cols
    cdef void update_cache(self, int global_row, int global_col, int *row_index, int *col_index, int *row_block_offset, int *col_block_offset)
    cdef void flush_cache(self)

cdef calculate_transport(
    outflow_direction_uri, outflow_weights_uri, deque[int] &sink_cell_deque,
    source_uri, absorption_rate_uri, loss_uri, flux_uri, absorption_mode,
    stream_uri=?, include_source=?)
cdef flat_edges(dem_uri, flow_direction_uri, deque[int] &high_edges,
                deque[int] &low_edges, int drain_off_edge=?)
cdef label_flats(dem_uri, deque[int] &low_edges, labels_uri)
cdef clean_high_edges(labels_uri, deque[int] &high_edges)
cdef drain_flats(deque[int] &high_edges, deque[int] &low_edges, labels_uri,
                 flow_direction_uri, flat_mask_uri)
cdef away_from_higher(deque[int] &high_edges, labels_uri, flow_direction_uri,
                      flat_mask_uri, map[int, int] &flat_height)
cdef towards_lower(deque[int] &low_edges, labels_uri, flow_direction_uri,
                   flat_mask_uri, map[int, int] &flat_height)
cdef find_outlets(dem_uri, flow_direction_uri, deque[int] &outlet_deque)
