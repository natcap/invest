"""Urban Heat Island Cython Core."""
import pygeoprocessing


def blob_mask(
        mask_raster_band_path, target_blob_id_raster_path,
        id_count_map_pickle_path):
    """Blob up touching raster fronts.

    Parameters:
        mask_raster_band_path (tuple): path/band tuple to a raster that has
            1 or 0 or nodata. Intent is to blob all the contiguoug

    """
    cdef int i, j
    for offset_dict, block_array in pygeoprocessing.iterblocks(
            mask_raster_band_path):
        for i in range(block_array.shape[0]):
            for j in range(block_array.shape[1]):
                pass
