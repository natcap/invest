import logging
from osgeo import gdal, ogr
import numpy
import scipy.ndimage

import os.path

import disk_sort

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.scenario_generator.despeckle')

def temporary_filename():
    """Returns a temporary filename using mkstemp. The file is deleted
        on exit using the atexit register.

        returns a unique temporary filename"""

    file_handle, path = tempfile.mkstemp()
    os.close(file_handle)

    def remove_file(path):
        """Function to remove a file and handle exceptions to register
            in atexit"""
        try:
            os.remove(path)
        except OSError:
            #This happens if the file didn't exist, which is okay because maybe
            #we deleted it in a method
            pass

    atexit.register(remove_file, path)
    return path

def new_raster_from_base_uri(base_uri, *args, **kwargs):
    """A wrapper for the function new_raster_from_base that opens up
        the base_uri before passing it to new_raster_from_base.

        base_uri - a URI to a GDAL dataset on disk.

        All other arguments to new_raster_from_base are passed in.

        Returns nothing.
        """
    base_raster = gdal.Open(base_uri)
    new_raster_from_base(base_raster, *args, **kwargs)

def new_raster_from_base(
    base, output_uri, gdal_format, nodata, datatype, fill_value=None):
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
    new_raster = driver.Create(
        output_uri.encode('utf-8'), n_cols, n_rows, 1, datatype,
        options=['COMPRESS=LZW'])
    new_raster.SetProjection(projection)
    new_raster.SetGeoTransform(geotransform)
    band = new_raster.GetRasterBand(1)

    band.SetNoDataValue(nodata)
    if fill_value != None:
        band.Fill(fill_value)
    else:
        band.Fill(nodata)
    band = None

    return new_raster

def filter_fragments_uri(input_uri, value, patch_size, pixel_count, output_uri, replace=None):
    src_ds = gdal.Open(input_uri)
    src_band = src_ds.GetRasterBand(1)
    src_array = src_band.ReadAsArray()

    driver = gdal.GetDriverByName("GTiff")
    driver.CreateCopy(output_uri, src_ds, 0 )

    dst_ds = gdal.Open(output_uri, 1)
    dst_band = dst_ds.GetRasterBand(1)

    if replace == None:
        replace = dst_band.GetNoDataValue()

    LOGGER.info("Filtering patches smaller than %i pixel(s) with value %i from %s.",
                 patch_size,
                 value,
                 os.path.basename(input_uri))
    dst_array = filter_fragments(src_array, value, patch_size, pixel_count, replace)

    dst_band.WriteArray(dst_array)
    LOGGER.info("Results written to %s.", os.path.basename(output_uri))

def filter_fragments(src_array, value, patch_size, pixel_count, replace=None):
    dst_array = numpy.copy(src_array)

    #clump and sieve

    suitability_values = numpy.unique(src_array)
    if suitability_values[0] == 0:
        suitability_values = suitability_values[1:]

    #connectedness preferred, 4 connectedness allowed
    mask = src_array == value # You get a mask with the polygons only
    ones_in_mask = numpy.sum(mask)
##    LOGGER.debug("Found %i pixels of value %i.", ones_in_mask, value)

    label_im, nb_labels = scipy.ndimage.label(mask)
    LOGGER.debug("Found %i patches of %i.", nb_labels, value)
    src_array[mask] = 1
    fragment_sizes = scipy.ndimage.sum(mask, label_im, range(nb_labels + 1))
    fragment_labels = numpy.array(range(nb_labels + 1))

    small_fragment_mask = numpy.where(fragment_sizes < patch_size)
    small_fragment_sizes = fragment_sizes[small_fragment_mask]
    small_fragment_labels = fragment_labels[small_fragment_mask]
    LOGGER.debug("Found %i patches smaller than %i pixel(s).", small_fragment_sizes.size - 1, patch_size)

    combined_small_fragment_size = numpy.sum(small_fragment_sizes)
##    print('fragments to remove', combined_small_fragment_size)
##    print('small fragment sizes', small_fragment_sizes)
##    print('small fragment labels', small_fragment_labels)
    #print('large_fragments', large_fragments.size, large_fragments)

    removed_pixels = 0
    small_fragment_labels = small_fragment_labels[1:]
    numpy.random.shuffle(small_fragment_labels)
    for label in small_fragment_labels:
        pixels_to_remove = numpy.where(label_im == label)
        if (pixel_count != None) and (removed_pixels + pixels_to_remove[0].size > pixel_count):
            LOGGER.debug("Removing part of patch %i.", label)
            patch_mask = numpy.zeros_like(dst_array)
            patch_mask[pixels_to_remove] = 1

            tmp_array = scipy.ndimage.morphology.distance_transform_edt(patch_mask)
            tmp_array = tmp_array[pixels_to_remove]

            tmp_index = numpy.argsort(tmp_array)
            tmp_index = tmp_index[:pixel_count - removed_pixels]

            pixels_to_remove = numpy.array(zip(pixels_to_remove[0], pixels_to_remove[1]))
            pixels_to_remove = pixels_to_remove[tmp_index]
            pixels_to_remove = apply(zip, pixels_to_remove)

            dst_array[pixels_to_remove] = replace

            break

        dst_array[pixels_to_remove] = replace
        removed_pixels += pixels_to_remove[0].size
        LOGGER.debug("Removed patch %i of %i pixel(s).", label, pixels_to_remove[0].size)

        if (pixel_count != None) and (removed_pixels == pixel_count):
            LOGGER.debug("Filtered maximum of %i pixel(s).", removed_pixels)
            break

    return dst_array

if __name__ == "__main__":
    input_uri = "/home/mlacayo/workspace/data/ScenarioGenerator/input/landcover.tif"
    patch_size = 5
    output_uri = "/home/mlacayo/workspace/data/ScenarioGenerator/despeckled.tif"
    value = 10
    replace = 1
    pixel_count = 2

    filter_fragments_uri(input_uri, value, patch_size, pixel_count, output_uri, replace)
