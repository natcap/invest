"""Raster Class."""

import os
import shutil
import logging

try:
    import gdal
    import ogr
    import osr
except ImportError:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr

import numpy as np
from shapely.geometry import Polygon
import shapely
import pygeoprocessing as pygeo
import pygeoprocessing.geoprocessing as geoprocess

from natcap.invest.coastal_blue_carbon.affine import Affine

LOGGER = logging.getLogger('natcap.invest.coastal_blue_carbon.classes.raster')
logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s \
    %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H/%M/%S')


class Raster(object):

    """A class for interacting with gdal raster files."""

    def __init__(self, uri, driver, resample_method=None):
        self.uri = uri
        self.driver = driver
        self.dataset = None
        self.resample_method = resample_method

    @staticmethod
    def from_array(array, affine, proj, datatype, nodata_val, resample_method=None, driver='GTiff', filepath=None):
        if len(array.shape) is 2:
            num_bands = 1
        elif len(array.shape) is 3:
            num_bands = len(array)
        else:
            raise ValueError

        if filepath:
            dataset_uri = filepath
        else:
            dataset_uri = geoprocess.temporary_filename()
        rows = array.shape[0]
        cols = array.shape[1]

        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(dataset_uri, cols, rows, num_bands, datatype)
        dataset.SetGeoTransform((affine.to_gdal()))

        for band_num in range(num_bands):
            band = dataset.GetRasterBand(band_num + 1)  # Get only raster band
            band.SetNoDataValue(nodata_val)
            if num_bands > 1:
                band.WriteArray(array[band_num])
            else:
                band.WriteArray(array)
            dataset_srs = osr.SpatialReference()
            dataset_srs.ImportFromEPSG(proj)
            dataset.SetProjection(dataset_srs.ExportToWkt())
            band.FlushCache()

        band = None
        dataset_srs = None
        dataset = None
        driver = None

        if not filepath:
            return Raster(
                dataset_uri, resample_method=resample_method, driver=driver)

    @staticmethod
    def from_file(uri, resample_method=None, driver='GTiff'):
        dataset_uri = geoprocess.temporary_filename()
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
        # assert existence
        shutil.copyfile(uri, dataset_uri)
        return Raster(dataset_uri, driver, resample_method=resample_method)

    @staticmethod
    def from_tempfile(uri, driver='GTiff'):
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
        return Raster(uri, driver)

    @staticmethod
    def create_simple_affine(top_left_x, top_left_y, pix_width, pix_height):
        return Affine(pix_width, 0, top_left_x, 0, -(pix_height), top_left_y)

    def _open_dataset(self):
        self.dataset = gdal.Open(self.uri)

    def _close_dataset(self):
        self.dataset = None

    def __del__(self):
        self._delete()

    def __exit__(self):
        self._delete()

    def _delete(self):
        os.remove(self.uri)

    def __str__(self):
        string = '\nRASTER___'
        string += '\nNumber of Bands: ' + str(self.band_count())
        string += '\nBand 1:\n' + self.get_band(1).__repr__()
        string += self.get_affine().__repr__()
        string += '\nNoData for Band 1: ' + str(self.get_nodata(1))
        string += '\nDatatype for Band 1: ' + str(self.get_band(1).dtype)
        string += '\nProjection (EPSG): ' + str(self.get_projection())
        string += '\nuri: ' + self.uri
        string += '\n'
        return string

    def __len__(self):
        return self.band_count()

    def __neg__(self):
        def neg_closure(nodata):
            def neg(x):
                return np.where((np.not_equal(x, nodata)), np.negative(x), nodata)
            return neg
        return self.local_op(None, neg_closure, broadcast=True)

    def __mul__(self, raster):
        if type(raster) in [float, int]:
            def mul_closure(nodata):
                def mul(x):
                    return np.where((np.not_equal(x, nodata)), np.multiply(x, raster), nodata)
                return mul
            return self.local_op(raster, mul_closure, broadcast=True)
        else:
            def mul_closure(nodata):
                def mul(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.multiply(x, y), nodata)
                return mul
            return self.local_op(raster, mul_closure)

    def __rmul__(self, raster):
        if type(raster) in [float, int]:
            def mul_closure(nodata):
                def mul(x):
                    return np.where((np.not_equal(x, nodata)), np.multiply(raster, x), nodata)
                return mul
            return self.local_op(raster, mul_closure, broadcast=True)
        else:
            def mul_closure(nodata):
                def mul(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.multiply(y, x), nodata)
                return mul
            return self.local_op(raster, mul_closure)

    def __div__(self, raster):
        if type(raster) in [float, int]:
            def div_closure(nodata):
                def div(x):
                    return np.where((np.not_equal(x, nodata)), np.divide(x, raster), nodata)
                return div
            return self.local_op(raster, div_closure, broadcast=True)
        else:
            def div_closure(nodata):
                def div(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.divide(x, y), nodata)
                return div
            return self.local_op(raster, div_closure)

    def __rdiv__(self, raster):
        if type(raster) in [float, int]:
            def div_closure(nodata):
                def div(x):
                    return np.where((np.not_equal(x, nodata)), np.divide(raster, x), nodata)
                return div
            return self.local_op(raster, div_closure, broadcast=True)
        else:
            def div_closure(nodata):
                def div(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.divide(y, x), nodata)
                return div
            return self.local_op(raster, div_closure)

    def __add__(self, raster):
        if type(raster) in [float, int]:
            def add_closure(nodata):
                def add(x):
                    return np.where((np.not_equal(x, nodata)), np.add(x, raster), nodata)
                return add
            return self.local_op(raster, add_closure, broadcast=True)
        else:
            def add_closure(nodata):
                def add(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.add(x, y), nodata)
                return add
            return self.local_op(raster, add_closure)

    def __radd__(self, raster):
        if type(raster) in [float, int]:
            def add_closure(nodata):
                def add(x):
                    return np.where((np.not_equal(x, nodata)), np.add(raster, x), nodata)
                return add
            return self.local_op(raster, add_closure, broadcast=True)
        else:
            def add_closure(nodata):
                def add(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.add(y, x), nodata)
                return add
            return self.local_op(raster, add_closure)

    def __sub__(self, raster):
        if type(raster) in [float, int]:
            def sub_closure(nodata):
                def sub(x):
                    return np.where((np.not_equal(x, nodata)), np.subtract(x, raster), nodata)
                return sub
            return self.local_op(raster, sub_closure, broadcast=True)
        else:
            def sub_closure(nodata):
                def sub(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.subtract(x, y), nodata)
                return sub
            return self.local_op(raster, sub_closure)

    def __rsub__(self, raster):
        if type(raster) in [float, int]:
            def sub_closure(nodata):
                def sub(x):
                    return np.where((np.not_equal(x, nodata)), np.subtract(raster, x), nodata)
                return sub
            return self.local_op(raster, sub_closure, broadcast=True)
        else:
            def sub_closure(nodata):
                def sub(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.subtract(y, x), nodata)
                return sub
            return self.local_op(raster, sub_closure)

    def __pow__(self, raster):
        if type(raster) in [float, int]:
            # Implement broadcast operation
            def pow_closure(nodata):
                def powe(x):
                    return np.where((np.not_equal(x, nodata)), np.power(x, raster), nodata)
                return powe
            return self.local_op(raster, pow_closure, broadcast=True)
        else:
            def pow_closure(nodata):
                def powe(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.power(x, y), nodata)
                return powe
            return self.local_op(raster, pow_closure)

    def __rpow__(self, raster):
        if type(raster) in [float, int]:
            # Implement broadcast operation
            def pow_closure(nodata):
                def powe(x):
                    return np.where((np.not_equal(x, nodata)), np.power(raster, x), nodata)
                return powe
            return self.local_op(raster, pow_closure, broadcast=True)
        else:
            def pow_closure(nodata):
                def powe(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.power(y, x), nodata)
                return powe
            return self.local_op(raster, pow_closure)

    def __mod__(self, raster):
        if type(raster) in [float, int]:
            # Implement broadcast operation
            def mod_closure(nodata):
                def mod(x):
                    return np.where((np.not_equal(x, nodata)), np.mod(x, raster), nodata)
                return mod
            return self.local_op(raster, mod_closure, broadcast=True)
        else:
            def mod_closure(nodata):
                def mod(x, y):
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), np.mod(x, y), nodata)
                return mod
            return self.local_op(raster, mod_closure)

    def __eq__(self, raster):
        if type(raster) in [float, int]:
            return (self.get_bands() == raster)
        else:
            if self.is_aligned(raster) and (self.get_shape() == raster.get_shape()):
                return (self.get_bands() == raster.get_bands())
            else:
                return False

    def minimum(self, raster):
        if type(raster) in [float, int]:
            # Implement broadcast operation
            def min_closure(nodata):
                def mini(x):
                    def f(x): return np.where(np.not_equal(x, raster), np.minimum(x, raster), np.minimum(x, raster))
                    return np.where((np.not_equal(x, nodata)), f(x), nodata)
                return mini
            return self.local_op(raster, min_closure, broadcast=True)
        else:
            def min_closure(nodata):
                def mini(x, y):
                    def f(x, y): return np.where(np.not_equal(x, y), np.minimum(x, y), np.minimum(x, y))
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), f(x, y), nodata)
                return mini
            return self.local_op(raster, min_closure)

    def fminimum(self, raster):
        if type(raster) in [float, int]:
            # Implement broadcast operation
            def min_closure(nodata):
                def mini(x):
                    def f(x): return np.where(np.not_equal(x, raster), np.fmin(x, raster), np.fmin(x, raster))
                    return np.where((np.not_equal(x, nodata)), f(x), nodata)
                return mini
            return self.local_op(raster, min_closure, broadcast=True)
        else:
            def min_closure(nodata):
                def mini(x, y):
                    def f(x, y): return np.where((np.not_equal(x, y)), np.fmin(x, y), np.fmin(x, y))
                    return np.where((np.not_equal(x, nodata)) & (np.not_equal(y, nodata)), f(x, y), nodata)
                return mini
            return self.local_op(raster, min_closure)

    def __getitem__(self, key):
        arr = self.get_band(1)
        return arr[key]

    def __setitem__(self, key, item):
        arr = self.get_band(1)
        arr[key] = item
        self.set_band(1, arr)

    def getslice(self, a, b, c):
        pass

    def setslice(self, a, b, c):
        pass

    def __iter__(self):
        pass  # iterate over bands?

    def __contains__(self):
        pass  # test numpy raster against all bands?

    def __repr__(self):
        return self.get_bands().__repr__()

    def _repr_png_(self):
        raise NotImplementedError

    def save_raster(self, uri):
        shutil.copyfile(self.uri, uri)

    def get_heatmap_image(self):
        raise NotImplementedError

    def min(self):
        pygeo.calculate_raster_stats_uri(self.uri)
        mini, _, _, _ = pygeo.get_statistics_from_uri(self.uri)
        return mini

    def max(self):
        pygeo.calculate_raster_stats_uri(self.uri)
        _, maxi, _, _ = pygeo.get_statistics_from_uri(self.uri)
        return maxi

    def mean(self):
        pygeo.calculate_raster_stats_uri(self.uri)
        _, _, mean, _ = pygeo.get_statistics_from_uri(self.uri)
        return mean

    def std(self):
        pygeo.calculate_raster_stats_uri(self.uri)
        _, _, _, std = pygeo.get_statistics_from_uri(self.uri)
        return std

    def unique(self):
        unique_vals_list = pygeo.unique_raster_values_uri(self.uri)
        return unique_vals_list

    def ones(self):
        def ones_closure(nodata):
            def ones(x):
                return np.where(x == x, 1, nodata)
            return ones
        return self.local_op(0, ones_closure, broadcast=True)

    def zeros(self):
        def zeros_closure(nodata):
            def zeros(x):
                return np.where(x == x, 0, nodata)
            return zeros
        return self.local_op(0, zeros_closure, broadcast=True)

    def band_count(self):
        self._open_dataset()
        count = self.dataset.RasterCount
        self._close_dataset()
        return count

    def get_band(self, band_num):
        a = None
        self._open_dataset()

        if band_num >= 1 and band_num <= self.dataset.RasterCount:
            band = self.dataset.GetRasterBand(band_num)
            a = band.ReadAsArray()
            nodata_val = band.GetNoDataValue()
            a = np.ma.masked_equal(a, nodata_val)
            band = None
        else:
            pass

        self._close_dataset()
        return a

    def get_bands(self):
        self._open_dataset()

        if self.dataset.RasterCount == 0:
            return None

        a = np.zeros((
            self.dataset.RasterCount,
            self.dataset.RasterYSize,
            self.dataset.RasterXSize))

        for num in np.arange(self.dataset.RasterCount):
            band = self.dataset.GetRasterBand(num+1)
            b = band.ReadAsArray()
            nodata_val = band.GetNoDataValue()
            b = np.ma.masked_equal(b, nodata_val)
            a[num] = b

        self._close_dataset()
        return a

    def get_resample_method(self):
        if not self.resample_method:
            raise AttributeError(
                'Raster object has no assigned resample_method attribute')
        else:
            return self.resample_method

    def get_nodata(self, band_num):
        nodata_val = None
        self._open_dataset()

        if band_num >= 1 and band_num <= self.dataset.RasterCount:
            band = self.dataset.GetRasterBand(band_num)
            nodata_val = band.GetNoDataValue()

        self._close_dataset()
        return nodata_val

    def get_datatype(self, band_num):
        datatype = None
        self._open_dataset()

        if band_num >= 1 and band_num <= self.dataset.RasterCount:
            band = self.dataset.GetRasterBand(band_num)
            datatype = band.DataType

        self._close_dataset()
        return datatype

    def get_rows(self):
        rows = None
        self._open_dataset()

        rows = self.dataset.RasterYSize

        self._close_dataset()
        return rows

    def get_cols(self):
        cols = None
        self._open_dataset()

        cols = self.dataset.RasterXSize

        self._close_dataset()
        return cols

    def get_pixel_value_at_pixel_indices(self, px, py):
        """Position relative to origin regardless of affine transform."""
        pix = None
        self._open_dataset()

        try:
            # Assertions
            band = self.dataset.GetRasterBand(1)
            pix = band.ReadAsArray(px, py, 1, 1)
        finally:
            self._close_dataset()

        return pix

    def get_georef_point_at_pixel_indices(self, px, py):
        """Georeferenced point of pixel center."""
        a = self.get_affine()
        gx = (a.a * (px + 0.5)) + (a.b * (py + 0.5)) + a.c
        gy = (a.e * (py + 0.5)) + (a.d * (px + 0.5)) + a.f
        return (gx, gy)

    def get_shapely_point_at_pixel_indices(self, px, py):
        """Georeferenced point of pixel center."""
        gx, gy = self.get_georef_point_at_pixel_indices(px, py)
        return shapely.geometry.point.Point(gx, gy)

    def get_pixel_indices_at_georef_point(self, gx, gy):
        """note: this may only apply to non-rotated rasters."""
        gt = self.get_geotransform()
        px = int((gx - gt[0]) / gt[1])
        py = int((gy - gt[3]) / gt[5])
        return (px, py)

    def get_pixel_indices_at_shapely_point(self, shapely_point):
        """note: this may only apply to non-rotated rasters."""
        return self.get_pixel_indices_at_georef_point(
            shapely_point.x, shapely_point.y)

    def get_pixel_value_at_georef_point(self, gx, gy):
        px, py = self.get_pixel_indices_at_georef_point(gx, gy)
        return self.get_pixel_value_at_pixel_indices(px, py)

    def get_pixel_value_at_shapely_point(self, shapely_point):
        gx, gy = shapely_point.x, shapely_point.y
        return self.get_pixel_value_at_georef_point(gx, gy)

    def get_shape(self):
        rows = self.get_rows()
        cols = self.get_cols()
        return (rows, cols)

    def get_projection(self):
        self._open_dataset()
        RasterSRS = osr.SpatialReference()
        RasterSRS.ImportFromWkt(self.dataset.GetProjectionRef())
        proj = int(RasterSRS.GetAttrValue("AUTHORITY", 1))

        RasterSRS = None
        self._close_dataset()

        return proj

    def get_projection_wkt(self):
        self._open_dataset()
        wkt = self.dataset.GetProjectionRef()
        self._close_dataset()
        return wkt

    def get_geotransform(self):
        geotransform = None
        self._open_dataset()

        geotransform = self.dataset.GetGeoTransform()

        self._close_dataset()
        return geotransform

    def get_affine(self):
        geotransform = self.get_geotransform()
        return Affine.from_gdal(*geotransform)

    def get_bounding_box(self):
        return geoprocess.get_bounding_box(self.uri)

    def get_aoi(self):
        """May only be suited for non-rotated rasters."""
        bb = self.get_bounding_box()
        u_x = max(bb[0::2])
        l_x = min(bb[0::2])
        u_y = max(bb[1::2])
        l_y = min(bb[1::2])
        return Polygon([(l_x, l_y), (l_x, u_y), (u_x, u_y), (u_x, l_y)])

    def get_aoi_as_shapefile(self, uri):
        """May only be suited for non-rotated rasters."""
        raise NotImplementedError

    def get_cell_area(self):
        a = self.get_affine()
        return abs((a.a * a.e))

    def set_band(self, band_num, masked_array):
        affine = self.get_affine()
        proj = self.get_projection()
        datatype = self.get_datatype(band_num)
        nodata_val = masked_array.fill_value
        if np.issubdtype(nodata_val, int):
            nodata_val = nodata_val.astype(int)
        else:
            nodata_val = nodata_val.astype(float)
        uri = self.uri
        Raster.from_array(
            masked_array, affine, proj, datatype, nodata_val, filepath=uri)

    def set_bands(self, array):
        raise NotImplementedError

    def set_datatype(self, datatype):

        def pixel_op_closure(nodata):
            def copy(x):
                return np.where(x == x, x, nodata)
            return copy

        bounding_box_mode = "dataset"
        resample_method = "nearest"

        dataset_uri_list = [self.uri]
        resample_list = [resample_method]

        nodata = self.get_nodata(1)
        pixel_op = pixel_op_closure(nodata)
        dataset_out_uri = geoprocess.temporary_filename()
        datatype_out = datatype
        nodata_out = nodata
        pixel_size_out = geoprocess.get_cell_size_from_uri(self.uri)

        geoprocess.vectorize_datasets(
            dataset_uri_list,
            pixel_op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=resample_list,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return Raster.from_tempfile(dataset_out_uri)

    def set_nodata(self, nodata_val):

        def pixel_op_closure(old_nodata, new_nodata):
            def copy(x):
                return np.where(x == old_nodata, new_nodata, x)
            return copy

        bounding_box_mode = "dataset"
        resample_method = "nearest"

        dataset_uri_list = [self.uri]
        resample_list = [resample_method]

        old_nodata = self.get_nodata(1)
        pixel_op = pixel_op_closure(old_nodata, nodata_val)
        dataset_out_uri = geoprocess.temporary_filename()
        datatype_out = geoprocess.get_datatype_from_uri(self.uri)
        nodata_out = nodata_val
        pixel_size_out = geoprocess.get_cell_size_from_uri(self.uri)

        geoprocess.vectorize_datasets(
            dataset_uri_list,
            pixel_op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=resample_list,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return Raster.from_tempfile(dataset_out_uri)

    def set_datatype_and_nodata(self, datatype, nodata_val):

        def pixel_op_closure(old_nodata, new_nodata):
            def copy(x):
                return np.where(x == old_nodata, new_nodata, x)
            return copy

        bounding_box_mode = "dataset"
        resample_method = "nearest"

        dataset_uri_list = [self.uri]
        resample_list = [resample_method]

        old_nodata = self.get_nodata(1)
        pixel_op = pixel_op_closure(old_nodata, nodata_val)
        dataset_out_uri = geoprocess.temporary_filename()
        datatype_out = datatype
        nodata_out = nodata_val
        pixel_size_out = geoprocess.get_cell_size_from_uri(self.uri)

        geoprocess.vectorize_datasets(
            dataset_uri_list,
            pixel_op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=resample_list,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return Raster.from_tempfile(dataset_out_uri)

    def copy(self, uri=None):
        if not uri:
            uri = geoprocess.temporary_filename()
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
        shutil.copyfile(self.uri, uri)
        return Raster.from_tempfile(uri, driver=self.driver)

    def is_aligned(self, raster):
        try:
            this_affine = self.get_affine()
            other_affine = raster.get_affine()
            return (this_affine == other_affine)
        except:
            raise TypeError

    def align(self, raster, resample_method):
        """Currently aligns other raster to this raster - later: union/intersection."""
        if not self.get_projection() == raster.get_projection():
            raise AssertionError("Different Projections")

        def dataset_pixel_op(x, y): return y
        dataset_uri_list = [self.uri, raster.uri]
        dataset_out_uri = geoprocess.temporary_filename()
        datatype_out = geoprocess.get_datatype_from_uri(raster.uri)
        nodata_out = geoprocess.get_nodata_from_uri(raster.uri)
        pixel_size_out = geoprocess.get_cell_size_from_uri(self.uri)
        bounding_box_mode = "dataset"

        geoprocess.vectorize_datasets(
            dataset_uri_list,
            dataset_pixel_op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=[resample_method]*2,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return Raster.from_tempfile(dataset_out_uri)

    def align_to(self, raster, resample_method):
        """Currently aligns self to provided raster - later: union/intersection"""
        if not self.get_projection() == raster.get_projection():
            raise AssertionError("Different Projections")

        def dataset_pixel_op(x, y): return y

        dataset_uri_list = [raster.uri, self.uri]
        dataset_out_uri = geoprocess.temporary_filename()
        datatype_out = geoprocess.get_datatype_from_uri(self.uri)
        nodata_out = geoprocess.get_nodata_from_uri(self.uri)
        pixel_size_out = geoprocess.get_cell_size_from_uri(raster.uri)
        bounding_box_mode = "dataset"

        geoprocess.vectorize_datasets(
            dataset_uri_list,
            dataset_pixel_op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=[resample_method]*2,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return Raster.from_tempfile(dataset_out_uri)

    def clip(self, aoi_uri):
        r = None
        dataset_out_uri = geoprocess.temporary_filename()
        datatype = self.get_datatype(1)
        nodata = self.get_nodata(1)
        pixel_size = self.get_affine().a

        try:
            geoprocess.clip_dataset_uri(
                self.uri,
                aoi_uri,
                dataset_out_uri,
                assert_projections=False)
            return Raster.from_tempfile(dataset_out_uri)
        except:
            os.remove(dataset_out_uri)

            ds = ogr.Open(aoi_uri)
            layer = ds.GetLayer()
            feature = layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            wkt = geom.ExportToWkt()
            shapely_object = shapely.wkt.loads(wkt)
            centroid = shapely_object.centroid
            value = self.get_pixel_value_at_shapely_point(centroid)

            if value is None:
                return None

            px, py = self.get_pixel_indices_at_shapely_point(centroid)
            src_af = self.get_affine()
            mx1 = src_af.c
            my1 = src_af.f
            sign_x = np.sign(src_af.a)
            sign_y = np.sign(src_af.e)
            mx2 = (src_af.a * px) + mx1
            my2 = (src_af.e * py) + my1
            affine = Affine(
                src_af.a,
                src_af.b,
                mx2,
                src_af.d,
                src_af.e,
                my2)

            r = Raster.from_array(
                value,
                affine,
                self.get_projection(),
                self.get_datatype(1),
                self.get_nodata(1))
            return r

    def reproject(self, proj, resample_method, pixel_size=None):
        if pixel_size is None:
            pixel_size = self.get_affine().a

        dataset_out_uri = geoprocess.temporary_filename()
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(proj)
        wkt = srs.ExportToWkt()

        geoprocess.reproject_dataset_uri(
            self.uri, pixel_size, wkt, resample_method, dataset_out_uri)

        return Raster.from_tempfile(dataset_out_uri)

    def resize_pixels(self, pixel_size, resample_method):
        bounding_box = self.get_bounding_box()
        output_uri = geoprocess.temporary_filename()

        geoprocess.resize_and_resample_dataset_uri(
            self.uri,
            bounding_box,
            pixel_size,
            output_uri,
            resample_method)

        return Raster.from_tempfile(output_uri)

    def reclass_masked_values(self, mask_raster, new_value):
        def reclass_masked_closure(nodata):
            def reclass(x, y):
                return np.where((np.not_equal(y, 0)), x, new_value)
            return reclass
        return self.local_op(mask_raster, reclass_masked_closure)

    def sample_from_raster(self, raster):
        '''way too slow!'''
        shape = self.get_shape()
        a = np.zeros(shape)
        dst_idxs = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
        src_raster = raster.get_band(1).data
        for dst_idx in dst_idxs:
            p = self.get_georef_point_at_pixel_indices(*dst_idx)
            p2 = self.reproject_georef_point(
                p[0], p[1], raster.get_projection())
            src_idx = self.get_pixel_indices_at_georef_point(*p2)
            a[dst_idx] = src_raster[src_idx]

        r = Raster.from_array(
            a,
            self.get_affine(),
            self.get_projection(),
            raster.get_datatype(1),
            raster.get_nodata(1))
        return r

    def reclass(self, reclass_table, out_nodata=None, out_datatype=None):
        if out_nodata is None:
            out_nodata = geoprocess.get_nodata_from_uri(self.uri)
        if out_datatype is None:
            out_datatype = geoprocess.get_datatype_from_uri(self.uri)
        dataset_out_uri = geoprocess.temporary_filename()

        geoprocess.reclassify_dataset_uri(
            self.uri,
            reclass_table,
            dataset_out_uri,
            out_datatype,
            out_nodata,
            assert_dataset_projected=False)

        return Raster.from_tempfile(dataset_out_uri)

    def overlay(self, raster):
        raise NotImplementedError

    def to_binary_raster(self, val):
        values_list = list(set(self.get_band(1).data.flatten()))
        reclass_list = [(v, 0) for v in values_list]
        reclass_dict = dict(reclass_list)
        if val in values_list:
            print val
            reclass_dict[val] = 1
        return self.reclass(reclass_dict)

    def local_op(self, raster, pixel_op_closure, broadcast=False):
        bounding_box_mode = "dataset"
        resample_method = "nearest"

        if not broadcast:
            if not self.is_aligned(raster):
                raise AssertionError("Not Aligned")
            if not self.get_nodata(1) == raster.get_nodata(1):
                raise AssertionError("Rasters have different nodata values: %f, %f" % (
                    self.get_nodata(1), raster.get_nodata(1)))
            dataset_uri_list = [self.uri, raster.uri]
            resample_list = [resample_method]*2
        else:
            dataset_uri_list = [self.uri]
            resample_list = [resample_method]

        nodata = self.get_nodata(1)
        pixel_op = pixel_op_closure(nodata)
        dataset_out_uri = geoprocess.temporary_filename()
        datatype_out = geoprocess.get_datatype_from_uri(self.uri)
        nodata_out = geoprocess.get_nodata_from_uri(self.uri)
        pixel_size_out = geoprocess.get_cell_size_from_uri(self.uri)

        geoprocess.vectorize_datasets(
            dataset_uri_list,
            pixel_op,
            dataset_out_uri,
            datatype_out,
            nodata_out,
            pixel_size_out,
            bounding_box_mode,
            resample_method_list=resample_list,
            dataset_to_align_index=0,
            dataset_to_bound_index=0,
            assert_datasets_projected=False,
            vectorize_op=False)

        return Raster.from_tempfile(dataset_out_uri)
