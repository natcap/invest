'''
Vector Class
'''

import os
import shutil

try:
    import gdal
    import ogr
    import osr
except:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr

from shapely.geometry import *
import shapely.wkt
import pygeoprocessing as pygeo


class Vector(object):
    # any global variables here
    def __init__(self, uri, driver):
        self.uri = uri
        self.driver = driver
        self.datasource = None

    @classmethod
    def from_shapely(self, shapely_object, proj, driver='ESRI Shapefile'):
        shapely_to_ogr = {
            'Point': ogr.wkbPoint,
            'LineString': ogr.wkbLineString,
            'Polygon': ogr.wkbPolygon,
            'MultiPoint': ogr.wkbMultiPoint,
            'MultiLineString': ogr.wkbMultiLineString,
            'MultiPolygon': ogr.wkbMultiPolygon
        }

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(proj)

        datasource_uri = pygeo.geoprocessing.temporary_filename(suffix='.shp')
        if os.path.exists(datasource_uri):
            os.remove(datasource_uri)

        driver = ogr.GetDriverByName(driver)
        datasource = driver.CreateDataSource(datasource_uri)
        layer = datasource.CreateLayer(
            '', srs, shapely_to_ogr[mapping(shapely_object)['type']])
        defn = layer.GetLayerDefn()
        feature = ogr.Feature(defn)
        geometry = ogr.CreateGeometryFromWkt(shapely_object.wkt)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)

        layer = feature = geometry = None
        datasource = None

        return Vector(datasource_uri, driver=driver)

    @classmethod
    def from_file(self, uri, driver='ESRI Shapefile'):
        dst_uri = pygeo.geoprocessing.temporary_filename()
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
        src_uri = os.path.splitext(uri)[0]
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            if os.path.exists(src_uri+ext):
                shutil.copyfile(src_uri+ext, dst_uri+ext)
        return Vector(dst_uri, driver)

    @classmethod
    def from_tempfile(self, uri, driver='ESRI Shapefile'):
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
        return Vector(uri, driver)

    def __del__(self):
        self._delete()

    def __exit__(self):
        self._delete()

    def _delete(self):
        fp = os.path.splitext(self.uri)[0]
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            os.remove(fp+ext)

    def __str__(self):
        return "<vector object at " + self.uri + ">"

    def __len__(self):
        return self.feature_count()

    def __eq__(self, vector):
        raise NotImplementedError

    def __getitem__(self):
        pass  # return numpy slice?  Raster object with sliced numpy array?

    def __setitem__(self):
        pass  # set numpy values to raster

    def __getslice__(self):
        pass

    def __setslice__(self):
        pass

    def __iter__(self):
        pass  # iterate over bands?

    def __contains__(self):
        pass  # test numpy raster against all bands?

    def __repr__(self):
        raise NotImplementedError

    def _repr_svg_(self):
        return self.get_geometry()._repr_svg_()

    def save_vector(self, uri):
        src_uri = os.path.splitext(self.uri)[0]
        dst_uri = os.path.splitext(uri)[0]
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            if os.path.exists(src_uri+ext):
                shutil.copyfile(src_uri+ext, dst_uri+ext)

    def feature_count(self):
        self._open_datasource()
        layer = self.datasource.GetLayer()
        count = layer.GetFeatureCount()

        layer = None
        self._close_datasource()
        return count

    def get_geometry(self):
        self._open_datasource()
        layer = self.datasource.GetLayer()
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        wkt = geom.ExportToWkt()
        shapely_object = shapely.wkt.loads(wkt)

        layer = feature = geom = None
        self._close_datasource()

        return shapely_object

    def get_geometry_centroid(self):
        centroid = self.get_geometry().centroid
        return centroid

    def get_geometries(self, layer):
        raise NotImplementedError

    def get_geometry_type(self, layer, feature_num):
        raise NotImplementedError

    def get_feature(self):
        raise NotImplementedError

    def get_features(self):
        raise NotImplementedError

    def get_projection(self):
        raise NotImplementedError
        # self._open_datasource()
        # layer = self.datasource.GetLayer()
        # print layer.GetSpatialRef().GetAuthorityCode(0)
        # wkt = layer.GetSpatialRef().ExportToWkt()
        # layer = None
        # self._close_datasource()
        # return wkt
        # RasterSRS = osr.SpatialReference()
        # RasterSRS.ImportFromWkt(layer.GetSpatialRef().ExportToWkt())
        # return int(RasterSRS.GetAttrValue("AUTHORITY", 1))

    def get_projection_wkt(self):
        self._open_datasource()
        layer = self.datasource.GetLayer()
        wkt = layer.GetSpatialRef().ExportToWkt()
        layer = None
        self._close_datasource()
        return wkt
        # RasterSRS = osr.SpatialReference()
        # RasterSRS.ImportFromWkt(layer.GetSpatialRef().ExportToWkt())
        # return int(RasterSRS.GetAttrValue("AUTHORITY", 1))

    def get_aoi(self):
        raise NotImplementedError

    def copy(self, uri):
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
        fp = os.path.splitext(self.uri)[0]
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            if os.path.exists(fp+ext):
                shutil.copyfile(fp+ext, uri+ext)
        return Vector.from_tempfile(uri)

    def clip(self, aoi_uri):
        raise NotImplementedError

    def reproject(self, proj):
        datasource_out_uri = pygeo.geoprocessing.temporary_filename('.shp')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(proj)
        wkt = srs.ExportToWkt()

        pygeo.geoprocessing.reproject_datasource_uri(
            self.uri, wkt, datasource_out_uri)

        return Vector.from_tempfile(datasource_out_uri)

    def reproject_wkt(self, wkt):
        datasource_out_uri = pygeo.geoprocessing.temporary_filename('.shp')

        pygeo.geoprocessing.reproject_datasource_uri(
            self.uri, wkt, datasource_out_uri)

        return Vector.from_tempfile(datasource_out_uri)

    def overlay(self, raster):
        raise NotImplementedError

    # Operations
    #   attribute data and tables
    #   buffer
    #   densify
    #   union
    #   intersection
    #   symetric difference
    #

    def to_raster(self):
        raise NotImplementedError

    def _open_datasource(self):
        # driver = ogr.GetDriverByName(self.driver)
        self.datasource = ogr.Open(self.uri)

    def _close_datasource(self):
        self.datasource = None
