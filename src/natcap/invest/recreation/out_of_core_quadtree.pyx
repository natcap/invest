# cython: profile=True
"""This module defines a hierarchical spatial indexing data structure amenable
    for fast culling of points in 2D space"""

import os
import sys
import time
import struct
import pickle
import uuid
import itertools
import collections
import traceback
import bisect
import operator
import shapely.geometry
import numpy
import logging

from osgeo import ogr
from osgeo import osr
cimport numpy

MAX_BYTES_TO_BUFFER = 2**27  # buffer a little over 128 megabytes

import buffered_file_manager

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger(
    'natcap.invest.recmodel_server.out_of_core_quadtree')


class OutOfCoreQuadTree(object):
    """An out of core quad tree spatial indexing structure.  Define with an
        initial range and continously add new points."""

    next_available_blob_id = 0

    def __init__(
            self, bounding_box, max_points_per_node, max_node_depth,
            quad_tree_storage_dir, node_depth=0, file_manager=None,
            pickle_filename=None):
        """Make a new quadtree node with a given initial_bounding_box range

            bounding_box - list of [x_min, y_min x_max, y_max]
            max_points_per_node - maximum number of points before a node is
                subdivided
            max_node_depth - max nodes to make the quadtree
            pickle_filename - name of file on disk which to pickle the tree to
                during a flush"""

        self.bounding_box = list(bounding_box)  # make a copy to avoid aliasing
        self.max_points_per_node = max_points_per_node
        self.max_node_depth = max_node_depth
        self.node_depth = node_depth
        self.n_points_in_node = 0
        self.is_leaf = True  # boolean to determine if this node stores data
        self.nodes = None  # a list of children on an internal node
        self.quad_tree_storage_dir = quad_tree_storage_dir
        if file_manager is None:
            self.file_manager = buffered_file_manager.BufferedFileManager(
                pickle_filename+'.db', MAX_BYTES_TO_BUFFER)
        else:
            self.file_manager = file_manager

        self.pickle_filename = pickle_filename

        #Unique blob_id
        self.blob_id = OutOfCoreQuadTree.next_available_blob_id
        OutOfCoreQuadTree.next_available_blob_id += 1

    def flush(self):
        """Flushes the file manager in anticipation of a pickle"""
        self.file_manager.flush()
        if self.pickle_filename is not None:
            pickle.dump(self, open(self.pickle_filename, 'wb'))

    def build_node_shapes(self, ogr_polygon_layer):
        """a helper function to visualize the quad tree

            ogr_polygon_layer - a open and writable ogr polygon layer with
                a field to indicate the depth of the tree"""

        # Create a new feature, setting the field and geometry
        if self.is_leaf:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(self.bounding_box[0], self.bounding_box[3])
            ring.AddPoint(self.bounding_box[0], self.bounding_box[1])
            ring.AddPoint(self.bounding_box[2], self.bounding_box[1])
            ring.AddPoint(self.bounding_box[2], self.bounding_box[3])
            ring.AddPoint(self.bounding_box[0], self.bounding_box[3])
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            feature = ogr.Feature(ogr_polygon_layer.GetLayerDefn())
            feature.SetGeometry(poly)
            feature.SetField('n_points', self.n_points_in_node)
            feature.SetField('bb_box', str(self.bounding_box))
            ogr_polygon_layer.CreateFeature(feature)
        else:
            for node_index in xrange(4):
                self.nodes[node_index].build_node_shapes(ogr_polygon_layer)

    def _get_points_from_node(self):
        """Internal function to get the points in a node as a list of
            tuples.

            returns a numpy array of the form [(data, x_coord, y_coord), ...]
            """
        return self.file_manager.read(self.blob_id)

    def _drain_node(self):
        """Internal function called when the current node needs to split, it
            loads the current node into a list of tuples and deletes
            the self.point_file file

            returns a list of (data, x_coord, y_coord)"""

        userday_tuples = self._get_points_from_node()

        #delete the file because it's drained
        self.file_manager.delete(self.blob_id)
        self.blob_id = None
        return userday_tuples

    def _split_node(self):
        """This splits a node into 4 quads and adds those points to the
            subquads

            returns nothing """

        point_list = self._drain_node()
        mid_x_coord = (self.bounding_box[0] + self.bounding_box[2]) / 2.
        mid_y_coord = (self.bounding_box[1] + self.bounding_box[3]) / 2.

        #make children node indexes arranged like this:
        #01
        #23
        #[x_min, y_min x_max, y_max]

        bounding_quads = [
            [self.bounding_box[0],  # xmin
             mid_y_coord,  # ymin
             mid_x_coord,  # xmax
             self.bounding_box[3]],
            [mid_x_coord,  # xmin
             mid_y_coord,  # ymin
             self.bounding_box[2],  # xmax
             self.bounding_box[3]],
            [self.bounding_box[0],  # xmin
             self.bounding_box[1],  # ymin
             mid_x_coord,  # xmax
             mid_y_coord],  # ymax
            [mid_x_coord,  # xmin
             self.bounding_box[1],  # ymin
             self.bounding_box[2],  # xmax
             mid_y_coord],  # ymax
        ]
        self.nodes = []
        for bounding_box in bounding_quads:
            self.nodes.append(OutOfCoreQuadTree(
                bounding_box,
                self.max_points_per_node,
                self.max_node_depth,
                self.quad_tree_storage_dir,
                node_depth=self.node_depth+1,
                file_manager=self.file_manager))

        self.n_points_in_node = 0  # points are drained out of this node
        self.is_leaf = False
        #now when we add these points to the node they'll be sorted and passed
        #to the subquads
        self.add_points(point_list, 0, point_list.size)

    def add_points(
            self, numpy.ndarray point_list, int left_bound, int right_bound):
        """Add a list of points to the current node, this function will split
            the current node if necessary, and split the point list to
            children nodes if necessary.

            point_list - a numpy array of (user_day_hash, x_coord, y_coord)
                tuples
            left_bound (int): left index inclusive of points to consider under
                `point_list`
            right_bound (int): right index non-inclusive of points to consider
                under `point_list`

            returns nothing"""
        cdef float x_coord, y_coord, mid_x_coord, mid_y_coord
        cdef int point_list_len

        if right_bound == left_bound:
            return

        point_list_len = right_bound - left_bound
        if self.is_leaf:
            if ((point_list_len + self.n_points_in_node) <=
                    self.max_points_per_node or
                    self.node_depth == self.max_node_depth):
                #if it's a leaf and the points fit, append to file end
                self.n_points_in_node += point_list_len
                self.file_manager.append(
                    self.blob_id, point_list[left_bound:right_bound])
                return  # this lets us pass through the two cases below
            else:
                #this cases handles a leaf node that needs to be split
                self._split_node()

        #split the list into the four quads
        mid_x_coord = self.nodes[0].bounding_box[2]
        mid_y_coord = self.nodes[0].bounding_box[1]

        cdef int left_y_split_index, x_split_index, right_y_split_index
        _sort_list_to_quads(
            point_list, left_bound, right_bound, mid_x_coord,
            mid_y_coord, &left_y_split_index, &x_split_index,
            &right_y_split_index)

        # quads organized like this:
        #01
        #23

        if left_bound != left_y_split_index:
            self.nodes[2].add_points(
                point_list, left_bound, left_y_split_index)
        if left_y_split_index != x_split_index:
            self.nodes[0].add_points(
                point_list, left_y_split_index, x_split_index)
        if x_split_index != right_y_split_index:
            self.nodes[3].add_points(
                point_list, x_split_index, right_y_split_index)
        if right_y_split_index != right_bound:
            self.nodes[1].add_points(
                point_list, right_y_split_index, right_bound)

    def _bounding_box_intersect(self, bb):
        """Function that tests if this node's bounding intersects another.

            bb - bounding box of form [xmin, ymin, xmax, ymax]

            returns True if self.bounding_box intersects with bb, False
                otherwise"""

        return not (
            self.bounding_box[0] > bb[2] or
            self.bounding_box[2] < bb[0] or
            self.bounding_box[1] > bb[3] or
            self.bounding_box[3] < bb[1])

    def n_nodes(self):
        """Returns the number of nodes in the quadtree"""

        if self.is_leaf:
            return 1
        return sum([self.nodes[index].n_nodes() for index in xrange(4)]) + 1

    def n_points(self):
        """Returns the number of nodes in the quadtree"""

        if self.is_leaf:
            return self.n_points_in_node
        return sum([self.nodes[index].n_points() for index in xrange(4)])

    def get_intersecting_points_in_polygon(self, shapely_polygon):
        """This function is a high performance test routine to return the points
            contained in the shapely_prepared_polygon that exist in the QT

            shapely_polygon - a shapely polygon to bound against

            return deque of (data, x_coord, y_coord) of nodes that are contained
                in shapely_prepared_polygon"""

        bounding_polygon = shapely.geometry.box(*self.bounding_box)

        if self.is_leaf:
            if shapely_polygon.contains(bounding_polygon):
                #trivial, all points are in the poly
                return self._get_points_from_node()
            elif shapely_polygon.intersects(bounding_polygon):
                #tricky, some points might be in poly
                result_deque = collections.deque()
                shapely_prepared_polygon = shapely.prepared.prep(
                    shapely_polygon)
                #I'm hoping if I clip against the BB first it will be faster
                polygon_box = shapely.geometry.box(
                    *shapely_polygon.bounds)
                for point in self._get_points_from_node():
                    shapely_point = shapely.geometry.Point(point[1], point[2])
                    if (polygon_box.contains(shapely_point) and
                            shapely_prepared_polygon.contains(shapely_point)):
                        result_deque.append(point)
                return result_deque
        elif shapely_polygon.intersects(bounding_polygon):
            #combine results of children
            result_deque = collections.deque()
            for node_index in xrange(4):
                result_deque.extend(
                    self.nodes[node_index].get_intersecting_points_in_polygon(
                        shapely_polygon))
            return result_deque

        return collections.deque()  # empty

    def get_intersecting_points_in_bounding_box(self, bounding_box):
        """This function takes in a bounding box and returns a list of
            (data, lat, lng) tuples that are contained in the leaf nodes that
            intersect that bounding box.

            bounding_box - list of the form [xmin, ymin, xmax, ymax]

            returns numpy array of (data, x_coord, lng) of nodes that intersect
                the bounding box"""

        if not self._bounding_box_intersect(bounding_box):
            return numpy.empty(0, dtype='a4,f4,f4')

        if self.is_leaf:
            #drain the node into a list, filter to current bounding box
            point_list = numpy.array([
                point for point in self._get_points_from_node() if _in_box(
                    bounding_box, point[1], point[2])], dtype='a4,f4,f4')
            return point_list
        else:
            point_list = numpy.empty(0, dtype='a4,f4,f4')
            for node_index in xrange(4):
                point_list = numpy.concatenate((
                    self.nodes[node_index].get_intersecting_points_in_bounding_box(
                        bounding_box), point_list))
            return point_list


cdef _sort_list_to_quads(
        numpy.ndarray point_list, int left_bound, int right_bound,
        float mid_x_coord, float mid_y_coord, int *left_y_split_index,
        int *x_split_index, int *right_y_split_index):
    """Sorts the points in point_list to be 4 sublists where each sublists's
        points lie within the

        01
        23

        quads.

        point_list - structured numpy array of (data, x_coord, y_coord) points.
            This parameter will be modified to have sorted points
        left_bound (int): left inclusive index in `point_list` to sort
        right_bound (int): right non-inclusive index in `point_list` to sort
        list_bounds - (left, right) tuple of valid points in point_list
        mid_x_coord - x coord to split the quad
        mid_y_coord - y coord to split the quad


        modifies left_y_split_index, x_split_index, right_y_split_index to
        report split"""

    #sort by x coordinates
    cdef numpy.ndarray sub_array = point_list[left_bound:right_bound]
    cdef numpy.ndarray idx = sub_array['f1'].argsort()

    sub_array[:] = sub_array[idx]
    x_split_index[0] = sub_array['f1'].searchsorted(mid_x_coord) + left_bound

    #sort the left y coordinates
    sub_array = point_list[left_bound:x_split_index[0]]
    idx = sub_array['f2'].argsort()
    sub_array[:] = sub_array[idx]
    left_y_split_index[0] = sub_array['f2'].searchsorted(
        mid_y_coord) + left_bound

    #sort the right y coordinates
    sub_array = point_list[x_split_index[0]:right_bound]
    idx = sub_array['f2'].argsort()
    sub_array[:] = sub_array[idx]
    right_y_split_index[0] = sub_array['f2'].searchsorted(
        mid_y_coord) + x_split_index[0]


def _in_box(bounding_box, x_coord, y_coord):
    """Returns true if the (x_coord, y_coord) coord is contained in the
        bounding box.

        bounding_box - list of form [xmin, ymin, xmax, ymax]
        x_coord, y_coord - x and y coordinate to test.

        return Ture if (x_coord, y_coord) bound in bounding_box

        """
    return (
        x_coord >= bounding_box[0] and  # xmin
        y_coord >= bounding_box[1] and  # ymin
        x_coord < bounding_box[2] and  # xmax
        y_coord < bounding_box[3])  # ymax
