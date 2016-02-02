"""Buffered file manager module."""

import uuid
import time
import collections
import os
import logging
import sqlite3

import numpy
import pygeoprocessing

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_server.buffered_file_manager')


class BufferedFileManager(object):
    """File manager to buffers reads and writes to mitigate file IO."""

    _ARRAY_TUPLE_TYPE = numpy.dtype('datetime64[D],a4,f4,f4')

    def __init__(self, manager_filename, max_bytes_to_buffer):
        """TODO"""
        self.manager_filename = manager_filename
        self.manager_directory = os.path.dirname(manager_filename)
        pygeoprocessing.create_directories([self.manager_directory])
        db_connection = sqlite3.connect(
            manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute('''CREATE TABLE IF NOT EXISTS array_table
            (array_id INTEGER PRIMARY KEY, array_path TEXT)''')

        db_connection.commit()
        db_connection.close()

        self.array_cache = collections.defaultdict(collections.deque)
        self.max_bytes_to_buffer = max_bytes_to_buffer
        self.current_bytes_in_system = 0

    def append(self, array_id, array_data):
        """Appends data to the file, this may be the buffer in memory or a
            file on disk"""

        self.array_cache[array_id].append(array_data.copy())
        self.current_bytes_in_system += (
            array_data.size * BufferedFileManager._ARRAY_TUPLE_TYPE.itemsize)
        if self.current_bytes_in_system > self.max_bytes_to_buffer:
            self.flush()

    def flush(self):
        """Method to flush the manager.  If the file exists we append to it,
            otherwise we write directly."""

        start_time = time.time()
        LOGGER.info(
            'Flushing %d bytes in %d arrays', self.current_bytes_in_system,
            len(self.array_cache))

        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()

        #get all the array data to append at once
        insert_list = []

        while len(self.array_cache) > 0:
            array_id = self.array_cache.iterkeys().next()
            array_deque = self.array_cache.pop(array_id)
            #Try to get data if it's there
            db_cursor.execute(
                """SELECT (array_path) FROM array_table
                    where array_id=? LIMIT 1""", [array_id])
            array_path = db_cursor.fetchone()
            if array_path is not None:
                # cache gets wiped at end so okay to use same deque
                array_deque.append(numpy.load(array_path[0]))
                array_data = numpy.concatenate(array_deque)
                array_deque = None
                numpy.save(array_path[0], array_data)
            else:
                #otherwise directly write
                #make a random filename and put it one directory deep named
                #off the last two characters in the filename
                array_filename = uuid.uuid4().hex + '.npy'
                #-6:-4 skips the extension and gets the last 2 characters
                array_directory = os.path.join(
                    self.manager_directory, array_filename[-6:-4])
                if not os.path.isdir(array_directory):
                    os.mkdir(array_directory)
                array_path = os.path.join(array_directory, array_filename)
                #save the file
                array_data = numpy.concatenate(array_deque)
                array_deque = None
                numpy.save(array_path, array_data)
                insert_list.append((array_id, array_path))
        db_cursor.executemany(
            '''INSERT INTO array_table
                (array_id, array_path)
            VALUES (?,?)''', insert_list)

        db_connection.commit()
        db_connection.close()

        self.current_bytes_in_system = 0
        LOGGER.info('Completed flush in %.2fs', time.time() - start_time)

    def read(self, array_id):
        """Read the entirety of the file.  Internally this might mean that
            part of the file is read from disk and the end from the buffer
            or any combination of those."""

        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT (array_path) FROM array_table where array_id=? LIMIT 1",
            [array_id])
        array_path = db_cursor.fetchone()
        if array_path is not None:
            array_data = numpy.load(array_path[0])
        else:
            array_data = numpy.empty(
                0, dtype=BufferedFileManager._ARRAY_TUPLE_TYPE)

        if len(self.array_cache[array_id]) > 0:
            local_deque = collections.deque(self.array_cache[array_id])
            local_deque.append(array_data)
            array_data = numpy.concatenate(local_deque)

        return array_data

    def delete(self, array_id):
        """Deletes the file on disk if it exists and also purges from the
            cache"""
        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT (array_path) FROM array_table where array_id=? LIMIT 1",
            [array_id])
        array_path = db_cursor.fetchone()
        if array_path is not None:
            os.remove(array_path[0])
            try:
                # attempt to remove the directory if it's empty
                os.rmdir(os.path.dirname(array_path[0]))
            except OSError:
                # it's not empty, not a big deal
                pass

        #delete the key from the table
        db_cursor.execute(
            "DELETE FROM array_table where array_id=?", [array_id])
        db_connection.close()

        #delete the cache and update cache size
        #The * 12 comes from the fact that the array is an 'a4 f4 f4'
        self.current_bytes_in_system -= (
            sum([x.size for x in self.array_cache[array_id]]) *
            BufferedFileManager._ARRAY_TUPLE_TYPE.itemsize)
        del self.array_cache[array_id]
