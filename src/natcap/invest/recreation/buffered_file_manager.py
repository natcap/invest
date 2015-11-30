"""Buffered file manager module"""

import uuid
import time
import collections
import os
import numpy
import io
import sqlite3
import logging

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s \
%(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %H:%M:%S ')

LOGGER = logging.getLogger('natcap.invest.recmodel_server.buffered_file_manager')


def _adapt_array(array):
    """Convert numpy array to sqlite3 binary type.
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    numpy.save(out, array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    """Adapter to convert an SQL blob to a numpy array"""
    out = io.BytesIO(text)
    out.seek(0)
    return numpy.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(numpy.ndarray, _adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", _convert_array)


def _deque_to_array(numpy_deque):
    """concatenate a deque of 'a4,f4,f4' numpy arrays"""
    n_elements = sum([x.size for x in numpy_deque])
    result = numpy.empty(n_elements, dtype='a4, f4, f4')
    valid_pos = 0
    while len(numpy_deque) > 0:
        array = numpy_deque.pop()
        result[valid_pos:valid_pos+array.size] = array
        valid_pos += array.size
    return result


class BufferedFileManager(object):
    """A file manager that buffers many reads and writes in hopes that
        expensive file operations are mitigated."""

    def __init__(self, manager_filename, max_bytes_to_buffer):
        """make a new BufferedFileManager object"""

        self.manager_filename = manager_filename
        self.manager_directory = os.path.dirname(manager_filename)
        if not os.path.exists(self.manager_directory):
            os.mkdir(self.manager_directory)
        db_connection = sqlite3.connect(
            manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute('PRAGMA page_size=4096')
        db_cursor.execute('PRAGMA cache_size=10000')
        db_cursor.execute('PRAGMA locking_mode=EXCLUSIVE')
        db_cursor.execute('PRAGMA synchronous=NORMAL')
        db_cursor.execute('PRAGMA auto_vacuum=NONE')
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

        self.array_cache[array_id].append(array_data)
        self.current_bytes_in_system += array_data.size * 12  # a4 f4 f4
        if self.current_bytes_in_system > self.max_bytes_to_buffer:
            self.flush()

    def flush(self):
        """Method to flush the manager.  If the file exists we append to it,
            otherwise we write directly."""

        start_time = time.time()
        LOGGER.debug('Flushing %d bytes in %d arrays' % (
            self.current_bytes_in_system, len(self.array_cache)))

        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute("PRAGMA synchronous = OFF")
        db_cursor.execute("PRAGMA journal_mode = OFF")
        db_cursor.execute("PRAGMA cache_size = 131072")

        #get all the array data to append at once
        insert_list = []

        for array_id, array_deque in self.array_cache.iteritems():
            #Try to get data if it's there
            db_cursor.execute(
                """SELECT (array_path) FROM array_table
                    where array_id=? LIMIT 1""", [array_id])
            array_path = db_cursor.fetchone()
            if array_path is not None:
                #append if so
                array_deque.append(numpy.load(array_path[0]))
                array_data = numpy.concatenate(array_deque)
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
                numpy.save(array_path, array_data)
                insert_list.append((array_id, array_path))
        db_cursor.executemany(
            '''INSERT INTO array_table
                (array_id, array_path)
            VALUES (?,?)''', insert_list)

        db_connection.commit()
        db_connection.close()

        self.array_cache.clear()
        self.current_bytes_in_system = 0
        LOGGER.debug('Completed flush in %.2fs' % (time.time() - start_time))

    def read(self, array_id):
        """Read the entirety of the file.  Internally this might mean that
            part of the file is read from disk and the end from the buffer
            or any combination of those."""

        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute("PRAGMA cache_size = 131072")
        db_cursor.execute(
            "SELECT (array_path) FROM array_table where array_id=? LIMIT 1",
            [array_id])
        array_path = db_cursor.fetchone()
        if array_path is not None:
            array_data = numpy.load(array_path[0])
        else:
            array_data = numpy.empty(0, dtype='a4, f4, f4')

        if len(self.array_cache[array_id]) > 0:
            local_deque = collections.deque(self.array_cache[array_id])
            local_deque.append(array_data)
            array_data = numpy.concatenate(self.array_cache[array_id])

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
        self.current_bytes_in_system -= sum(
            [x.size for x in self.array_cache[array_id]]) * 12
        del self.array_cache[array_id]
