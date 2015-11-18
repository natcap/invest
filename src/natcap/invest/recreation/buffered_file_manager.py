"""Buffered file manager module"""

import collections
import os
import numpy
import io
import sqlite3


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


class BufferedFileManager(object):
    """A file manager that buffers many reads and writes in hopes that
        expensive file operations are mitigated."""

    def __init__(self, manager_filename, max_bytes_to_buffer):
        """make a new BufferedFileManager object"""

        self.manager_filename = manager_filename
        if not os.path.exists(os.path.dirname(manager_filename)):
            os.mkdir(os.path.dirname(manager_filename))
        db_connection = sqlite3.connect(
            manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute('''CREATE TABLE IF NOT EXISTS array_table
            (array_id text PRIMARY KEY, array_data array)''')
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

        print 'Flushing %d bytes in %d arrays' % (
            self.current_bytes_in_system, len(self.array_cache))

        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()

        for array_id, array_deque in self.array_cache.iteritems():
            #Try to get data if it's there
            db_cursor.execute(
                "SELECT (array_data) FROM array_table where array_id=?", [array_id])
            array_data = db_cursor.fetchone()
            if array_data is not None:
                #append if so
                array_data = numpy.concatenate(
                    (array_data, numpy.concatenate(array_deque)))
            else:
                #otherwise directly write
                array_data = numpy.concatenate(array_deque)

            #query handles both cases of an existing ID or a non-existant one
            db_cursor.execute(
                '''INSERT OR REPLACE INTO array_table
                    (array_id, array_data)
                VALUES (?,?)''', [array_id, array_data])

        db_connection.commit()
        db_connection.close()

        self.array_cache.clear()
        self.current_bytes_in_system = 0

    def read(self, array_id):
        """Read the entirety of the file.  Internally this might mean that
            part of the file is read from disk and the end from the buffer
            or any combination of those."""

        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT (array_data) FROM array_table where array_id=?", [array_id])
        query_result = db_cursor.fetchone()
        db_connection.close()
        if query_result is not None:
            #append if so
            array_data = query_result
        else:
            array_data = numpy.empty(0, dtype='a4, f4, f4')

        array_data = numpy.concatenate(
            (array_data, numpy.concatenate(self.array_cache[array_id])))
        return array_data

    def delete(self, array_id):
        """Deletes the file on disk if it exists and also purges from the
            cache"""
        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "DELETE FROM array_table where array_id=?", [array_id])
        db_connection.close()
        #The * 12 comes from the fact that the array is an 'a4 f4 f4'
        self.current_bytes_in_system -= sum(
            [x.size for x in self.array_cache[array_id]]) * 12
        del self.array_cache[array_id]
