"""Buffered file manager module"""

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
        db_cursor.execute('PRAGMA page_size=4096')
        db_cursor.execute('PRAGMA cache_size=10000')
        db_cursor.execute('PRAGMA locking_mode=EXCLUSIVE')
        db_cursor.execute('PRAGMA synchronous=NORMAL')
        db_cursor.execute('PRAGMA auto_vacuum=NONE')
        db_cursor.execute('''CREATE TABLE IF NOT EXISTS array_table
            (array_id INTEGER PRIMARY KEY, array_data array)''')

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
        db_cursor.execute("PRAGMA synchronous = OFF")
        db_cursor.execute("PRAGMA journal_mode = OFF")
        db_cursor.execute("PRAGMA cache_size = 131072")

        #get all the array data to append at once
        insert_list = []

        for array_id, array_deque in self.array_cache.iteritems():
            #Try to get data if it's there
            db_cursor.execute(
                """SELECT (array_data) FROM array_table
                    where array_id=? LIMIT 1""", [array_id])
            array_data = db_cursor.fetchone()
            if array_data is not None:
                #append if so
                array_data = numpy.concatenate(
                    (array_data[0], numpy.concatenate(array_deque)))
            else:
                #otherwise directly write
                array_data = numpy.concatenate(array_deque)
            insert_list.append((array_id, array_data))
            if len(insert_list) > 1000:
                try:
                    db_cursor.executemany(
                        '''INSERT OR REPLACE INTO array_table
                            (array_id, array_data)
                        VALUES (?,?)''', insert_list)
                    insert_list = []
                except sqlite3.InterfaceError:
                    try:
                        last_array_data = None
                        last_array_id = None
                        for array_id, array_data in insert_list:
                            db_cursor.execute(
                                '''INSERT OR REPLACE INTO array_table
                                    (array_id, array_data)
                                VALUES (?,?)''', (array_id, array_data))
                            last_array_id = array_id
                            last_array_data = array_data
                    except sqlite3.InterfaceError:
                        LOGGER.debug((array_id, array_data, array_data.shape))
                        LOGGER.debug((last_array_id, last_array_data, last_array_data.shape))
                    #LOGGER.debug(insert_list)
                        raise

        if len(insert_list) > 100:
            try:
                db_cursor.executemany(
                    '''INSERT OR REPLACE INTO array_table
                        (array_id, array_data)
                    VALUES (?,?)''', insert_list)
                insert_list = []
            except sqlite3.InterfaceError:
                for x in insert_list:
                    LOGGER.debug(x.dtype)
                raise

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
        db_cursor.execute("PRAGMA cache_size = 131072")
        db_cursor.execute(
            "SELECT (array_data) FROM array_table where array_id=? LIMIT 1",
            [array_id])
        query_result = db_cursor.fetchone()
        if query_result is not None:
            #append if so
            array_data = query_result[0]
        else:
            array_data = numpy.empty(0, dtype='a4, f4, f4')

        if len(self.array_cache[array_id]) > 0:
            try:
                array_data = numpy.concatenate(
                    (array_data, numpy.concatenate(self.array_cache[array_id])))
            except ValueError:
                LOGGER.debug(array_data)
                for x in self.array_cache[array_id]:
                    LOGGER.debug(x)
                LOGGER.debug(array_id)
                raise

        return array_data

    def delete(self, array_id):
        """Deletes the file on disk if it exists and also purges from the
            cache"""
        db_connection = sqlite3.connect(
            self.manager_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        db_cursor = db_connection.cursor()
        db_cursor.execute("PRAGMA cache_size = 131072")
        db_cursor.execute(
            "DELETE FROM array_table where array_id=?", [array_id])
        db_connection.close()
        #The * 12 comes from the fact that the array is an 'a4 f4 f4'
        self.current_bytes_in_system -= sum(
            [x.size for x in self.array_cache[array_id]]) * 12
        del self.array_cache[array_id]
