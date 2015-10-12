"""Buffered file manager module"""

import collections
import os

import sqlite3

class BufferedFileManager(object):
    """A file manager that buffers many reads and writes in hopes that
        expensive file operations are mitigated."""

    def __init__(self, manager_filename, max_bytes_to_buffer):
        """make a new BufferedFileManager object"""

        self.manager_filename = manager_filename
        if not os.path.exists(os.path.dirname(manager_filename)):
            os.mkdir(os.path.dirname(manager_filename))
        db_connection = sqlite3.connect(manager_filename)
        db_cursor = db_connection.cursor()
        db_cursor.execute('''CREATE TABLE IF NOT EXISTS blob_table
            (blob_id text PRIMARY KEY, blob_data blob)''')
        db_connection.commit()
        db_connection.close()

        self.blob_cache = collections.defaultdict(collections.deque)
        self.max_bytes_to_buffer = max_bytes_to_buffer
        self.current_bytes_in_system = 0


    def append(self, blob_id, data):
        """Appends data to the file, this may be the buffer in memory or a
            file on disk"""

        self.blob_cache[blob_id].append(data)
        self.current_bytes_in_system += len(data)
        if self.current_bytes_in_system > self.max_bytes_to_buffer:
            self.flush()

    def flush(self):
        """Method to flush the manager.  If the file exists we append to it,
            otherwise we write directly."""

        print 'Flushing %d bytes in %d blobs' % (
            self.current_bytes_in_system, len(self.blob_cache))

        db_connection = sqlite3.connect(self.manager_filename)
        db_cursor = db_connection.cursor()

        for blob_id, blob_deque in self.blob_cache.iteritems():
            #Try to get data if it's there
            db_cursor.execute(
                "SELECT (blob_data) FROM blob_table where blob_id=?", [blob_id])
            blob_data = db_cursor.fetchone()
            if blob_data is not None:
                #append if so
                blob_data = blob_data[0] + ''.join(blob_deque)
            else:
                #otherwise directly write
                blob_data = ''.join(blob_deque)

            #query handles both cases of an existing ID or a non-existant one
            db_cursor.execute(
                '''INSERT OR REPLACE INTO blob_table
                    (blob_id, blob_data)
                VALUES (?,?)''', [blob_id, sqlite3.Binary(blob_data)])

        db_connection.commit()
        db_connection.close()

        self.blob_cache.clear()
        self.current_bytes_in_system = 0


    def read(self, blob_id):
        """Read the entirety of the file.  Internally this might mean that
            part of the file is read from disk and the end from the buffer
            or any combination of those."""

        db_connection = sqlite3.connect(self.manager_filename)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "SELECT (blob_data) FROM blob_table where blob_id=?", [blob_id])
        query_result = db_cursor.fetchone()
        db_connection.close()
        if query_result is not None:
            #append if so
            blob_data = query_result[0]
        else:
            blob_data = ''

        blob_data += ''.join(self.blob_cache[blob_id])
        return blob_data


    def delete(self, blob_id):
        """Deletes the file on disk if it exists and also purges from the
            cache"""
        db_connection = sqlite3.connect(self.manager_filename)
        db_cursor = db_connection.cursor()
        db_cursor.execute(
            "DELETE FROM blob_table where blob_id=?", [blob_id])
        db_connection.close()
        self.current_bytes_in_system -= len(
            ''.join(self.blob_cache[blob_id]))
        del self.blob_cache[blob_id]
