"""recmodel client demo"""

import sys
import os
import glob
import zipfile
import time

import Pyro4

#this serializer lets us pass null bytes in strings unlike the default
Pyro4.config.SERIALIZER = 'marshal'

def main():
    """entry point"""

    if len(sys.argv) != 4:
        print 'usage: %s aoi_filename hostname port' % sys.argv[0]
        sys.exit(-1)

    hostname = sys.argv[2]
    port = int(sys.argv[3])

    workspace_uri = 'rec_client_workspace'
    tmp_dir = os.path.join(workspace_uri, 'tmp')
    output_dir = os.path.join(workspace_uri, 'output')

    for directory in [tmp_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    recmodel_server = Pyro4.Proxy(
        "PYRO:natcap.invest.recreation@%s:%d" % (hostname, port))
    aoi_shapefile_filename = sys.argv[1]

    #gather shapefile components and zip them
    basename = os.path.splitext(aoi_shapefile_filename)[0]
    aoi_archive_uri = os.path.join(tmp_dir, 'to_server.zip')
    with zipfile.ZipFile(aoi_archive_uri, 'w') as myzip:
        for filename in glob.glob(basename + '.*'):
            print 'archiving ', filename
            myzip.write(filename, os.path.basename(filename))

    #convert shapefile to binary string for serialization
    zip_file_binary = open(aoi_archive_uri, 'rb').read()

    #transfer zipped file to server
    try:
        start_time = time.time()
        print 'server version is ', recmodel_server.get_version()
        result_zip_file_binary = (
            recmodel_server.calc_user_days_binary(zip_file_binary))
        print 'recieved result, took %f seconds' % (time.time() - start_time)

        #unpack result
        result_zip_uri = os.path.join(output_dir, 'pud_result.zip')
        open(result_zip_uri, 'wb').write(result_zip_file_binary)
        zipfile.ZipFile(result_zip_uri, 'r').extractall(output_dir)
        os.remove(result_zip_uri)

    except Exception:
        #This catches and prints Pyro tracebacks.
        print "Pyro traceback:"
        print "".join(Pyro4.util.getPyroTraceback())

if __name__ == "__main__":
    main()
