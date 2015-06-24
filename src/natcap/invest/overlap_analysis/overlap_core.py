'''
Core module for both overlap analysis and management zones. This function
can be used by either of the secondary modules within the OA model.
'''
import os
import fnmatch

from osgeo import ogr


def get_files_dict(folder):
    '''Returns a dictionary of all .shp files in the folder.

        Input:
            folder- The location of all layer files. Among these, there should
                be files with the extension .shp. These will be used for all
                activity calculations.

        Returns:
            file_dict- A dictionary which maps the name (minus file extension)
                of a shapefile to the open datasource itself. The key in this
                dictionary is the name of the file (not including file path or
                extension), and the value is the open shapefile.
    '''

    #Glob.glob gets all of the files that fall into the form .shp, and makes 
    #them into a list. Then, each item in the list is added to a dictionary as
    #an open file with the key of it's filename without the extension, and that
    #whole dictionary is made an argument of the mz_args dictionary
    dir_list = listdir(folder)
    file_names = []
    file_names += fnmatch.filter(dir_list, '*.shp')
    file_dict = {}

    for file in file_names:
        #The return of os.path.split is a tuple where everything after the final
        #slash is returned as the 'tail' in the second element of the tuple
        #path.splitext returns a tuple such that the first element is what comes
        #before the file extension, and the second is the extension itself 
        name = os.path.splitext(os.path.split(file)[1])[0]

        file_dict[name] = ogr.Open(file)

    return file_dict


def listdir(path):
    '''A replacement for the standar os.listdir which, instead of returning
    only the filename, will include the entire path. This will use os as a
    base, then just lambda transform the whole list.

    Input:
        path- The location container from which we want to gather all files.

    Returns:
        A list of full URIs contained within 'path'.
    '''
    file_names = os.listdir(path)
    uris = map(lambda x: os.path.join(path, x), file_names)

    return uris
