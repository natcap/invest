"""This Python Script converts and compresses yearly wave watch three text data
    to a binary format that's faster to load and cheaper to build in an
    installer than the bulky text format

    Example from command line:
    >> python wave_watch_data_to_binary.py ww3_yearly.txt ww3_binary.bin

"""

import sys
import struct

import numpy as np


def text_wave_data_to_binary(wave_watch_file_uri, binary_file_uri):
    """Convert and compress the wave watch three data into binary format,
        packing in a specific manner such that the InVEST3.0 wave energy
        model can properly unpack it

        wave_watch_file_uri - a URI to the formatted yearly wave watch three
            data (required)

        binary_file_uri - a URI to write out the binary file (.bin) (required)

        returns - Nothing"""

    # Open the wave watch three files
    wave_file = open(wave_watch_file_uri,'rU')
    # Open the binary output file as writeable
    bin_file = open(binary_file_uri, 'wb')

    # Initiate an empty list for packing up the wave periods
    wave_periods = []
    # Initiate an empty list for packing up the wave heights
    wave_heights = []

    # Get the periods and heights, removing the newline characters and splitting
    # on commas
    wave_periods = map(float,wave_file.readline().rstrip('\n').split(','))
    wave_heights = map(float,wave_file.readline().rstrip('\n').split(','))

    # Pack up the number of wave period and wave height entries into two
    # integers. This is used to properly unpack
    s=struct.pack('ii',len(wave_periods),len(wave_heights))
    bin_file.write(s)

    # Pack up the wave period values as float types
    s=struct.pack('f'*len(wave_periods), *wave_periods)
    bin_file.write(s)

    # Pack up the wave height values as float types
    s=struct.pack('f'*len(wave_heights), *wave_heights)
    bin_file.write(s)

    # For the rest of the file
    while True:
        # Get the next line
        line = wave_file.readline()

        # Check for the end of the file
        if len(line) == 0:
            #end of file
            break

        # If it is the start of a new location, get (I,J) values
        if line[0] == 'I':
            # Split I, n, J, m into just the numeric part
            i,j = map(int,line.split(',')[1:4:2])
            # Pack up I,J values as integers
            s=struct.pack('ii',i,j)
            bin_file.write(s)
        else:
            # Get the data values as floats
            float_list = map(float,line.split(','))
            # Pack up the data values as float types
            s=struct.pack('f'*len(float_list), *float_list)
            bin_file.write(s)


if __name__ == '__main__':
    # Get the wave watch three uri from the first command line argument
    wave_watch_file_uri = sys.argv[1]

    # Get the out binary uri from the second command line argument
    binary_file_uri = sys.argv[2]

    # Call the function to properly convert and compress data
    text_wave_data_to_binary(wave_watch_file_uri, binary_file_uri)
