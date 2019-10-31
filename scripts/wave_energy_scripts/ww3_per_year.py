"""This Python Script converts wave watch 3 data to yearly values

    Example from command line:
    >> python ww3_per_year.py ww3_formatted.txt ww3_yearly.txt 10
"""

import os
import sys

import numpy as np


def extrapolate_wave_data(wave_watch_file_uri, out_file_uri,  num_years):
    """Divide all the wave watch data values by 'num_years' to get per year data

        wave_watch_file_uri - a URI to a formatted wave watch three text file
            (required)

        out_file_uri - a URI to write the updated yearly data to (required)

        num_years - an Integer for the number of years the wave watch data was
            collected over (required)

        returns - Nothing"""

    # Get the number of years as a float just to make sure we do floating
    # division
    num_years = float(num_years)

    # Open the wave watch text file
    wave_file = open(wave_watch_file_uri)

    # Get the periods and heights in that order
    wave_periods = wave_file.readline()
    wave_heights = wave_file.readline()

    # Open the output uri as writeable
    out_file = open(out_file_uri, 'w')
    # Write the wave periods and heights to the output file as we DO NOT want to
    # divide the ranges by the number of years
    out_file.write(wave_periods)
    out_file.write(wave_heights)

    # For each remaining line
    while True:
        # Get the next line
        line = wave_file.readline()
        # Set a blank string to build up altered yearly data line
        out_string = ''
        # Check for the end of the file
        if len(line) == 0:
            #end of file
            break
        # If it is the start of a new location (I,J) write it as is to the
        # output file
        if line[0] == 'I':
            out_file.write(line)
        else:
            # Get the data line (row) as a list splitting on commas
            val_array = line.split(',')
            # Convert the list to type float from string
            values = np.array(val_array, dtype='f')
            # Divide all the values in the list by the number of years to get
            # per year values
            per_yr_vals = values / num_years

            # Now that we have a list of floating values we need to build up a
            # string to write back to the output file. Start by iterating over
            # each per year value in the list
            for val in per_yr_vals:
                # Set the value as a string in scientific notation
                val = "%E"  %val
                # Build up the output line string that will be written to the
                # output file
                out_string = out_string + str(val) + ','

            # Once the string is completed for the line, remove the trailing
            # comma at the end of the line and replace with a newline character
            out_string = out_string[0:len(out_string)-1] + '\n'

            # Write the line string with the yearly data values to the output
            # file
            out_file.write(out_string)

    # Close the files
    wave_file.close()
    out_file.close()

# When called from the command line this statement will call our above function
# with the arguments from the command line
if __name__ == "__main__":
    extrapolate_wave_data(sys.argv[1],sys.argv[2],sys.argv[3])
