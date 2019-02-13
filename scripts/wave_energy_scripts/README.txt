OVERVIEW

This README file outlines and provides instructions for formatting data for the
wave energy model. There are scripts provided in the wave energy module that
can be used to properly format wave watch 3 data as well as wave energy point
data.

There are two formatting tracks that can be followed depending on the stage of
the data. These two tracks are outlined below as well as a short description of
each script is provided and the final data files that the wave energy model
expects.

==============================================================================
Wave Energy Model Data Needs

    The wave energy model needs three files for each location of interest.
    When selecting a location in the dropdown list from the user interface, the
    model will expect that the required input data live in input/WaveData/ .
    This relationship of dropdown selection to needed data is hardcoded in the
    model. In order for a new location of interest to be added and to properly
    link the new data to the wave energy model a request must be made to the
    developer. The data that is needed is as follows:
    1) ww3_binary_data.bin
        A binary file that has the wave watch three data

    2) wave_point_data.shp
        A shapefile of the wave point data with wave periods and wave height
        values

    3) wave_point_bounding_area.shp
        A polygon shapefile that is a bounding box for the wave_point_data.shp.
        This is used to properly clip outputs. This polygon should cover all
        the points in wave_point_data.shp

===============================================================================
Wave Energy Data Formatting Scripts

1) format_wave_watch_data.sh

    This script does a fast, in place, search and replace to properly format a
    wave watch three text file. This script is useful and should be run if the
    wave watch three text file has the wave period and wave height names
    included in the file. It also should be run if the lines end with commas.
    See the script for more detailed documentation and an example of an input
    wave watch three file.

    Example:
    >> ./format_wave_watch_data.sh < ww3_text_data.txt

    The '<' indicates that a file should be passed into the script as an input.
    In this case we want to pass in the wave watch three text file as the input

2) format_wave_point_data.sh
    
    This script does a fast, in place, search and replace to properly format a
    wave energy point data text file. This script expects a very specific input
    file. It will delete the first line of the file, remove whitespaces and
    replace with commas, and add new line characters. This script creates a csv
    formatted file from the given text file. See the script for more details
    and documentation.
    
    Example:
    >> ./format_wave_point_data.sh < wave_point_data.txt

    The '<' indicates that a file should be passed into the script as an input.
    In this case we want to pass in the wave point data file as the input
    
3) ww3_per_year.py
    
    This python script converts formatted wave watch three data values into
    yearly values. Usually wave watch three data is collected over x number of
    years and the wave energy model expects the data to be yearly. Thus this
    script divides the values by the number of years provided in the arguments.
    See the python script for more detailed documentation and examples.

    Example:
    >> python ww3_per_year.py ww3_formatted_data.txt ww3_yearly.txt 10

4) wave_watch_data_to_binary.py

    This python script converts yearly formatted wave watch three text data
    into a binary formatted file. This compression technique allows us to save
    a lot of space and overhead on very large wave watch three data sets. The
    wave energy model expects the wave watch three data to be packed in a
    specific manner so that it can properly unpack it. See the python script
    for more detailed documentation.

    Example:
    >> python wave_watch_data_to_binary.py ww3_yearly.txt ww3_out_binary.bin

5) wave_csv_to_points.py

    This python script converts a formatted wave energy point data csv file
    into an OGR point shapefile. See the python script for more detailed
    documentation.

    Example:
    >> python wave_csv_to_points.py wave_data_formatted.txt my_layer out_shape.shp

BASH Script Notes:

    A bash script can be run natively in Linux or Mac OS from the
    command line. A bash script can also be run on Windows using Cygwin
    (cygwin.com). Example of calling a bash script:
    >> ./my_script.sh
    
===============================================================================
Wave Energy Data Formatting Tracks

Track 1:

    Completely unformatted data for both wave data points and wave watch three.
    This indicates that the bash scripts should be run.
    
    Steps for formatting wave watch three data:
    a) Get the wave watch three data in a text file with the following format:
        TPbin=.37500E+00,.10000E+00,...,.20000E+00,
        HSbin=.37500E+00,.10000E+00,...,.20000E+00,
        I,102,J,370
        .0000E+00,.18000E+02,...,.36000E+02,
        .0000E+00,.18000E+02,...,.36000E+02,
        ...
        I,102,J,371
        .0000E+00,.18000E+02,...,.36000E+02,
        .0000E+00,.18000E+02,...,.36000E+02,
        ...
        
        The first line should be the range for wave periods followed by the
        second line which should be the range for wave heights. The following
        lines should be an I,J value line which indicates position that matches
        the wave point data, followed by the data values. For the data values,
        each line represents a row with each value being a row-col value.

    b) Run 'format_wave_watch_data.sh'
        >> ./format_wave_watch_data.sh < wave_watch_data.txt

    c) Run 'ww3_per_year.py'
        >> python ww3_per_year.py ww3_formatted_data.txt ww3_yearly.txt 10

    d) Run 'wave_watch_data_to_binary.py'
        >> python wave_watch_data_to_binary.py ww3_yearly.txt ww3_out_binary.bin

    Steps for formatting wave point data:
    a) Get the wave point data in a text file with the following format:
        C** GRID.ABD (This is a throw away first line)
        ID  I   J   LONG    LATI    HSAVG   TPAVG
        1   102 370 25.34   54.32   3.28    13.10
        2   102 370 25.34   54.32   3.28    13.10
        3   102 370 25.34   54.32   3.28    13.10
    
    b) Run 'format_wave_point_data.sh'
        >> ./format_wave_point_data.sh < wave_point_data.txt

    c) Run 'wave_csv_to_points.py'
        >> python wave_csv_to_points.py wave_data_formatted.txt my_layer out_shape.shp

Track 2:

    It is possible to format the wave watch three and wave point data files by
    hand such that running the bash scripts is unnecessary. In this case only
    the following scripts need to be run.

    Steps for wave watch three data:
    a) Run 'ww3_per_year.py'
        >> python ww3_per_year.py ww3_formatted_data.txt ww3_yearly.txt 10

    b) Run 'wave_watch_data_to_binary.py'
        >> python wave_watch_data_to_binary.py ww3_yearly.txt ww3_out_binary.bin

    Steps for wave point data:
    a) Run 'format_wave_point_data.sh'
        >> ./format_wave_point_data.sh < wave_point_data.txt

    b) Run 'wave_csv_to_points.py'
        >> python wave_csv_to_points.py wave_data_formatted.txt my_layer out_shape.shp

