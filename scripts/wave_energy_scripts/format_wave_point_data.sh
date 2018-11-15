#!/bin/bash

# This script properly formats the wave energy points text file into a proper
# CSV file. The input file should have at least the following columns:
# ID - a unique integer for each point 
# I, J - integer values that match the locations from the wave watch three file
# LONG - the Longitude of the point
# LATI - the Latitude of the point 
# HSAVG - a floating point value for the average wave height at that location
# TPAVG - a floating point value for the average wave period at that location
#
# Example file:
#C** GRID_NSB4M
#        ID    I    J   LONG        LATI        HSAVG     TPAVG    NWATER      DCOVER(%)
#         1  102  370   111.733368  -25.399878   3.28     13.10     29068          99.88
#         2  102  371   111.733368  -25.399878   3.28     13.10     29068          99.88
#         3  102  372   111.733368  -25.399878   3.28     13.10     29068          99.88
#         4  102  373   111.733368  -25.399878   3.28     13.10     29068          99.88
#

# 'sed' is a find and replace command. First we delete the first row which
# should have the name of the data. Then we delete all the whitespace and
# replace with a comma for separation. Then we remove the leading comma which
# was introduced in the previous sed operation. Then we remove the last two
# columns
sed '1d' | sed -r -e 's/\s+/,/g' | sed -r -e 's/,//' | sed 's/,[^,]*//7g'
