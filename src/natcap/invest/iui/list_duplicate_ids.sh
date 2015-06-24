# list_duplicate_ids.sh
# 
# This script takes an arbitrary number of arguments when run from the command
# line.  It prints a list of duplicate ID strings from an IUI json object.  If
# none are found, it prints nothing.

#!/bin/sh
grep -ih \"id\": $@ | grep -io \"[a-z_]*\"\,\\?$ | uniq -d
