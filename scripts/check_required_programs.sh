#!/bin/sh
# Check that all programs provided at the command line are available on the
# PATH.
#
# Parameters:
#     * - program names to look for on the PATH.
#
# Example invokation:
#     ./check_required_programs.sh program1 program2 program3

EXITCODE=0
for program in "$@"
do
    if which "$program" > /dev/null
    then
        echo "OK: $program  $(which $program)"
    else
        echo "MISSING: $program"
        EXITCODE=1
    fi
done
exit $EXITCODE
