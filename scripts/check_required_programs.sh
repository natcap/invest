#!/bin/sh
# Check that all programs provided at the command line are available on the
# PATH.
#
# Parameters:
#     * - program names to look for on the PATH.
#
# Example invokation:
#     ./check_required_programs.sh program1 program2 program3

for program in $@
do
    which $program > /dev/null \
        && echo "\e[32mOK\e[0m: $program  \t$(which $program)" \
        || echo "\e[31mMISSING\e[0m: $program"
done
