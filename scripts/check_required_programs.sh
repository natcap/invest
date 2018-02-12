#!/bin/sh

for program in $@
do
    which $program > /dev/null \
        && echo "\e[32mOK\e[0m: $program" \
        || echo "\e[31mMISSING\e[0m: $program"
done
