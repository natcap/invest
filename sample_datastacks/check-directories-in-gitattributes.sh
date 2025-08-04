#! /bin/bash
#
# Script to verify that all data directories are represented in the
# .gitattributes file for LFS filtering.


GITATTRIBUTES_COMPLETE=1
for dir in `ls .`
do
        if [ -d $dir ]
        then
                # Check that the directory is in gitattributes
                grep "^$dir/\*\* filter=lfs diff=lfs merge=lfs -text" .gitattributes > /dev/null
                if [ $? -ne 0 ]
                then
                        echo "$dir missing from .gitattributes or LFS config is malformed"
                        GITATTRIBUTES_COMPLETE=0
                fi
        fi
done

# If git attributes are missing a directory, fail the script.
if [ $GITATTRIBUTES_COMPLETE -eq 0 ]
then
        exit 1
fi
