!#/bin/bash
# REQUIREMENTS:
# -------------
# * invest.arcgis repo has been cloned ($ paver fetch src)
# * html and pdf documentation has been created ($ paver build_docs --skip-api)

DIR=../../test_dir
if [ ! -d $DIR ]
then
    mkdir $DIR
    touch $DIR/foo.txt
fi

# This file has to exist ... just providing a dummy file for the test build of the installer.
touch vcredist_x86.exe

wine $HOME/.wine/drive_c/Program\ Files/NSIS/makensis \
    /DVERSION=$1 \
    /DVERSION_DISK=$1 \
    /DINVEST_3_FOLDER=$(basename $DIR)\
    /DSHORT_VERSION=$1 \
    /DARCHITECTURE=x86 \
    ./invest_installer.nsi
