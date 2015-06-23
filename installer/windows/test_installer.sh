!#/bin/bash
#wine /home/jadoug06/.wine/drive_c/Program\ Files/NSIS/makensis /DVERSION=$1 /DVERSION_DISK=$1 /DINVEST_3_FOLDER=invest-3-x86 /DINVEST_3_FOLDER_x64=invest-3-x64 /DSHORT_VERSION=$1 /DARCHITECTURE=x86 ./invest_installer.nsi

wine /home/jadoug06/.wine/drive_c/Program\ Files/NSIS/makensis /DVERSION=$1 /DVERSION_DISK=$1 /DINVEST_3_FOLDER=test_dir /DSHORT_VERSION=$1 /DARCHITECTURE=x86 ./invest_installer.nsi
