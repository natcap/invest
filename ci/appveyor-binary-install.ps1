# This is an install script needed for our binary build on AppVeyor.
#
# It assumes that the applications choco, wget, 7zip, unzip and curl are available
# and on the PATH.  It also requires that the environment variable
#

choco install make wget vcredist140 pandoc

# Our Makefile depends on zip rather than 7zip, so we need to get those binaries.
choco install zip

# The binary build requires the shapely DLL to be named something specific.
# /B copies the file as a binary file.
copy /B $env:PYTHON\Lib\site-packages\shapely\DLLs\geos_c.dll $env:PYTHON\Lib\site-packages\shapely\DLLs\geos.dll

# Download and install NSIS plugins to their correct places.
wget -nv https://storage.googleapis.com/natcap-build-dependencies/windows/Inetc.zip
wget -nv https://storage.googleapis.com/natcap-build-dependencies/windows/Nsisunz.zip
wget -nv https://storage.googleapis.com/natcap-build-dependencies/windows/NsProcess.zip
7z e NsProcess.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugin\nsProcess.dll
7z e NsProcess.zip -o"C:\Program Files (x86)\NSIS\Include" Include\nsProcess.nsh
7z e Inetc.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugins\x86-ansi\INetC.dll
7z e Nsisunz.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" nsisunz\Release\nsisunz.dll

# Download, quietly unzip and install the google cloud utilities.
curl -o gcloud.zip https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-261.0.0-windows-x86_64-bundled-python.zip
unzip -q gcloud.zip
