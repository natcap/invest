# This is an install script needed for our binary build on AppVeyor.
#
# It assumes that the applications choco, wget, 7zip, unzip and curl are available
# and on the PATH.  It also requires that these environment variables are defined:
#
#       GOOGLE_SERVICE_ACC_KEY - the base64-encoded version of the service account key to use
#       STANFORD_CERT_KEY_PASS - the string password to use for our code signing cert
#       PYTHON - the directory of the python installation to use.  Packages for this build
#                are already assumed to be installed and available in this installation.
#
# NOTE: it turns out that `wget` is an alias for the powershell command `Invoke-WebRequest`,
# which I've made a point of using here instead of the actual wget. See https://superuser.com/a/693179

choco install make vcredist140 pandoc zip 7zip unzip
$env:PATH += ";C:\ProgramData\chocolatey\bin"

# Install Zip.  This has been failing recently, so better to just install directly.
# See http://gnuwin32.sourceforge.net/setup.html for full list of installer CLI flags.
# The installer doesn't add the target directory to the PATH, so we need to do that too.
Write-Host "Installing GNU Zip"
Invoke-WebRequest https://managedway.dl.sourceforge.net/project/gnuwin32/zip/3.0/zip-3.0-setup.exe -OutFile zip-setup.exe
& ./zip-setup.exe /VERYSILENT /SP /SUPPRESSMSGBOXES
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files (x86)\GnuWin32\bin", "Machine")

# Choco-provided command to reload environment variables
refreshenv

# Install NSIS.  The choco-provided NSIS puts it somewhere else and
# the choco CLI option --install-directory isn't available in the OSS
# version of choco.
Invoke-WebRequest https://iweb.dl.sourceforge.net/project/nsis/NSIS%203/3.05/nsis-3.05-setup.exe -OutFile nsis.exe

# See http://www.silentinstall.org/nsis for flags used.
& nsis.exe /SD /D="C:\Program Files (x86)\NSIS"

# The binary build requires the shapely DLL to be named something specific.
# /B copies the file as a binary file.
#Write-Host "Copying shapely DLL"
#Copy-Item "$env:PYTHON\Lib\site-packages\shapely\DLLs\geos_c.dll" -Destination "$env:PYTHON\Lib\site-packages\shapely\DLLs\geos.dll"

# Download and install NSIS plugins to their correct places.
Write-Host "Downloading and extracting NSIS"
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/Inetc.zip -OutFile Inetc.zip
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/Nsisunz.zip -OutFile Nsisunz.zip
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/NsProcess.zip -OutFile NsProcess.zip
& 7z e NsProcess.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugin\nsProcess.dll
& 7z e NsProcess.zip -o"C:\Program Files (x86)\NSIS\Include" Include\nsProcess.nsh
& 7z e Inetc.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugins\x86-ansi\INetC.dll
& 7z e Nsisunz.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" nsisunz\Release\nsisunz.dll
