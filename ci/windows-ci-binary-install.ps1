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

choco install make vcredist140 zip 7zip unzip
$env:PATH += ";C:\ProgramData\chocolatey\bin"

# Choco-provided command to reload environment variables
refreshenv

# Download and install NSIS plugins to their correct places.
Write-Host "Downloading and extracting NSIS"
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/Inetc.zip -OutFile Inetc.zip
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/Nsisunz.zip -OutFile Nsisunz.zip
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/NsProcess.zip -OutFile NsProcess.zip
Invoke-WebRequest https://storage.googleapis.com/natcap-build-dependencies/windows/NsisMultiUser.zip -OutFile NsisMultiUser.zip
& 7z e NsProcess.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugin\nsProcess.dll
& 7z e NsProcess.zip -o"C:\Program Files (x86)\NSIS\Include" Include\nsProcess.nsh
& 7z e Inetc.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugins\x86-ansi\INetC.dll
& 7z e Nsisunz.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" nsisunz\Release\nsisunz.dll
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugins\x86-ansi\StdUtils.dll
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Plugins\x86-ansi" Plugins\x86-ansi\UAC.dll
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Include" Include\NsisMultiUser.nsh
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Include" Include\NsisMultiUserLang.nsh
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Include" Include\StdUtils.nsh
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Include" Include\UAC.nsh
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Include" Demos\Common\Utils.nsh
& 7z e NsisMultiUser.zip -o"C:\Program Files (x86)\NSIS\Include" Demos\Common\un.Utils.nsh
