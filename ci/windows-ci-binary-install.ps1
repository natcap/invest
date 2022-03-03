# Download and install NSIS plugins to their correct places.
# Update the list of cached files in .github/workflows/build-and-test.ym if we add or remove any plugins.
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
