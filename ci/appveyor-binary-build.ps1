# Attempt to download repositories in parallel
make -j3 fetch

# Build the Windows installer, binaries.
# Attempt to zip things up in parallel
# Python packages were already built in the ``install`` step.
make -j3 sampledata windows_installer

# figure out the path to signtool.exe (it keeps changing with SDK updates)
$env:SIGNTOOL_PATH = @(Get-ChildItem -Path 'C:\Program Files (x86)\\Windows Kits\\10' -Include 'signtool.exe' -File -Recurse -ErrorAction SilentlyContinue)[0] | Select-Object -ExpandProperty FullName

# get the path to the installer and sign it.
$env:INSTALLER_BINARY = @(gci 'dist/*.exe')[0] | Select-Object -ExpandProperty FullName

# gsutil writes its output to stderr, which is treated as an error in appveyor.
# https://help.appveyor.com/discussions/problems/5413-calling-external-executable-causes-nativecommanderror-despite-no-apparent-error
cmd.exe --% /c make GSUTIL=.\google-cloud-sdk\bin\gsutil CERT_KEY_PASS=%STANFORD_CERT_KEY_PASS% BIN_TO_SIGN="%INSTALLER_BINARY%" SIGNTOOL="%SIGNTOOL_PATH%" signcode_windows

# push the artifacts
cmd.exe --% /c make GSUTIL=.\google-cloud-sdk\bin\gsutil deploy
