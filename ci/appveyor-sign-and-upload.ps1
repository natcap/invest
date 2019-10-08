# This script uses a GCP service account to sign the installer and upload artifacts to GCS.
#
# The specific steps involved are:
#   1. Download and unzip a self-contained gcloud SDK
#   2. Create a service account key file from an environment variable
#      and authenticate with GCP using the key.
#   3. Copy the code-signing certificate to the local VM.
#   4. Sign the InVEST installer using the certificate.
#   5. Upload the artifacts to the correct bucket via make deploy.
#
# Requirements:
#   * An environment variable GOOGLE_SERVICE_ACC_KEY exists and contains a
#     base64-encoded string version of the service account key to use.
#   * An environment variable STANFORD_CERT_KEY_PASS exists and contains
#     the string password for our code-signing certificate.
#   * The InVEST installer has been created correctly and exists at dist/<installer>.exe
#   * The Windows 10 SDK is installed locally.
#   * The command-line applications unzip, curl and make are available.

# Download, quietly unzip and install the google cloud utilities.
curl -o gcloud.zip https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-261.0.0-windows-x86_64-bundled-python.zip
unzip -q gcloud.zip

# GCloud service account key conversion from base64 taken from https://stackoverflow.com/a/56140959/299084
# Assumes that GOOGLE_SERVICE_ACC_KEY is a base64-encoded JSON service account key.
$content = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($env:GOOGLE_SERVICE_ACC_KEY))
$Utf8NoBomEncoding = New-Object System.Text.UTF8Encoding $False
[System.IO.File]::WriteAllLines('C:/Users/appveyor/client-secret.json', $content, $Utf8NoBomEncoding)
cmd.exe --% /c .\google-cloud-sdk\bin\gcloud auth activate-service-account --key-file=%USERPROFILE%\client-secret.json

# figure out the path to signtool.exe (it keeps changing with SDK updates)
$env:SIGNTOOL_PATH = @(Get-ChildItem -Path 'C:\Program Files (x86)\\Windows Kits\\10' -Include 'signtool.exe' -File -Recurse -ErrorAction SilentlyContinue)[0] | Select-Object -ExpandProperty FullName

# get the path to the installer and sign it.
$env:INSTALLER_BINARY = @(gci 'dist/*.exe')[0] | Select-Object -ExpandProperty FullName

# gsutil writes its output to stderr, which is treated as an error in appveyor.
# https://help.appveyor.com/discussions/problems/5413-calling-external-executable-causes-nativecommanderror-despite-no-apparent-error
cmd.exe --% /c make GSUTIL=.\google-cloud-sdk\bin\gsutil CERT_KEY_PASS=%STANFORD_CERT_KEY_PASS% BIN_TO_SIGN="%INSTALLER_BINARY%" SIGNTOOL="%SIGNTOOL_PATH%" signcode_windows

# push the artifacts
cmd.exe --% /c make GSUTIL=.\google-cloud-sdk\bin\gsutil deploy
