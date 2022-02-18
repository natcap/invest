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

$env:PATH += ";C:\ProgramData\chocolatey\bin"

# Choco-provided command to reload environment variables
refreshenv
