# A wrapper around the powershell equivalent of zipping functionality.
# This script takes 3 parameters:
#  --> whether to recurse (this is ignored, but we have to take it anyways for compatibility)
#  --> The target archive path
#  --> The folder to zip.
#
param (
    [Parameter(Mandatory=$false, ValueFromPipeline=$false, ParameterSetName='r')]
    [switch]
    $recurse,
    
    [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
    [String[]]
    $target_file,

    [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
    [String[]]
    $directory
)

# Calling the powershell-equivalent of GNU Zip
Compress-Archive `
    -CompressionLevel Optimal `
    -DestinationPath "$target_file" `
    -Path "$directory"
