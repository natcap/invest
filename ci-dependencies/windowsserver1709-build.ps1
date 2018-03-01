# Based on the docker image microsoft/windowsservercore

# Install Chocolatey.
# Taken from https://chocolatey.org/install
# After install, reload all environment variables
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
refreshenv

# downloads all available chocolatey vcredist packages, both x86 and x64 versions.
choco install -y vcredist-all
choco install -y vcpython27 --version 9.0.0.30729

# scoop is *really* useful for installing standalone developer packages, but
# its index is limited and it doesn't provide dependency resolution.
Set-ExecutionPolicy RemoteSigned -Scope Process
iex (new-object net.webclient).downloadstring('https://get.scoop.sh')

scoop install git-with-openssh@2.16.2.windows.1
scoop install sliksvn@1.9.7
scoop install mercurial@4.5
scoop install make@4.2

# InVEST setup
choco install -y miniconda --forcex86 --version 4.3.21
$env:Path += ";C:\ProgramData\Miniconda2\Scripts"
$env:PYTHONIOENCODING="UTF-8"  # corrects issue with cp65001 as default encoding in powershell/cmd

# create an env for InVEST
conda create -n invest-env python=2.7




