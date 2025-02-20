# install micromamba dependencies: https://github.com/mamba-org/mamba/issues/2928
Invoke-WebRequest -URI "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "$env:Temp\vc_redist.x64.exe"; Start-Process "$env:Temp\vc_redist.x64.exe" -ArgumentList "/quiet /norestart" -Wait; Remove-Item "$env:Temp\vc_redist.x64.exe"
