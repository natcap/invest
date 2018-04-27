:: Execute the jenkins build.
::
:: This script takes no parameters.
::
:: This script assumes that it will be called from the repo root 
:: as .\jenkins-test.bat
make env
env\Scripts\python.exe -m pip install --upgrade .
env\Scripts\python.exe -m pip install -r requirements-dev.txt
make PYTHON=env\Scripts\python.exe test
