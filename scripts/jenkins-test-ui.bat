:: Execute the jenkins UI tests.
::
:: This script takes no parameters.
::
:: This script assumes that it will be called from the repo root 
:: as .\jenkins-test-ui.bat
make env
env\Scripts\python.exe -m pip install --upgrade .
env\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-gui.txt
make PYTHON=env\Scripts\python.exe test
