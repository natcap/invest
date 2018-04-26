:: Execute the jenkins build.
::
:: This script takes no parameters.
::
:: This script assumes that it will be called from the repo root through the
:: ``make jenkins`` target on Windows.
make env
env\Scripts\python.exe -m pip install --upgrade .
env\Scripts\python.exe install -r requirements-gui.txt
make PYTHON=env\Scripts\python.exe windows_installer sampledata userguide python_packages
env\scripts\python.exe scripts\jenkins_push_artifacts.py
