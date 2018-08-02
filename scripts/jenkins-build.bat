:: Execute the jenkins build.
::
:: This script takes no parameters.
::
:: This script assumes that it will be called from the repo root through the
:: ``make jenkins`` target on Windows.
::
:: Unlike bash, which has ``set -e`` to fail on the first error encountered,
:: CMD requires us to check the exit code after each command.
:: See https://stackoverflow.com/a/734634/299084 for slightly more info.
::
:: Using ``exit /b`` causes only this batch script to exit with an error code,
:: not the parent CMD process.  We probably want this, since we don't want to
:: immediately quit the jenkins CMD process.
make env
if %errorlevel% neq 0 exit /b %errorlevel%

env\Scripts\python.exe -m pip install --upgrade .
if %errorlevel% neq 0 exit /b %errorlevel%

:: GUI requirements are not automatically installed as part of the ``make env`` recipe.
env\Scripts\python.exe -m pip install -r requirements-gui.txt
if %errorlevel% neq 0 exit /b %errorlevel%

make PYTHON=env\Scripts\python.exe windows_installer sampledata userguide python_packages
if %errorlevel% neq 0 exit /b %errorlevel%

env\scripts\python.exe scripts\jenkins_push_artifacts.py
if %errorlevel% neq 0 exit /b %errorlevel%
