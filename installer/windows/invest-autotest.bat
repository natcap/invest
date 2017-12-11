@REM Run this script from within the invest-3-x86 directory to run the
@REM invest-autotest program on windows.
@REM
@REM Assumes that python is on the PATH.

python "%CD%\invest-autotest.py" --cwd=".." --binary="%CD%\invest.exe
