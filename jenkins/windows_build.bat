:: Jenkins build script for Windows.
::
:: Necessary because controlling a virtual environment doesn't work
:: so well from within a python session.
::

SET ENV=release_env
call %ENV%\Scripts\activate.bat
paver build

