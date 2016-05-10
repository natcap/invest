:: runmodel.bat
::
:: USAGE: runmodel.bat MODELNAME
::
:: Run the given model via the InVEST CLI, writing the model's exit status to
:: .\invest_bintest_results.txt.  This model will be run with the --test flag,
:: assuming all default parameters.

set logfile=invest_bintest_results.txt
call .\invest --test %*
if errorlevel 1 goto failed
    echo "Success: %*" >> %logfile%
    goto :end
:failed
    echo "Failure: %*" >> %logfile%

:end
exit /B 0
