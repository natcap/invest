@echo off

FOR %a in (%*) DO (
    @where /Q %a
    IF %ERRORLEVEL% NEQ 0 (
        echo \033[31mMISSING\033[0m: %a
    ) ELSE (
        echo \033[32mOK\e[0m: %a
    )
)
