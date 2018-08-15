@echo off
REM Check that all programs provided at the command line are available on the
REM PATH.
REM
REM Parameters:
REM     * - program names to look for on the PATH.
REM
REM Example invokation:
REM     .\check_required_programs.bat program1 program2 program3
REM
REM No fun colors here ... standard CMD prompt doesn't support it.

FOR %%a in (%*) DO (
    @where /Q %%a && ( echo OK: %%a ) || ( echo MISSING: %%a )
)
