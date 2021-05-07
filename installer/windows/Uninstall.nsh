!include un.Utils.nsh

; Variables
Var SemiSilentMode ; installer started uninstaller in semi-silent mode using /SS parameter
Var RunningFromInstaller ; installer started uninstaller using /uninstall parameter
Var RunningAsShellUser ; uninstaller restarted itself under the user of the running shell

Section "un.Program Files" SectionUninstallProgram
    SectionIn RO

    !insertmacro MULTIUSER_GetCurrentUserString $0

    ; InvokeShellVerb only works on existing files, so we call it before
    ; deleting the EXE, https://github.com/lordmulder/stdutils/issues/22

    ; Clean up "Start Menu Icon"
    ${if} ${AtLeastWin7}
        ${StdUtils.InvokeShellVerb} $1 "$INSTDIR" "${PROGEXE}" ${StdUtils.Const.ShellVerb.UnpinFromStart}
    ${else}
        !insertmacro DeleteRetryAbort "$STARTMENU\${PRODUCT_NAME}$0.lnk"
    ${endif}

    ; Try to delete the EXE as the first step - if it's in use, don't remove anything else
    !insertmacro DeleteRetryAbort "$INSTDIR\${PROGEXE}"

    ; Clean up "Program Group" - we check that we created Start menu folder, 
    ; if $StartMenuFolder is empty, the whole $SMPROGRAMS directory will be removed!
    ${if} "$StartMenuFolder" != ""
        RMDir /r "$SMPROGRAMS\$StartMenuFolder"
    ${endif}

    ; this section is executed only explicitly and shouldn't be placed in SectionUninstallProgram
    DeleteRegKey HKCU "Software\${PRODUCT_NAME}"
SectionEnd

Section "-Uninstall" ; hidden section, must always be the last one!
    ; we cannot use un.DeleteRetryAbort here - when using the _? parameter
    ; the uninstaller cannot delete itself and Delete fails, which is OK
    Delete "$INSTDIR\${UNINSTALL_FILENAME}"
    ; Wipe the install directory regardless of what's in there.
    rmdir /r "$INSTDIR"

    ; Remove the uninstaller from registry as the very last step - if sth. goes wrong, let the user run it again
    !insertmacro MULTIUSER_RegistryRemoveInstallInfo ; Remove registry keys	

    ; If the uninstaller still exists, use cmd.exe on exit to remove it (along with $INSTDIR if it's empty)
    ${if} ${FileExists} "$INSTDIR\${UNINSTALL_FILENAME}"
        Exec 'cmd.exe /c (del /f /q "$INSTDIR\${UNINSTALL_FILENAME}") && (rmdir "$INSTDIR")'
    ${endif}
SectionEnd

; Callbacks
Function un.onInit
    ${GetParameters} $R0

    !insertmacro CheckProgramRunning "invest"

    ${GetOptions} $R0 "/uninstall" $R1
    ${ifnot} ${errors}
        StrCpy $RunningFromInstaller 1
    ${else}
        StrCpy $RunningFromInstaller 0
    ${endif}

    ${GetOptions} $R0 "/SS" $R1
    ${ifnot} ${errors}
        StrCpy $SemiSilentMode 1
        StrCpy $RunningFromInstaller 1
        ; auto close (if no errors) if we are called from the installer; 
        ; if there are errors, will be automatically set to false
        SetAutoClose true
    ${else}
        StrCpy $SemiSilentMode 0
    ${endif}

    ${GetOptions} $R0 "/shelluser" $R1
    ${ifnot} ${errors}
        StrCpy $RunningAsShellUser 1
    ${else}
        StrCpy $RunningAsShellUser 0
    ${endif}

    ${ifnot} ${UAC_IsInnerInstance}
    ${andif} $RunningFromInstaller = 0
        ; Restarting the uninstaller using the user of the running shell, in order to overcome the Windows bugs that:
        ; - Elevates the uninstallers of single-user installations when called from 'Apps & features' of Windows 10
        ; causing them to fail when using a different account for elevation.
        ; - Elevates the uninstallers of all-users installations when called from 'Add/Remove Programs' of Control Panel,
        ; preventing them of eleveting on their own and correctly recognize the user that started the uninstaller. If a
        ; different account was used for elevation, all user-context operations will be performed for the user of that
        ; account. In this case, the fix causes the elevetion prompt to be displayed twice (one from Control Panel and
        ; one from the uninstaller).
        ${if} ${UAC_IsAdmin}
        ${andif} $RunningAsShellUser = 0
            ${StdUtils.ExecShellAsUser} $0 "$INSTDIR\${UNINSTALL_FILENAME}" "open" "/shelluser $R0"
            Quit
        ${endif}
        !insertmacro CheckSingleInstance "Setup" "Global" "${SETUP_MUTEX}"
        !insertmacro CheckSingleInstance "Application" "Local" "${APP_MUTEX}"
    ${endif}

    !insertmacro MULTIUSER_UNINIT

    !insertmacro MUI_UNGETLANGUAGE ; we always get the language, since the outer and inner instance might have different language
FunctionEnd

Function un.PageInstallModeChangeMode
    !insertmacro MUI_STARTMENU_GETFOLDER "" $StartMenuFolder
FunctionEnd

Function un.onUninstFailed
    ${if} $SemiSilentMode = 0
        MessageBox MB_ICONSTOP "${PRODUCT_NAME} ${VERSION} could not be fully uninstalled.$\r$\nPlease, restart Windows and run the uninstaller again." /SD IDOK
    ${else}
        MessageBox MB_ICONSTOP "${PRODUCT_NAME} could not be fully installed.$\r$\nPlease, restart Windows and run the setup program again." /SD IDOK
    ${endif}
FunctionEnd
