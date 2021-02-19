; Variables needed at the command line:
; VERSION         - the version of InVEST we're building (example: 3.8.9)
;                   This string must not contain characters that are
;                   problematic in Windows paths (no : , etc)
; BINDIR          - The local folder of binaries to include.
; ARCHITECTURE    - The architecture we're building for. Generally this is x64.
; FORKNAME        - The username of the InVEST fork we're building off of.
; DATA_LOCATION   - Where (relative to datportal) the data should be downloaded
;                   from.
;
; NOTE ON INSTALLING SAMPLE DATA:
; ===============================
; There are three ways to install sample data with this installer:
;
; 1) Through the Installer's GUI.
;    This approach requires users to interact with the GUI of the installer,
;    where the user will select the data zipfile he/she would like to have
;    installed as part of the installation. If the user does not have an active
;    internet connection (or if there are problems with a download), an error
;    dialog will be presented for each failed download.
;
; 2) Through the 'Advanced' input on the front pane of the installer.
;    This approach is particularly convenient for users wishing to distribute
;    sample data as a single zipfile with the installer, as might be the case
;    for sysadmins installing on many computers, or Natcappers installing on
;    user's computers at a training.  To make this work, a specially formatted
;    zipfile must be used.  This zipfile may be created with make by calling:
;
;        $ make sampledata_single
;
;    Alternately, this zipfile may be assembled by hand, so long as the
;    zipfile has all sample data folders at the top level.  Whatever is in the
;    archive will be unzipped to the install directory.
;
;    It's also worth noting that this 'Advanced' install may be used at the
;    command-line, optionally as part of a silent install.  If we assume that
;    the InVEST 3.8.9 installer and the advanced sampledata zipfile are copied
;    to the same directory, and we open a cmd prompt within that same
;    directory:
;
;        > .\InVEST_3.8.8_Setup_x64.exe /S /DATAZIP=%CD%\sampledata.zip
;
;    This will execute the installer silently, and extract the contents of
;    sampledata.zip to the installation directory.
;
; 3) By having the installer and sample data archives in the right places
;    This approach is an alternative to the silent install with the 'advanced'
;    input functionality, and is useful when the user has control over the
;    location of the installer and the sampledata zipfiles on the local
;    computer. The gist is that if the installer finds the sample data zipfile
;    it's looking for in the right place, it'll use that instead of going to
;    the network.
;
;    To use this, the following folder structure must exist:
;
;    some directory/
;        InVEST_<version>_Setup.exe
;        sample_data/
;           Marine.zip
;           Pollination.zip
;           Base_Data.zip
;           <other zipfiles, as desired, downloaded from our website>
!include nsProcess.nsh
!include LogicLib.nsh

;;;; NSIS MultiUser Includes;;;;
!include UAC.nsh
!include NsisMultiUser.nsh
!include StdUtils.nsh
;;;;;;;;; ;;;;;;;;;;;;;;;

; HM NIS Edit Wizard helper defines
!define SOFTWARE_NAME "InVEST"
!define PRODUCT_VERSION "${VERSION} ${ARCHITECTURE}"
!define PRODUCT_PUBLISHER "The Natural Capital Project"
!define COMPANY_NAME "The Natural Capital Project"
!define PRODUCT_WEB_SITE "https://naturalcapitalproject.stanford.edu"
!define URL_INFO_ABOUT "https://naturalcapitalproject.stanford.edu"
!define MUI_COMPONENTSPAGE_NODESC
!define PACKAGE_NAME "${SOFTWARE_NAME} ${PRODUCT_VERSION}"
!define PRODUCT_NAME "${SOFTWARE_NAME}_${VERSION}_${ARCHITECTURE}"
!define UNINSTALL_FILENAME "Uninstall_${VERSION}.exe"

; PROGEXE is needed in NsisMultiUser
!define PROGEXE "invest-3-x64\invest.exe" ; main application filename

;;;;; NSIS MultiUser ;;;;;
!define SETUP_MUTEX "${PRODUCT_PUBLISHER} ${SOFTWARE_NAME} Setup Mutex" ; do not change this between program versions!
!define APP_MUTEX "${PRODUCT_PUBLISHER} ${SOFTWARE_NAME} App Mutex" ; do not change this between program versions!
!define INVEST_MUTEX "${SOFTWARE_NAME} ${PRODUCT_VERSION} Mutex" ; do not change this between program versions!
!define SETTINGS_REG_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

; NsisMultiUser optional defines
!define MULTIUSER_INSTALLMODE_ALLOW_BOTH_INSTALLATIONS 0
!define MULTIUSER_INSTALLMODE_ALLOW_ELEVATION 1
; required for silent-mode allusers-uninstall to work, when using the workaround for Windows elevation bug
!define MULTIUSER_INSTALLMODE_ALLOW_ELEVATION_IF_SILENT 1 
!define MULTIUSER_INSTALLMODE_DEFAULT_ALLUSERS 1
; Assume 64-bit for now
!define MULTIUSER_INSTALLMODE_64_BIT 1
!define MULTIUSER_INSTALLMODE_DISPLAYNAME "${SOFTWARE_NAME} ${PRODUCT_VERSION}"
;;;;;;;;; ;;;;;;;;;;

; Installer Attributes
Name "${SOFTWARE_NAME} ${PRODUCT_VERSION}"
OutFile ..\..\dist\InVEST_${FORKNAME}${VERSION}_${ARCHITECTURE}_Setup.exe
ShowInstDetails show
BrandingText "2021 ${PRODUCT_PUBLISHER}"
SetCompressor zlib

; Include after SetCompressor
!include Utils.nsh

; MUI has some graphical files that I want to define, which must be defined
; here before the macros are declared.
;
; NOTES ABOUT GRAPHICS:
; ---------------------
; NSIS is surprisingly picky about the sorts of graphics that can be displayed.
; Here's what I know about these images after a fair amount of
; trial and error:
;  * Image format must be Windows Bitmap (.bmp).
;       * I've used 24-bit ad 32-bit encodings without issue.
;       * 24-bit encodings should be sufficient, and yield ~30% filesize reduction.
;       * If using GIMP, be sure to check the compatibility option marked
;         "Do not write color space information".
;  * Vertical images must have dimensions 164Wx314H.
;       * Within this, the InVEST logo currently has dimensions 130Wx109H.
;  * Horizontal (top) banner must have dimensions 150Wx57H.
;       * Within this, the InVEST logo currently has dimensions 48Wx40H.
;
; GIMP notes: I've had good results with just opening the existing BMPs from
; the repo, inserting a new layer with the InVEST logo, scaling the layer,
; repositioning the logo to perfectly cover the old logo, flattening the
; layers and then exporting as a 24-bit windows bitmap.
!define MUI_WELCOMEFINISHPAGE_BITMAP "InVEST-vertical.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "InVEST-vertical.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "InVEST-header-wcvi-rocks.bmp"
!define MUI_UNHEADERIMAGE_BITMAP "InVEST-header-wcvi-rocks.bmp"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\orange-uninstall.ico"

; MUI 1.67 compatible ------
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"
!include "nsDialogs.nsh"
!include "WinVer.nsh"

; MUI Settings
!define MUI_ABORTWARNING
!define MUI_LANGDLL_ALLLANGUAGES ; Show all languages, despite user's codepage
!define MUI_ICON "InVEST-2.ico"

; Remember the installer language
!define MUI_LANGDLL_REGISTRY_ROOT SHCTX
!define MUI_LANGDLL_REGISTRY_KEY "${SETTINGS_REG_KEY}"
!define MUI_LANGDLL_REGISTRY_VALUENAME "Language"

; Add an advanced options control for the welcome page.
!define MUI_PAGE_CUSTOMFUNCTION_SHOW AddAdvancedOptions
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE ValidateAdvZipFile

;;;;;;;;; NSIS MultiUser ;;;;;;;;
!define MUI_PAGE_CUSTOMFUNCTION_PRE PageWelcomeLicensePre
;;;;;;;;;;;; ;;;;;;;;;;;;;

!insertmacro MUI_PAGE_WELCOME
;;;;;;;;; NSIS MultiUser ;;;;;;;;
!define MUI_PAGE_CUSTOMFUNCTION_PRE PageWelcomeLicensePre
;;;;;;;;;;;; ;;;;;;;;;;;;;
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE.txt"

; Variables
Var StartMenuFolder
!define MULTIUSER_INSTALLMODE_CHANGE_MODE_FUNCTION PageInstallModeChangeMode
!insertmacro MULTIUSER_PAGE_INSTALLMODE

!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipComponents
!insertmacro MUI_PAGE_COMPONENTS

;;;;;;;;;;;;; NSIS MultiUser ;;;;;;;;;;;;;;;;;
!define MUI_PAGE_CUSTOMFUNCTION_PRE PageDirectoryPre
!define MUI_PAGE_CUSTOMFUNCTION_SHOW PageDirectoryShow
;;;;;;;;;; ;;;;;;;;;;;;;;;;;;

!insertmacro MUI_PAGE_DIRECTORY

;;;;;;;;;;;;;; NSIS MultiUser ;;;;;;;;;;;;;;;
!define MUI_STARTMENUPAGE_NODISABLE ; Do not display the checkbox to disable the creation of Start Menu shortcuts
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "${PRODUCT_NAME}"
; writing to $StartMenuFolder happens in MUI_STARTMENU_WRITE_END, so it's safe to use SHCTX here
!define MUI_STARTMENUPAGE_REGISTRY_ROOT SHCTX
!define MUI_STARTMENUPAGE_REGISTRY_KEY "${SETTINGS_REG_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "StartMenuFolder"
!define MUI_PAGE_CUSTOMFUNCTION_PRE PageStartMenuPre
!insertmacro MUI_PAGE_STARTMENU "" "$StartMenuFolder"
; the MUI_PAGE_STARTMENU macro undefines MUI_STARTMENUPAGE_DEFAULTFOLDER, but we need it
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "${PRODUCT_NAME}"
;;;;;;;;;;;;;;; ;;;;;;;;;;;;;;

!define MUI_PAGE_CUSTOMFUNCTION_SHOW PageInstFilesPre
!insertmacro MUI_PAGE_INSTFILES

!insertmacro MUI_PAGE_FINISH

; MUI Uninstaller settings---------------
; Installer Attributes
ShowUninstDetails show

; Interface settings
!define MUI_UNABORTWARNING ; Show a confirmation when cancelling the installation

; Pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!define MULTIUSER_INSTALLMODE_CHANGE_MODE_FUNCTION un.PageInstallModeChangeMode
!insertmacro MULTIUSER_UNPAGE_INSTALLMODE
!insertmacro MUI_UNPAGE_INSTFILES


; Languages (first is default language) - must be inserted after all pages
!insertmacro MUI_LANGUAGE "English"
!insertmacro MULTIUSER_LANGUAGE_INIT

; Reserve files
!insertmacro MUI_RESERVEFILE_LANGDLL

; MUI end ------

; This function allows us to test to see if a process is currently running.
; If the process name passed in is actually found, a message box is presented
; and the uninstaller should quit.
!macro CheckProgramRunning process_name
    ${nsProcess::FindProcess} "${process_name}.exe" $R0
    Pop $R0

    StrCmp $R0 603 +3
        MessageBox MB_OK|MB_ICONEXCLAMATION "InVEST is still running.  Please close all InVEST models and try again."
        Abort
!macroend

Function PageStartMenuPre
    GetDlgItem $1 $HWNDPARENT 1
    Call MultiUser.CheckPageElevationRequired
    ${if} $0 = 2
        SendMessage $1 ${BCM_SETSHIELD} 0 1 ; display SHIELD (Windows Vista and above)
    ${endif}
FunctionEnd

Function PageInstallModeChangeMode
    !insertmacro MUI_STARTMENU_GETFOLDER "" $StartMenuFolder

    ${if} "$StartMenuFolder" == "${MUI_STARTMENUPAGE_DEFAULTFOLDER}"
        !insertmacro MULTIUSER_GetCurrentUserString $0
        StrCpy $StartMenuFolder "$StartMenuFolder$0"
    ${endif}
FunctionEnd

Function PageWelcomeLicensePre
    ${if} $InstallShowPagesBeforeComponents = 0
        Abort ; don't display the Welcome and License pages
    ${endif}
FunctionEnd

Function PageDirectoryPre
    GetDlgItem $1 $HWNDPARENT 1
    SendMessage $1 ${WM_SETTEXT} 0 "STR:$(^NextBtn)" ; this is not the last page before installing
    Call MultiUser.CheckPageElevationRequired
    ${if} $0 = 2
        SendMessage $1 ${BCM_SETSHIELD} 0 1 ; display SHIELD (Windows Vista and above)
    ${endif}
FunctionEnd

Function PageDirectoryShow
    ${if} $CmdLineDir != ""
        FindWindow $R1 "#32770" "" $HWNDPARENT

        GetDlgItem $0 $R1 1019 ; Directory edit
        SendMessage $0 ${EM_SETREADONLY} 1 0 ; read-only is better than disabled, as user can copy contents

        GetDlgItem $0 $R1 1001 ; Browse button
        EnableWindow $0 0
    ${endif}
FunctionEnd

Function PageInstFilesPre
    GetDlgItem $0 $HWNDPARENT 1
    SendMessage $0 ${BCM_SETSHIELD} 0 0 ; hide SHIELD (Windows Vista and above)
FunctionEnd

var AdvCheckbox
var AdvFileField
var AdvZipFile
var LocalDataZipFile
var WarnLabel
Function AddAdvancedOptions
    ; NSD_CREATE* macros take 5 params: x, y, width, height and text
    ${NSD_CreateCheckBox} 120u -18u 15% 12u "Advanced"
    pop $AdvCheckbox
    ${NSD_OnClick} $AdvCheckbox EnableAdvFileSelect

    ${NSD_CreateFileRequest} 175u -18u 36% 12u $LocalDataZipFile
    pop $AdvFileField
    ShowWindow $AdvFileField 0

    ${NSD_CreateBrowseButton} 300u -18u 5% 12u "..."
    pop $AdvZipFile
    ${NSD_OnClick} $AdvZipFile GetZipFile
    ShowWindow $AdvZipFile 0

    ; if $LocalDataZipFile has a value, check the 'advanced' checkbox by default.
    ${If} $LocalDataZipFile != ""
        ${NSD_Check} $AdvCheckbox
        Call EnableAdvFileSelect
    ${EndIf}

    ; if install computer is 32-bit then warn the user of a 64-bit app.
    ${IfNot} ${RunningX64}
        Call WarnArchitecture
    ${EndIf}
FunctionEnd

Function EnableAdvFileSelect
    ${NSD_GetState} $AdvCheckbox $0
    ShowWindow $AdvFileField $0
    ShowWindow $AdvZipFile $0
FunctionEnd

Function GetZipFile
    nsDialogs::SelectFileDialog "open" "" "Zipfiles *.zip"
    pop $0
    ${GetFileExt} $0 $1
    ${If} $1 != "zip"
        MessageBox MB_OK "File must be a zipfile"
        Abort
    ${EndIf}
    ${NSD_SetText} $AdvFileField $0
    strcpy $LocalDataZipFile $0
FunctionEnd

Function SkipComponents
    GetDlgItem $0 $HWNDPARENT 1
    SendMessage $0 ${BCM_SETSHIELD} 0 0 ; hide Shield (Windows Vista and above)
    ${If} $LocalDataZipFile != ""
        Abort
    ${EndIf}
FunctionEnd

Function ValidateAdvZipFile
    ${NSD_GetText} $AdvFileField $0
    ${If} $0 != ""
        ${GetFileExt} $0 $1
        ${If} $1 != "zip"
            MessageBox MB_OK "File must be a zipfile $1"
            Abort
        ${EndIf}
        IfFileExists $0 +3 0
        MessageBox MB_OK "File not found or not accessible: $0"
        Abort
    ${Else}
        ; Save the value in the advanced filefield as $LocalDataZipFile
        strcpy $LocalDataZipFile $0
    ${EndIf}
FunctionEnd

Function WarnArchitecture
    !define msg1 " Warning: 32-bit architecture detected. Installing $\r$\n"
    !define msg2 " 64-bit software could lead to unexpected results. $\r$\n"
    ${NSD_CreateLabel} 120u -56u 50% 16u "${msg1}${msg2}"
    pop $WarnLabel
    SetCtlColors $WarnLabel 0x000000 0xFFE300
FunctionEnd

; NSIS 3.x defines these variables, NSIS 2.x does not.  This supports both versions.
!ifndef LVM_GETITEMCOUNT
    !define LVM_GETITEMCOUNT 0x1004
!endif
!ifndef LVM_GETITEMTEXT
    !define LVM_GETITEMTEXT 0x102D
!endif

Function DumpLog
    Exch $5
    Push $0
    Push $1
    Push $2
    Push $3
    Push $4
    Push $6

    FindWindow $0 "#32770" "" $HWNDPARENT
    GetDlgItem $0 $0 1016
    StrCmp $0 0 exit
    FileOpen $5 $5 "w"
    StrCmp $5 "" exit
        SendMessage $0 ${LVM_GETITEMCOUNT} 0 0 $6
        System::Alloc ${NSIS_MAX_STRLEN}
        Pop $3
        StrCpy $2 0
        System::Call "*(i, i, i, i, i, i, i, i, i) i \
            (0, 0, 0, 0, 0, r3, ${NSIS_MAX_STRLEN}) .r1"
        loop: StrCmp $2 $6 done
            System::Call "User32::SendMessageA(i, i, i, i) i \
            ($0, ${LVM_GETITEMTEXT}, $2, r1)"
            System::Call "*$3(&t${NSIS_MAX_STRLEN} .r4)"
            FileWrite $5 "$4$\r$\n"
            IntOp $2 $2 + 1
            Goto loop
        done:
            FileClose $5
            System::Free $1
            System::Free $3
    exit:
        Pop $6
        Pop $4
        Pop $3
        Pop $2
        Pop $1
        Pop $0
        Exch $5
FunctionEnd

; Copied into the invest folder later in the NSIS script
!define INVEST_BINARIES "$INSTDIR\invest-3-x64"
!define INVEST_ICON "${INVEST_BINARIES}\InVEST-2.ico"
!define UNINSTALL_ICON "${INVEST_BINARIES}\InVEST-2.ico"
!define SAMPLEDATADIR "$INSTDIR\sample_data"
!macro StartMenuLink linkName modelName
    CreateShortCut "${linkName}.lnk" "${INVEST_BINARIES}\invest.exe" "run ${modelName}" "${INVEST_ICON}"
!macroend

; Sections
Section "InVEST Tools" Section_InVEST_Tools
    AddSize 230793  ; This size is based on Windows build of InVEST 3.4.0
    SectionIn RO ;require this section

    ; Write the uninstaller to disk
    SetOutPath "$INSTDIR"
    !define UNINSTALL_PATH "$INSTDIR\Uninstall_${VERSION}.exe"
    writeUninstaller "${UNINSTALL_PATH}"

    ;;;;;;;;;;;;;; NSIS MultiUSER ;;;;;;;;;;;;
    !insertmacro MULTIUSER_RegistryAddInstallInfo ; add registry keys
    ;;;;;;;;;;;;;;;;;;;;;

    !insertmacro MUI_STARTMENU_WRITE_BEGIN ""
        ; Create start  menu shortcuts.
        ; These shortcut paths are set in the appropriate places based on the SetShellVarConext flag.
        ; This flag is automatically set based on the MULTIUSER installation mode selected by the user.
        ;!define SMPATH "$SMPROGRAMS\${PACKAGE_NAME}"
        !define SMPATH "$SMPROGRAMS\$StartMenuFolder"
        CreateDirectory "${SMPATH}"
        !insertmacro StartMenuLink "${SMPATH}\Crop Production (Percentile)" "crop_production_percentile"
        !insertmacro StartMenuLink "${SMPATH}\Crop Production (Regression)" "crop_production_regression"
        !insertmacro StartMenuLink "${SMPATH}\Scenic Quality" "scenic_quality"
        !insertmacro StartMenuLink "${SMPATH}\Habitat Quality" "habitat_quality"
        !insertmacro StartMenuLink "${SMPATH}\Carbon" "carbon"
        !insertmacro StartMenuLink "${SMPATH}\Forest Carbon Edge Effect" "forest_carbon_edge_effect"
        !insertmacro StartMenuLink "${SMPATH}\GLOBIO" "globio"
        !insertmacro StartMenuLink "${SMPATH}\Pollination" "pollination"
        !insertmacro StartMenuLink "${SMPATH}\Finfish Aquaculture" "finfish_aquaculture"
        !insertmacro StartMenuLink "${SMPATH}\Wave Energy" "wave_energy"
        !insertmacro StartMenuLink "${SMPATH}\Wind Energy" "wind_energy"
        !insertmacro StartMenuLink "${SMPATH}\Coastal Vulnerability" "cv"
        !insertmacro StartMenuLink "${SMPATH}\SDR: Sediment Delivery Ratio" "sdr"
        !insertmacro StartMenuLink "${SMPATH}\NDR: Nutrient Delivery Ratio" "ndr"
        !insertmacro StartMenuLink "${SMPATH}\Scenario Generator: Proximity Based" "sgp"
        !insertmacro StartMenuLink "${SMPATH}\Water Yield" "hwy"
        !insertmacro StartMenuLink "${SMPATH}\Seasonal Water Yield" "swy"
        !insertmacro StartMenuLink "${SMPATH}\RouteDEM" "routedem"
        !insertmacro StartMenuLink "${SMPATH}\DelineateIt" "delineateit"
        !insertmacro StartMenuLink "${SMPATH}\Recreation" "recreation"
        !insertmacro StartMenuLink "${SMPATH}\Urban Flood Risk Mitigation" "ufrm"
        !insertmacro StartMenuLink "${SMPATH}\Urban Cooling Model" "ucm"
        !insertmacro StartMenuLink "${SMPATH}\Habitat Risk Assessment" "hra"

        !define COASTALBLUECARBON "${SMPATH}\Coastal Blue Carbon"
        CreateDirectory "${COASTALBLUECARBON}"
        !insertmacro StartMenuLink "${COASTALBLUECARBON}\Coastal Blue Carbon (1) Preprocessor" "cbc_pre"
        !insertmacro StartMenuLink "${COASTALBLUECARBON}\Coastal Blue Carbon (2)" "cbc"

        !define FISHERIES "${SMPATH}\Fisheries"
        CreateDirectory "${FISHERIES}"
        !insertmacro StartMenuLink "${FISHERIES}\Fisheries" "fisheries"
        !insertmacro StartMenuLink "${FISHERIES}\Fisheries Habitat Scenario Tool" "fisheries_hst"

        ;;;;;;;;;;;; NSIS MultiUser ;;;;;;;;;;;;;;;;;;;
        ${if} $MultiUser.InstallMode == "AllUsers"
            CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "${UNINSTALL_PATH}" "/allusers"
        ${else}
            CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "${UNINSTALL_PATH}" "/currentuser"
        ${endif}
        ;;;;;;;;;;;;;; ;;;;;;;;;;;;;;;;;;;;;

    !insertmacro MUI_STARTMENU_WRITE_END

    ; Actually install the information we want to disk.
    SetOutPath "$INSTDIR"
    File ..\..\LICENSE.txt
    file ..\..\HISTORY.rst

    SetOutPath "${SAMPLEDATADIR}"
    ; Copy over all the sample parameter files
    File ..\..\data\invest-sample-data\*.invs.json
    File ..\..\data\invest-sample-data\*.invest.json

    SetOutPath "${INVEST_BINARIES}"
    File /r /x *.hg* /x *.svn* ..\..\${BINDIR}\*
    ; invest-autotest.bat is here to help automate testing the UIs.
    File invest-autotest.bat
    File ..\..\scripts\invest-autotest.py
    File InVEST-2.ico

    SetOutPath "$INSTDIR\documentation"
    File /r /x *.hg* /x *.svn* ..\..\dist\userguide

    ; If the user has provided a custom data zipfile, unzip the data.
    ${If} $LocalDataZipFile != ""
        nsisunz::UnzipToLog $LocalDataZipFile "${SAMPLEDATADIR}"
    ${EndIf}

    ; Write the install log to a text file on disk.
    StrCpy $0 "$INSTDIR\install_log.txt"
    Push $0
    Call DumpLog

SectionEnd

; Only add this section if we're running the installer on Windows 7 or below.
; See InVEST Issue #3515 (https://bitbucket.org/natcap/invest/issues/3515)
; This section is disabled in .onInit if we're running Windows 8 or later.
Section "MSVCRT 2008 Runtime (Recommended)" Sec_VCRedist2008
    SetOutPath "$INSTDIR"
    File ..\..\build\vcredist_x86.exe
    ExecWait "vcredist_x86.exe /q"
SectionEnd

Var LocalDataZip
Var INSTALLER_DIR

!macro downloadFile RemoteFilepath LocalFilepath
    ; Using inetc instead of the included NSISdl because inetc plugin supports downloading over HTTPS.
    ; NSISdl only supports downloading over HTTP.  This will be a problem when serving datasets from
    ; storage buckets, which is only done over HTTPS.
    inetc::get "${RemoteFilepath}" "${LocalFilepath}" /END
    Pop $R0 ;Get the status of the file downloaded
    StrCmp $R0 "OK" got_it failed
    got_it:
        nsisunz::UnzipToLog ${LocalFilepath} "."
        Delete ${LocalFilepath}
        goto done
    failed:
        MessageBox MB_OK "Download failed: $R0 ${RemoteFilepath}. This might have happened because your Internet connection timed out, or our download server is experiencing problems.  The installation will continue normally, but you'll be missing the ${RemoteFilepath} dataset in your installation.  You can manually download that later by visiting the 'Individual inVEST demo datasets' section of our download page at www.naturalcapitalproject.org."
    done:
!macroend

!macro downloadData Title Filename AdditionalSizeKb
    ; AdditionalSizeKb is in kilobytes.  Easy way to find this out is to do
    ; "$ du -BK -c <directory with model sample data>" and then use the total.
    Section "${Title}"
        AddSize "${AdditionalSizeKb}"

        ; Check to see if the user defined an 'advanced options' zipfile.
        ; If yes, then we should skip all of this checking, since we only want to use
        ; the data that was in that zip.
        ${If} $LocalDataZipFile != ""
            goto end_of_section
        ${EndIf}

        ; Use a local zipfile if it exists in ./sample_data
        ${GetExePath} $INSTALLER_DIR
        StrCpy $LocalDataZip "$INSTALLER_DIR\sample_data\${Filename}"

        ; MessageBox MB_OK "zip: $LocalDataZip"
        IfFileExists "$LocalDataZip" LocalFileExists DownloadFile
        LocalFileExists:
            nsisunz::UnzipToLog "$LocalDataZip" "${SAMPLEDATADIR}"
            ; MessageBox MB_OK "found it locally"
        goto done
        DownloadFile:
            ;This is hard coded so that all the download data macros go to the same site
            SetOutPath "${SAMPLEDATADIR}"
            !insertmacro downloadFile "${DATA_LOCATION}/${Filename}" "${Filename}"
        end_of_section:
    SectionEnd
!macroend

SectionGroup /e "InVEST Datasets" SEC_DATA
    ;here all the numbers indicate the size of the downloads in kilobytes
    ;they were calculated by hand by decompressing all the .zip files and recording
    ;the size by hand.
    !insertmacro downloadData "Annual Water Yield (optional)" "Annual_Water_Yield.zip" 20513
    !insertmacro downloadData "Aquaculture (optional)" "Aquaculture.zip" 116
    !insertmacro downloadData "Carbon (optional)" "Carbon.zip" 17748
    !insertmacro downloadData "Coastal Blue Carbon (optional)" "CoastalBlueCarbon.zip" 332
    !insertmacro downloadData "Coastal Vulnerability (optional)" "CoastalVulnerability.zip" 169918
    !insertmacro downloadData "Crop Production (optional)" "CropProduction.zip" 111898
    !insertmacro downloadData "DelineateIt (optional)" "DelineateIt.zip" 536
    !insertmacro downloadData "Fisheries (optional)" "Fisheries.zip" 637
    !insertmacro downloadData "Forest Carbon Edge Effect (required for forest carbon edge model)" "forest_carbon_edge_effect.zip" 8060
    !insertmacro downloadData "GLOBIO (optional)" "globio.zip" 186020
    !insertmacro downloadData "Habitat Quality (optional)" "HabitatQuality.zip" 1880
    !insertmacro downloadData "Habitat Risk Assessment (optional)" "HabitatRiskAssess.zip" 7791
    !insertmacro downloadData "Nutrient Delivery Ratio (optional)" "NDR.zip" 10973
    !insertmacro downloadData "Pollination (optional)" "pollination.zip" 687
    !insertmacro downloadData "Recreation (optional)" "recreation.zip" 5826
    !insertmacro downloadData "RouteDEM (optional)" "RouteDEM.zip" 532
    !insertmacro downloadData "Scenario Generator: Proximity Based (optional)" "scenario_proximity.zip" 7508
    !insertmacro downloadData "Scenic Quality (optional)" "ScenicQuality.zip" 165792
    !insertmacro downloadData "Seasonal Water Yield: (optional)" "Seasonal_Water_Yield.zip" 6044
    !insertmacro downloadData "Sediment Delivery Ratio (optional)" "SDR.zip" 15853
    !insertmacro downloadData "Urban Flood Risk Mitigation (optional)" "UrbanFloodMitigation.zip" 688
    !insertmacro downloadData "Urban Cooling Model (optional)" "UrbanCoolingModel.zip" 6885
    !insertmacro downloadData "Wave Energy (required to run model)" "WaveEnergy.zip" 831423
    !insertmacro downloadData "Wind Energy (required to run model)" "WindEnergy.zip" 7984
    !insertmacro downloadData "Global DEM & Polygon (optional)" "Base_Data.zip" 631322

    Section "-hidden section"
        ; StrCpy is only available inside a Section or Function.
        ; Write the install log to a text file on disk.
        StrCpy $0 "$INSTDIR\install_data_log.txt"
        Push $0
        Call DumpLog
    SectionEnd
SectionGroupEnd


Function .onInit
    ${GetOptions} $CMDLINE "/?" $0

    ;;;;;;;;;;; NSIS MultiUser ;;;;;;;;;;;;;;;;;
    ; this is really just checking if there is another instance of the 
    ; installer running
    ${ifnot} ${UAC_IsInnerInstance}
        !insertmacro CheckSingleInstance "Setup" "Global" "${SETUP_MUTEX}"
        !insertmacro CheckSingleInstance "Application" "Local" "${APP_MUTEX}"
    ${endif}

    !insertmacro MULTIUSER_INIT

    ${if} $IsInnerInstance = 0
        !insertmacro MUI_LANGDLL_DISPLAY
    ${endif}
    ;;;;;;;;;;;;;;;; ;;;;;;;;;;;;;;;;;;;;;;;;;;

    IfErrors skiphelp showhelp
    showhelp:
         MessageBox MB_OK "InVEST: Integrated Valuation of Ecosystem Services and Tradeoffs$\r$\n\
         $\r$\n\
         For more information about InVEST or the Natural Capital Project, visit our \
         website: https://naturalcapitalproject.stanford.edu/invest$\r$\n\
         $\r$\n\
         Command-Line Options:$\r$\n\
             /?$\t$\t=$\tDisplay this help and exit$\r$\n\
             /S$\t$\t=$\tSilently install InVEST.$\r$\n\
             /D=$\t$\t=$\tSet the installation directory.$\r$\n\
             /DATAZIP=$\t=$\tUse this sample data zipfile.$\r$\n\
             "
         abort
    skiphelp:

        ;System::Call 'kernel32::CreateMutexA(i 0, i 0, t "InVEST ${VERSION}") i .r1 ?e'
        ;Pop $R0

        ;StrCmp $R0 0 +3
        ;    MessageBox MB_OK|MB_ICONEXCLAMATION "An InVEST ${VERSION} installer is already running."
        ;    Abort

        ${ifNot} ${AtMostWin7}
            ; disable the section if we're not running on Windows 7 or earlier.
            ; This section should not execute for Windows 8 or later.
            SectionGetFlags ${Sec_VCRedist2008} $0
            IntOp $0 $0 & ${SECTION_OFF}
            SectionSetFlags ${Sec_VCRedist2008} $0
            SectionSetText ${Sec_VCRedist2008} ""
        ${endIf}

        ; If the user has defined the /DATAZIP flag, set the 'advanced' option
        ; to the user's defined value.
        ${GetOptions} $CMDLINE "/DATAZIP=" $0
        strcpy $LocalDataZipFile $0
FunctionEnd

; remove next line if you're using signing after the uninstaller is extracted from the initially compiled setup
!include Uninstall.nsh
