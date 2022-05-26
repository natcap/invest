# Repositories managed by the makefile task tree
DATA_DIR := data
GIT_SAMPLE_DATA_REPO        := https://bitbucket.org/natcap/invest-sample-data.git
GIT_SAMPLE_DATA_REPO_PATH   := $(DATA_DIR)/invest-sample-data
GIT_SAMPLE_DATA_REPO_REV    := 1ba3d13680caff0b8382de22987a4ac118ec98a6

GIT_TEST_DATA_REPO          := https://bitbucket.org/natcap/invest-test-data.git
GIT_TEST_DATA_REPO_PATH     := $(DATA_DIR)/invest-test-data
GIT_TEST_DATA_REPO_REV      := ac7023d684478485fea89c68f8f4154163541e1d

GIT_UG_REPO                 := https://github.com/natcap/invest.users-guide
GIT_UG_REPO_PATH            := doc/users-guide
GIT_UG_REPO_REV             := eef0643feea534db130198e81a69a0590c1a8480

ENV = "./env"
ifeq ($(OS),Windows_NT)
	# Double $$ indicates windows environment variable
	NULL := $$null
	PROGRAM_CHECK_SCRIPT := .\scripts\check_required_programs.bat
	ENV_SCRIPTS = $(ENV)\Scripts
	ENV_ACTIVATE = $(ENV_SCRIPTS)\activate
	CP := cp
	COPYDIR := $(CP) -r
	MKDIR := mkdir -p
	RM := rm
	RMDIR := $(RM) -r
	# Windows doesn't install a python3 binary, just python.
	PYTHON = python
	# Just use what's on the PATH for make.  Avoids issues with escaping spaces in path.
	MAKE := make
	# Powershell has been inconsistent for allowing make commands to be
	# ignored on failure. Many times if a command writes to std error
	# powershell interprets that as a failure and exits. Bash shells are
	# widely available on Windows now, especially through git-bash
	SHELL := /usr/bin/bash
	CONDA := conda.bat
	BASHLIKE_SHELL_COMMAND := '$(SHELL)' -c
	.DEFAULT_GOAL := windows_installer
	RM_DATA_DIR := $(RMDIR) $(DATA_DIR)
	/ := '\'
	OSNAME = 'windows'
else
	NULL := /dev/null
	PROGRAM_CHECK_SCRIPT := ./scripts/check_required_programs.sh
	SHELL := /bin/bash
	CONDA := conda
	BASHLIKE_SHELL_COMMAND := $(SHELL) -c
	CP := cp
	COPYDIR := $(CP) -r
	MKDIR := mkdir -p
	RM := rm
	RMDIR := $(RM) -r
	/ := /
	# linux, mac distinguish between python2 and python3
	PYTHON = python3
	RM_DATA_DIR := yes | $(RMDIR) $(DATA_DIR)

	ifeq ($(shell sh -c 'uname -s 2>/dev/null || echo not'),Darwin)  # mac OSX
		.DEFAULT_GOAL := mac_dmg
		OSNAME = 'mac'
	else
		.DEFAULT_GOAL := binaries
	endif
endif

REQUIRED_PROGRAMS := make zip pandoc $(PYTHON) git git-lfs conda yarn
ifeq ($(OS),Windows_NT)
	REQUIRED_PROGRAMS += makensis
endif

ZIP := zip
PIP = $(PYTHON) -m pip
VERSION := $(shell $(PYTHON) -m setuptools_scm)
PYTHON_ARCH := $(shell $(PYTHON) -c "import sys; print('x86' if sys.maxsize <= 2**32 else 'x64')")

GSUTIL := gsutil
SIGNTOOL := SignTool

# local directory names
DIST_DIR := dist
DIST_DATA_DIR := $(DIST_DIR)/data
BUILD_DIR := build
WORKBENCH := workbench
WORKBENCH_DIST_DIR := $(WORKBENCH)/dist

# The fork name and user here are derived from the git path on github.
# The fork name will need to be set manually (e.g. make FORKNAME=natcap/invest)
# if someone wants to build from source outside of git (like if they grabbed
# a zipfile of the source code).
FORKNAME := $(word 2, $(subst :,,$(subst github.com, ,$(shell git remote get-url origin))))
FORKUSER := $(word 1, $(subst /, ,$(FORKNAME)))

# We use these release buckets here in Makefile and also in our release scripts.
# See scripts/release-3-publish.sh.
RELEASES_BUCKET := gs://releases.naturalcapitalproject.org
DEV_BUILD_BUCKET := gs://natcap-dev-build-artifacts

ifeq ($(FORKUSER),natcap)
	BUCKET := $(RELEASES_BUCKET)
	DIST_URL_BASE := $(BUCKET)/invest/$(VERSION)
	INSTALLER_NAME_FORKUSER :=
else
	BUCKET := $(DEV_BUILD_BUCKET)
	DIST_URL_BASE := $(BUCKET)/invest/$(FORKUSER)/$(VERSION)
	INSTALLER_NAME_FORKUSER := $(FORKUSER)
endif
DOWNLOAD_DIR_URL := $(subst gs://,https://storage.googleapis.com/,$(DIST_URL_BASE))
DATA_BASE_URL := $(DOWNLOAD_DIR_URL)/data

TESTRUNNER := pytest -vs --import-mode=importlib --durations=0

DATAVALIDATOR := $(PYTHON) scripts/invest-autovalidate.py $(GIT_SAMPLE_DATA_REPO_PATH)
TEST_DATAVALIDATOR := $(PYTHON) -m pytest -vs scripts/invest-autovalidate.py

UG_FILE_VALIDATOR := $(PYTHON) scripts/userguide-filevalidator.py $(GIT_UG_REPO_PATH)

# Target names.
INVEST_BINARIES_DIR := $(DIST_DIR)/invest
# INVEST_BINARIES_DIR_ZIP := $(OSNAME)_invest_binaries.zip

APIDOCS_BUILD_DIR := $(BUILD_DIR)/sphinx/apidocs
APIDOCS_TARGET_DIR := $(DIST_DIR)/apidocs
APIDOCS_ZIP_FILE := $(DIST_DIR)/InVEST_$(VERSION)_apidocs.zip

USERGUIDE_BUILD_DIR := $(BUILD_DIR)/sphinx/userguide
USERGUIDE_TARGET_DIR := $(DIST_DIR)/userguide
USERGUIDE_ZIP_FILE := $(DIST_DIR)/InVEST_$(VERSION)_userguide.zip

MAC_DISK_IMAGE_FILE := $(DIST_DIR)/InVEST_$(VERSION).dmg
MAC_BINARIES_ZIP_FILE := $(DIST_DIR)/InVEST-$(VERSION)-mac.zip
MAC_APPLICATION_BUNDLE_NAME := InVEST.app
MAC_APPLICATION_BUNDLE_DIR := $(BUILD_DIR)/mac_app_$(VERSION)
MAC_APPLICATION_BUNDLE := $(MAC_APPLICATION_BUNDLE_DIR)/$(MAC_APPLICATION_BUNDLE_NAME)


.PHONY: fetch install binaries apidocs userguide windows_installer mac_dmg sampledata sampledata_single test test_ui clean help check python_packages jenkins purge mac_zipfile deploy codesign_mac codesign_windows $(GIT_SAMPLE_DATA_REPO_PATH) $(GIT_TEST_DATA_REPO_PATH) $(GIT_UG_REPO_REV)

# Very useful for debugging variables!
# $ make print-FORKNAME, for example, would print the value of the variable $(FORKNAME)
print-%:
	@echo "$* = $($*)"

# Very useful for printing variables within scripts!
# Like `make print-<variable>, only without also printing the variable name.
# the 'j' prefix stands for just.  We're just printing the variable name.
jprint-%:
	@echo "$($*)"

help:
	@echo "Please use make <target> where <target> is one of"
	@echo "  check             to verify all needed programs and packages are installed"
	@echo "  env               to create a virtualenv with packages from requirements.txt, requirements-dev.txt"
	@echo "  fetch             to clone all managed repositories"
	@echo "  install           to build and install a wheel of natcap.invest into the active python installation"
	@echo "  binaries          to build pyinstaller binaries"
	@echo "  apidocs           to build HTML API documentation"
	@echo "  userguide         to build HTML version of the users guide"
	@echo "  python_packages   to build natcap.invest wheel and source distributions"
	@echo "  windows_installer to build an NSIS installer for distribution"
	@echo "  mac_dmg           to build a disk image for distribution"
	@echo "  codesign_mac      to sign the mac disk image using the codesign utility"
	@echo "  codesign_windows  to sign the windows installer using the SignTool utility"
	@echo "  sampledata        to build sample data zipfiles"
	@echo "  sampledata_single to build a single self-contained data zipfile.  Used for advanced NSIS install."
	@echo "  test              to run pytest on the tests directory"
	@echo "  test_ui           to run pytest on the ui_tests directory"
	@echo "  clean             to remove temporary directories and files (but not dist/)"
	@echo "  purge             to remove temporary directories, cloned repositories and the built environment."
	@echo "  help              to print this help and exit"

$(BUILD_DIR) $(DATA_DIR) $(DIST_DIR) $(DIST_DATA_DIR):
	$(MKDIR) $@

test: $(GIT_TEST_DATA_REPO_PATH)
	$(TESTRUNNER) tests

test_ui: $(GIT_TEST_DATA_REPO_PATH)
	$(TESTRUNNER) ui_tests

validate_sampledata: $(GIT_SAMPLE_DATA_REPO_PATH)
	$(TEST_DATAVALIDATOR)
	$(DATAVALIDATOR)

validate_userguide_filenames: $(GIT_UG_REPO_PATH)
	$(UG_FILE_VALIDATOR)

clean:
	-$(RMDIR) $(BUILD_DIR)
	-$(RMDIR) natcap.invest.egg-info
	-$(RMDIR) cover
	-$(RMDIR) doc/api-docs/api
	-$(RM) coverage.xml

purge: clean
	-$(RM_DATA_DIR)
	-$(RMDIR) $(GIT_UG_REPO_PATH)
	-$(RMDIR) $(ENV)

check:
	@echo "Checking required applications"
	@$(PROGRAM_CHECK_SCRIPT) $(REQUIRED_PROGRAMS)
	@echo "----------------------------"
	@echo "Checking python packages"
	@$(PIP) freeze --all -r requirements.txt -r requirements-dev.txt > $(NULL)


# Subrepository management.
$(GIT_UG_REPO_PATH):
	-git clone $(GIT_UG_REPO) $(GIT_UG_REPO_PATH)
	git -C $(GIT_UG_REPO_PATH) fetch
	git -C $(GIT_UG_REPO_PATH) checkout $(GIT_UG_REPO_REV)

$(GIT_SAMPLE_DATA_REPO_PATH): | $(DATA_DIR)
	-git clone $(GIT_SAMPLE_DATA_REPO) $(GIT_SAMPLE_DATA_REPO_PATH)
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) fetch
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) lfs install
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) lfs fetch
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) checkout $(GIT_SAMPLE_DATA_REPO_REV)
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) lfs checkout

$(GIT_TEST_DATA_REPO_PATH): | $(DATA_DIR)
	-git clone $(GIT_TEST_DATA_REPO) $(GIT_TEST_DATA_REPO_PATH)
	git -C $(GIT_TEST_DATA_REPO_PATH) fetch
	git -C $(GIT_TEST_DATA_REPO_PATH) lfs install
	git -C $(GIT_TEST_DATA_REPO_PATH) lfs fetch
	git -C $(GIT_TEST_DATA_REPO_PATH) checkout $(GIT_TEST_DATA_REPO_REV)
	git -C $(GIT_TEST_DATA_REPO_PATH) lfs checkout

fetch: $(GIT_UG_REPO_PATH) $(GIT_SAMPLE_DATA_REPO_PATH) $(GIT_TEST_DATA_REPO_PATH)


# Python conda environment management
env:
	@echo "NOTE: requires 'requests' be installed in base Python"
	$(PYTHON) ./scripts/convert-requirements-to-conda-yml.py requirements.txt requirements-dev.txt requirements-gui.txt > requirements-all.yml
	$(CONDA) create -p $(ENV) -y -c conda-forge python=3.8 nomkl
	$(CONDA) env update -p $(ENV) --file requirements-all.yml
	@echo "----------------------------"
	@echo "To activate the new conda environment and install natcap.invest:"
	@echo ">> conda activate $(ENV)"
	@echo ">> make install"


# compatible with pip>=7.0.0
# REQUIRED: Need to remove natcap.invest.egg-info directory so recent versions
# of pip don't think CWD is a valid package.
install: $(DIST_DIR)/natcap.invest%.whl
	-$(RMDIR) natcap.invest.egg-info
	$(PIP) install --isolated --upgrade --no-index --only-binary natcap.invest --find-links=dist "natcap.invest==$(VERSION)"


# Build python packages and put them in dist/
python_packages: $(DIST_DIR)/natcap.invest%.whl $(DIST_DIR)/natcap.invest%.tar.gz
$(DIST_DIR)/natcap.invest%.whl: | $(DIST_DIR)
	$(PYTHON) -m build --wheel

$(DIST_DIR)/natcap.invest%.tar.gz: | $(DIST_DIR)
	$(PYTHON) -m build --sdist


# Build binaries and put them in dist/invest
# The `invest list` is to test the binaries.  If something doesn't
# import, we want to know right away.  No need to provide the `.exe` extension
# on Windows as the .exe extension is assumed.
binaries: $(INVEST_BINARIES_DIR)
$(INVEST_BINARIES_DIR): | $(DIST_DIR) $(BUILD_DIR)
	-$(RMDIR) $(BUILD_DIR)/pyi-build
	-$(RMDIR) $(INVEST_BINARIES_DIR)
	$(PYTHON) -m PyInstaller --workpath $(BUILD_DIR)/pyi-build --clean --distpath $(DIST_DIR) exe/invest.spec
	$(CONDA) list --export > $(INVEST_BINARIES_DIR)/package_versions.txt
	$(INVEST_BINARIES_DIR)/invest list

# Documentation.
# API docs are built in build/sphinx and copied to dist/apidocs
apidocs: $(APIDOCS_TARGET_DIR) $(APIDOCS_ZIP_FILE)

$(APIDOCS_BUILD_DIR): install  # need contents of build/lib to build apidocs
	# -a: always build all files
	$(PYTHON) -m sphinx -a -b html -d $(APIDOCS_BUILD_DIR)/doctrees doc/api-docs $(APIDOCS_BUILD_DIR)/html

$(APIDOCS_TARGET_DIR): $(APIDOCS_BUILD_DIR) | $(DIST_DIR)
	# only copy over the built html files, not the doctrees
	$(COPYDIR) $(APIDOCS_BUILD_DIR)/html $(APIDOCS_TARGET_DIR)

$(APIDOCS_ZIP_FILE): $(APIDOCS_TARGET_DIR)
	$(BASHLIKE_SHELL_COMMAND) "cd $(DIST_DIR) && $(ZIP) -r $(notdir $(APIDOCS_ZIP_FILE)) $(notdir $(APIDOCS_TARGET_DIR))"

# need to get the working directory path because ln doesn't work with relative paths
WORKING_DIR := $(shell pwd)
ifeq ($(OS),Windows_NT)
	# this setting is necessary for ln to work on git bash for windows
	export MSYS = winsymlinks:nativestrict
endif
# Userguide HTML docs are copied to dist/userguide
userguide: $(USERGUIDE_TARGET_DIR) $(USERGUIDE_ZIP_FILE)
$(USERGUIDE_TARGET_DIR): $(GIT_UG_REPO_PATH) $(GIT_SAMPLE_DATA_REPO_PATH) | $(DIST_DIR)
	ln -s $(WORKING_DIR)/$(GIT_SAMPLE_DATA_REPO_PATH) $(WORKING_DIR)/$(GIT_UG_REPO_PATH)
	$(MAKE) -C $(GIT_UG_REPO_PATH) SPHINXBUILD="$(PYTHON) -m sphinx" BUILDDIR=../../$(USERGUIDE_BUILD_DIR) html
	$(COPYDIR) $(USERGUIDE_BUILD_DIR)/html $(USERGUIDE_TARGET_DIR)

$(USERGUIDE_ZIP_FILE): $(USERGUIDE_TARGET_DIR)
	cd $(DIST_DIR) && $(ZIP) -r $(notdir $(USERGUIDE_ZIP_FILE)) $(notdir $(USERGUIDE_TARGET_DIR))

# Tracking the expected zipfiles here avoids a race condition where we can't
# know which data zipfiles to create until the data repo is cloned.
# All data zipfiles are written to dist/data/*.zip
ZIPDIRS = Annual_Water_Yield \
		  Base_Data \
		  Carbon \
		  CoastalBlueCarbon \
		  CoastalVulnerability \
		  CropProduction \
		  DelineateIt \
		  forest_carbon_edge_effect \
		  globio \
		  GridSeascape \
		  HabitatQuality \
		  HabitatRiskAssess \
		  Malaria \
		  NDR \
		  pollination \
		  recreation \
		  RouteDEM \
		  scenario_proximity \
		  ScenicQuality \
		  SDR \
		  Seasonal_Water_Yield \
		  UrbanCoolingModel \
		  UrbanFloodMitigation \
		  UrbanStormwater \
		  WaveEnergy \
		  WindEnergy

ZIPTARGETS = $(foreach dirname,$(ZIPDIRS),$(addprefix $(DIST_DATA_DIR)/,$(dirname).zip))

sampledata: $(ZIPTARGETS)
	$(PYTHON) scripts/build_sampledata_filesize_registry.py $(CURDIR)/$(DIST_DATA_DIR)
$(DIST_DATA_DIR)/%.zip: $(DIST_DATA_DIR) $(GIT_SAMPLE_DATA_REPO_PATH)
	cd $(GIT_SAMPLE_DATA_REPO_PATH); $(BASHLIKE_SHELL_COMMAND) "$(ZIP) -r $(addprefix ../../,$@) $(subst $(DIST_DATA_DIR)/,$(DATADIR),$(subst .zip,,$@))"

SAMPLEDATA_SINGLE_ARCHIVE := dist/InVEST_$(VERSION)_sample_data.zip
sampledata_single: $(SAMPLEDATA_SINGLE_ARCHIVE)

$(SAMPLEDATA_SINGLE_ARCHIVE): $(GIT_SAMPLE_DATA_REPO_PATH) dist
	$(BASHLIKE_SHELL_COMMAND) "cd $(GIT_SAMPLE_DATA_REPO_PATH) && $(ZIP) -r ../../$(SAMPLEDATA_SINGLE_ARCHIVE) ./* -x .svn -x .git"


# Installers for each platform.
# Windows (NSIS) installer is written to dist/InVEST_<version>_x86_Setup.exe
# Mac (DMG) disk image is written to dist/InVEST <version>.dmg
WINDOWS_INSTALLER_FILE := $(DIST_DIR)/InVEST_$(INSTALLER_NAME_FORKUSER)$(VERSION)_$(PYTHON_ARCH)_Setup.exe
windows_installer: $(WINDOWS_INSTALLER_FILE)
$(WINDOWS_INSTALLER_FILE): $(INVEST_BINARIES_DIR) $(USERGUIDE_ZIP_FILE) build/vcredist_x86.exe | $(GIT_SAMPLE_DATA_REPO_PATH)
	-$(RM) $(WINDOWS_INSTALLER_FILE)
	makensis /DVERSION=$(VERSION) /DBINDIR=$(INVEST_BINARIES_DIR) /DARCHITECTURE=$(PYTHON_ARCH) /DFORKNAME=$(INSTALLER_NAME_FORKUSER) /DDATA_LOCATION=$(DATA_BASE_URL) installer\windows\invest_installer.nsi

mac_dmg: $(MAC_DISK_IMAGE_FILE)
$(MAC_DISK_IMAGE_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)
	# everything in the source directory $(MAC_APPLICATION_BUNDLE_DIR) will be copied into the DMG.
	# so that directory should only contain the app bundle.
	installer/darwin/create_dmg.sh "InVEST $(VERSION)" $(MAC_APPLICATION_BUNDLE_DIR) $(MAC_DISK_IMAGE_FILE)

mac_app: $(MAC_APPLICATION_BUNDLE)
$(MAC_APPLICATION_BUNDLE): $(BUILD_DIR) $(INVEST_BINARIES_DIR) $(USERGUIDE_TARGET_DIR)
	./installer/darwin/build_app_bundle.sh $(VERSION) $(INVEST_BINARIES_DIR) $(USERGUIDE_TARGET_DIR) $(MAC_APPLICATION_BUNDLE)

mac_zipfile: $(MAC_BINARIES_ZIP_FILE)
$(MAC_BINARIES_ZIP_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)
	./installer/darwin/build_zip.sh $(VERSION) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)

build/vcredist_x86.exe: | build
	powershell.exe -Command "Start-BitsTransfer -Source https://download.microsoft.com/download/5/D/8/5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x86.exe -Destination build\vcredist_x86.exe"

KEYCHAIN_NAME := codesign_keychain
# only need password to be able to create the keychain, not for security
KEYCHAIN_PASS := password
codesign_mac:
	# download the p12 certificate file from google cloud
	$(GSUTIL) cp gs://stanford_cert/$(CERT_FILE) $(BUILD_DIR)/$(CERT_FILE)
	# create a new keychain (so that we can know what the password is)
	security create-keychain -p $(KEYCHAIN_PASS) $(KEYCHAIN_NAME)
	# add the keychain to the search list so it can be found
	security list-keychains -s $(KEYCHAIN_NAME)
	# unlock the keychain so we can import to it (stays unlocked 5 minutes by default)
	security unlock-keychain -p $(KEYCHAIN_PASS) $(KEYCHAIN_NAME)
	# add the certificate to the keychain
	# -T option says that the codesign executable can access the keychain
	# for some reason this alone is not enough, also need the following step
	security import $(BUILD_DIR)/$(CERT_FILE) -k $(KEYCHAIN_NAME) -P "$(CERT_PASS)" -T /usr/bin/codesign
	# this is essential to avoid the UI password prompt
	security set-key-partition-list -S apple-tool:,apple: -s -k $(KEYCHAIN_PASS) $(KEYCHAIN_NAME)
	# sign the dmg using certificate that's looked up by unique identifier 'Stanford'
	codesign --timestamp --verbose --sign Stanford $(BIN_TO_SIGN) $(WORKBENCH_BIN_TO_SIGN)

codesign_windows:
	$(GSUTIL) cp gs://stanford_cert/$(CERT_FILE) $(BUILD_DIR)/$(CERT_FILE)
	"$(SIGNTOOL)" sign -fd SHA256 -f $(BUILD_DIR)/$(CERT_FILE) -p $(CERT_PASS) $(BIN_TO_SIGN) $(WORKBENCH_BIN_TO_SIGN)
	"$(SIGNTOOL)" timestamp -tr http://timestamp.sectigo.com -td SHA256 $(BIN_TO_SIGN) $(WORKBENCH_BIN_TO_SIGN)
	$(RM) $(BUILD_DIR)/$(CERT_FILE)
	@echo "Installer was signed with signtool"

deploy:
	-$(GSUTIL) -m rsync $(DIST_DIR) $(DIST_URL_BASE)
	-$(GSUTIL) -m rsync -r $(DIST_DIR)/data $(DIST_URL_BASE)/data
	-$(GSUTIL) -m rsync -r $(DIST_DIR)/userguide $(DIST_URL_BASE)/userguide
	-$(GSUTIL) -m rsync -r $(WORKBENCH_DIST_DIR) $(DIST_URL_BASE)/workbench
	@echo "Application binaries (if they were created) can be downloaded from:"
	@echo "  * $(DOWNLOAD_DIR_URL)"

# Notes on Makefile development
#
# * Use the -drR to show the decision tree (and none of the implicit rules)
#   if a task is (or is not) executing when expected.
# * Use -n to print the actions to be executed instead of actually executing them.
