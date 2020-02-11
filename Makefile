# Repositories managed by the makefile task tree
DATA_DIR := data
GIT_SAMPLE_DATA_REPO        := https://bitbucket.org/natcap/invest-sample-data.git
GIT_SAMPLE_DATA_REPO_PATH   := $(DATA_DIR)/invest-sample-data
GIT_SAMPLE_DATA_REPO_REV    := a280ef2cf79b7a794ebb3e8678ca27cbe37d1f5b

GIT_TEST_DATA_REPO          := https://bitbucket.org/natcap/invest-test-data.git
GIT_TEST_DATA_REPO_PATH     := $(DATA_DIR)/invest-test-data
GIT_TEST_DATA_REPO_REV      := 0dfeed7f216f6c1f9af815aabb3e7e04f8cf662b

HG_UG_REPO                  := https://bitbucket.org/natcap/invest.users-guide
HG_UG_REPO_PATH             := doc/users-guide
HG_UG_REPO_REV              := ec7070b5dea47089718795e71beff182240d4203


ENV = env
ifeq ($(OS),Windows_NT)
	NULL := $$null
	PROGRAM_CHECK_SCRIPT := .\scripts\check_required_programs.bat
	ENV_SCRIPTS = $(ENV)\Scripts
	ENV_ACTIVATE = $(ENV_SCRIPTS)\activate
	CP := powershell.exe Copy-Item
	COPYDIR := $(CP) -Recurse
	MKDIR := powershell.exe mkdir -Force -Path
	RM := powershell.exe Remove-Item -Force -Recurse -Path
	RMDIR := cmd /C "rmdir /S /Q"
	# Windows doesn't install a python3 binary, just python.
	PYTHON = python
	# Just use what's on the PATH for make.  Avoids issues with escaping spaces in path.
	MAKE := make
	SHELL := powershell.exe
	BASHLIKE_SHELL_COMMAND := cmd.exe /C
	.DEFAULT_GOAL := windows_installer
	JENKINS_BUILD_SCRIPT := .\scripts\jenkins-build.bat
	RM_DATA_DIR := $(RM) $(DATA_DIR)
	/ := '\'
else
	NULL := /dev/null
	PROGRAM_CHECK_SCRIPT := ./scripts/check_required_programs.sh
	SHELL := /bin/bash
	BASHLIKE_SHELL_COMMAND := $(SHELL) -c
	CP := cp
	COPYDIR := $(CP) -r
	MKDIR := mkdir -p
	RM := rm -r
	RMDIR := $(RM)
	/ := /
	# linux, mac distinguish between python2 and python3
	PYTHON = python3
	RM_DATA_DIR := yes | rm -r $(DATA_DIR)

	ifeq ($(shell sh -c 'uname -s 2>/dev/null || echo not'),Darwin)  # mac OSX
		.DEFAULT_GOAL := mac_installer
		JENKINS_BUILD_SCRIPT := ./scripts/jenkins-build.sh
	else
		.DEFAULT_GOAL := binaries
		JENKINS_BUILD_SCRIPT := @echo "NOTE: There is not currently a linux jenkins build."; exit 1
	endif
endif

REQUIRED_PROGRAMS := make zip pandoc $(PYTHON) git git-lfs hg
ifeq ($(OS),Windows_NT)
	REQUIRED_PROGRAMS += makensis
endif

PIP = $(PYTHON) -m pip
VERSION := $(shell $(PYTHON) setup.py --version)
PYTHON_ARCH := $(shell $(PYTHON) -c "import sys; print('x86' if sys.maxsize <= 2**32 else 'x64')")

GSUTIL := gsutil
SIGNTOOL := SignTool

# Output directory names
DIST_DIR := dist
DIST_DATA_DIR := $(DIST_DIR)/data
BUILD_DIR := build

# The fork name and user here are derived from the mercurial path.
# They will need to be set manually (e.g. make FORKNAME=natcap/invest)
# if someone wants to build from source outside of mercurial (like if
# they grabbed a zipfile of the source code)
# FORKUSER should not need to be set from the CLI.
FORKNAME := $(filter-out ssh: http: https:, $(subst /, ,$(shell hg config paths.default)))
FORKUSER := $(word 2, $(subst /, ,$(FORKNAME)))
ifeq ($(FORKUSER),natcap)
	BUCKET := gs://releases.naturalcapitalproject.org
	DIST_URL_BASE := $(BUCKET)/invest/$(VERSION)
else
	BUCKET := gs://natcap-dev-build-artifacts
	DIST_URL_BASE := $(BUCKET)/invest/$(FORKUSER)/$(VERSION)
endif
DOWNLOAD_DIR_URL := $(subst gs://,https://storage.googleapis.com/,$(DIST_URL_BASE))
DATA_BASE_URL := $(DOWNLOAD_DIR_URL)/data


TESTRUNNER := $(PYTHON) -m nose -vsP --with-coverage --cover-package=natcap.invest --cover-erase --with-xunit --cover-tests --cover-html --cover-xml --with-timer

DATAVALIDATOR := $(PYTHON) scripts/invest-autovalidate.py $(GIT_SAMPLE_DATA_REPO_PATH)
TEST_DATAVALIDATOR := $(PYTHON) -m nose -vsP scripts/invest-autovalidate.py

# Target names.
INVEST_BINARIES_DIR := $(DIST_DIR)/invest
APIDOCS_HTML_DIR := $(DIST_DIR)/apidocs
APIDOCS_ZIP_FILE := $(DIST_DIR)/InVEST_$(VERSION)_apidocs.zip
USERGUIDE_HTML_DIR := $(DIST_DIR)/userguide
USERGUIDE_ZIP_FILE := $(DIST_DIR)/InVEST_$(VERSION)_userguide.zip
MAC_DISK_IMAGE_FILE := "$(DIST_DIR)/InVEST_$(VERSION).dmg"
MAC_BINARIES_ZIP_FILE := "$(DIST_DIR)/InVEST-$(VERSION)-mac.zip"
MAC_APPLICATION_BUNDLE := "$(BUILD_DIR)/mac_app_$(VERSION)/InVEST.app"


.PHONY: fetch install binaries apidocs userguide windows_installer mac_installer sampledata sampledata_single test test_ui clean help check python_packages jenkins purge mac_zipfile deploy signcode $(GIT_SAMPLE_DATA_REPO_PATH) $(GIT_TEST_DATA_REPO_PATH) $(HG_UG_REPO_REV)

# Very useful for debugging variables!
# $ make print-FORKNAME, for example, would print the value of the variable $(FORKNAME)
print-%:
	@echo "$* = $($*)"

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
	@echo "  mac_installer     to build a disk image for distribution"
	@echo "  sampledata        to build sample data zipfiles"
	@echo "  sampledata_single to build a single self-contained data zipfile.  Used for advanced NSIS install."
	@echo "  test              to run nosetests on the tests directory"
	@echo "  test_ui           to run nosetests on the ui_tests directory"
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

clean:
	$(PYTHON) setup.py clean
	-$(RMDIR) $(BUILD_DIR)
	-$(RMDIR) natcap.invest.egg-info
	-$(RMDIR) cover
	-$(RM) coverage.xml

purge: clean
	-$(RM_DATA_DIR)
	-$(RMDIR) $(HG_UG_REPO_PATH)
	-$(RMDIR) $(ENV)

check:
	@echo "Checking required applications"
	@$(PROGRAM_CHECK_SCRIPT) $(REQUIRED_PROGRAMS)
	@echo "----------------------------"
	@echo "Checking python packages"
	@$(PIP) freeze --all -r requirements.txt -r requirements-dev.txt > $(NULL)


# Subrepository management.
$(HG_UG_REPO_PATH):
	-hg clone --noupdate $(HG_UG_REPO) $(HG_UG_REPO_PATH)
	-hg pull $(HG_UG_REPO) -R $(HG_UG_REPO_PATH)
	hg update -r $(HG_UG_REPO_REV) -R $(HG_UG_REPO_PATH)

$(GIT_SAMPLE_DATA_REPO_PATH): | $(DATA_DIR)
	-git clone $(GIT_SAMPLE_DATA_REPO) $(GIT_SAMPLE_DATA_REPO_PATH)
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) fetch
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) lfs install
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) lfs fetch
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) fetch
	git -C $(GIT_SAMPLE_DATA_REPO_PATH) checkout $(GIT_SAMPLE_DATA_REPO_REV)

$(GIT_TEST_DATA_REPO_PATH): | $(DATA_DIR)
	-git clone $(GIT_TEST_DATA_REPO) $(GIT_TEST_DATA_REPO_PATH)
	git -C $(GIT_TEST_DATA_REPO_PATH) fetch
	git -C $(GIT_TEST_DATA_REPO_PATH) lfs install
	git -C $(GIT_TEST_DATA_REPO_PATH) lfs fetch
	git -C $(GIT_TEST_DATA_REPO_PATH) fetch
	git -C $(GIT_TEST_DATA_REPO_PATH) checkout $(GIT_TEST_DATA_REPO_REV)

fetch: $(HG_UG_REPO_PATH) $(GIT_SAMPLE_DATA_REPO_PATH) $(GIT_TEST_DATA_REPO_PATH)


# Python environment management
env:
    ifeq ($(OS),Windows_NT)
		$(PYTHON) -m virtualenv --system-site-packages $(ENV)
		$(BASHLIKE_SHELL_COMMAND) "$(ENV_ACTIVATE) && $(PIP) install -r requirements.txt -r requirements-gui.txt"
		$(BASHLIKE_SHELL_COMMAND) "$(ENV_ACTIVATE) && $(PIP) install -I -r requirements-dev.txt"
		$(BASHLIKE_SHELL_COMMAND) "$(ENV_ACTIVATE) && $(MAKE) install"
    else
		$(PYTHON) ./scripts/convert-requirements-to-conda-yml.py requirements.txt requirements-dev.txt requirements-gui.txt > requirements-all.yml
		conda create -p $(ENV) -y -c conda-forge
		conda env update -p $(ENV) --file requirements-all.yml
		$(BASHLIKE_SHELL_COMMAND) "source activate ./$(ENV) && $(MAKE) install"
    endif

# compatible with pip>=7.0.0
# REQUIRED: Need to remove natcap.invest.egg-info directory so recent versions
# of pip don't think CWD is a valid package.
install: $(DIST_DIR)/natcap.invest%.whl
	-$(RMDIR) natcap.invest.egg-info
	$(PIP) install --isolated --upgrade --only-binary natcap.invest --find-links=dist natcap.invest


# Bulid python packages and put them in dist/
python_packages: $(DIST_DIR)/natcap.invest%.whl $(DIST_DIR)/natcap.invest%.zip
$(DIST_DIR)/natcap.invest%.whl: | $(DIST_DIR)
	$(PYTHON) setup.py bdist_wheel

$(DIST_DIR)/natcap.invest%.zip: | $(DIST_DIR)
	$(PYTHON) setup.py sdist --formats=zip


# Build binaries and put them in dist/invest
# The `invest list` is to test the binaries.  If something doesn't
# import, we want to know right away.  No need to provide the `.exe` extension
# on Windows as the .exe extension is assumed.
binaries: $(INVEST_BINARIES_DIR)
$(INVEST_BINARIES_DIR): | $(DIST_DIR) $(BUILD_DIR)
	-$(RMDIR) $(BUILD_DIR)/pyi-build
	-$(RMDIR) $(INVEST_BINARIES_DIR)
	$(PYTHON) -m PyInstaller --workpath $(BUILD_DIR)/pyi-build --clean --distpath $(DIST_DIR) exe/invest.spec
	$(BASHLIKE_SHELL_COMMAND) "$(PYTHON) -m pip freeze --all > $(INVEST_BINARIES_DIR)/package_versions.txt"
	$(INVEST_BINARIES_DIR)/invest list

# Documentation.
# API docs are copied to dist/apidocs
# Userguide HTML docs are copied to dist/userguide
# Userguide PDF file is copied to dist/InVEST_<version>_.pdf
apidocs: $(APIDOCS_HTML_DIR) $(APIDOCS_ZIP_FILE)
$(APIDOCS_HTML_DIR): | $(DIST_DIR)
	$(PYTHON) setup.py build_sphinx -a --source-dir doc/api-docs
	$(COPYDIR) build/sphinx/html $(APIDOCS_HTML_DIR)

$(APIDOCS_ZIP_FILE): $(APIDOCS_HTML_DIR)
	$(BASHLIKE_SHELL_COMMAND) "cd $(DIST_DIR) && zip -r $(notdir $(APIDOCS_ZIP_FILE)) $(notdir $(APIDOCS_HTML_DIR))"

userguide: $(USERGUIDE_HTML_DIR) $(USERGUIDE_ZIP_FILE)
$(USERGUIDE_HTML_DIR): $(HG_UG_REPO_PATH) | $(DIST_DIR)
	$(MAKE) -C doc/users-guide SPHINXBUILD=sphinx-build BUILDDIR=../../build/userguide html
	-$(RMDIR) $(USERGUIDE_HTML_DIR)
	$(COPYDIR) build/userguide/html dist/userguide

$(USERGUIDE_ZIP_FILE): $(USERGUIDE_HTML_DIR)
	$(BASHLIKE_SHELL_COMMAND) "cd $(DIST_DIR) && zip -r $(notdir $(USERGUIDE_ZIP_FILE)) $(notdir $(USERGUIDE_HTML_DIR))"

# Tracking the expected zipfiles here avoids a race condition where we can't
# know which data zipfiles to create until the data repo is cloned.
# All data zipfiles are written to dist/data/*.zip
ZIPDIRS = Annual_Water_Yield \
		  Aquaculture \
		  Base_Data \
		  Carbon \
		  CoastalBlueCarbon \
		  CoastalVulnerability \
		  CropProduction \
		  DelineateIt \
		  Fisheries \
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
		  storm_impact \
		  UrbanFloodMitigation \
		  UrbanCoolingModel \
		  WaveEnergy \
		  WindEnergy

ZIPTARGETS = $(foreach dirname,$(ZIPDIRS),$(addprefix $(DIST_DATA_DIR)/,$(dirname).zip))

sampledata: $(ZIPTARGETS)
$(DIST_DATA_DIR)/%.zip: $(DIST_DATA_DIR) $(GIT_SAMPLE_DATA_REPO_PATH)
	cd $(GIT_SAMPLE_DATA_REPO_PATH); $(BASHLIKE_SHELL_COMMAND) "zip -r $(addprefix ../../,$@) $(subst $(DIST_DATA_DIR)/,$(DATADIR),$(subst .zip,,$@))"

SAMPLEDATA_SINGLE_ARCHIVE := dist/InVEST_$(VERSION)_sample_data.zip
sampledata_single: $(SAMPLEDATA_SINGLE_ARCHIVE)

$(SAMPLEDATA_SINGLE_ARCHIVE): $(GIT_SAMPLE_DATA_REPO_PATH) dist
	$(BASHLIKE_SHELL_COMMAND) "cd $(GIT_SAMPLE_DATA_REPO_PATH) && zip -r ../../$(SAMPLEDATA_SINGLE_ARCHIVE) ./* -x .svn -x .git -x *.json"


# Installers for each platform.
# Windows (NSIS) installer is written to dist/InVEST_<version>_x86_Setup.exe
# Mac (DMG) disk image is written to dist/InVEST <version>.dmg
ifeq ($(FORKUSER), natcap)
	INSTALLER_NAME_FORKUSER :=
else
	INSTALLER_NAME_FORKUSER := $(FORKUSER)
endif
WINDOWS_INSTALLER_FILE := $(DIST_DIR)/InVEST_$(INSTALLER_NAME_FORKUSER)$(VERSION)_$(PYTHON_ARCH)_Setup.exe
windows_installer: $(WINDOWS_INSTALLER_FILE)
$(WINDOWS_INSTALLER_FILE): $(INVEST_BINARIES_DIR) $(USERGUIDE_HTML_DIR) build/vcredist_x86.exe | $(GIT_SAMPLE_DATA_REPO_PATH)
	-$(RM) $(WINDOWS_INSTALLER_FILE)
	makensis /DVERSION=$(VERSION) /DBINDIR=$(INVEST_BINARIES_DIR) /DARCHITECTURE=$(PYTHON_ARCH) /DFORKNAME=$(INSTALLER_NAME_FORKUSER) /DDATA_LOCATION=$(DATA_BASE_URL) installer\windows\invest_installer.nsi

mac_app: $(MAC_APPLICATION_BUNDLE)
$(MAC_APPLICATION_BUNDLE): $(BUILD_DIR) $(INVEST_BINARIES_DIR)
	./installer/darwin/build_app_bundle.sh "$(VERSION)" "$(INVEST_BINARIES_DIR)" "$(MAC_APPLICATION_BUNDLE)"

mac_installer: $(MAC_DISK_IMAGE_FILE)
$(MAC_DISK_IMAGE_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_HTML_DIR)
	./installer/darwin/build_dmg.sh "$(VERSION)" "$(MAC_APPLICATION_BUNDLE)" "$(USERGUIDE_HTML_DIR)"

mac_zipfile: $(MAC_BINARIES_ZIP_FILE)
$(MAC_BINARIES_ZIP_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_HTML_DIR)
	./installer/darwin/build_zip.sh "$(VERSION)" "$(MAC_APPLICATION_BUNDLE)" "$(USERGUIDE_HTML_DIR)"

build/vcredist_x86.exe: | build
	powershell.exe -Command "Start-BitsTransfer -Source https://download.microsoft.com/download/5/D/8/5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x86.exe -Destination build\vcredist_x86.exe"

jenkins:
	$(JENKINS_BUILD_SCRIPT)

jenkins_test_ui: env
	$(MAKE) PYTHON=$(ENV_SCRIPTS)/python test_ui

jenkins_test: env $(GIT_TEST_DATA_REPO_PATH)
	$(MAKE) PYTHON=$(ENV_SCRIPTS)/python test

CERT_FILE := StanfordUniversity.crt
KEY_FILE := Stanford-natcap-code-signing-2019-03-07.key.pem
signcode:
	$(GSUTIL) cp gs://stanford_cert/$(CERT_FILE) $(BUILD_DIR)/$(CERT_FILE)
	$(GSUTIL) cp gs://stanford_cert/$(KEY_FILE) $(BUILD_DIR)/$(KEY_FILE)
	# On some OS (including our build container), osslsigncode fails with Bus error if we overwrite the binary when signing.
	osslsigncode -certs $(BUILD_DIR)/$(CERT_FILE) -key $(BUILD_DIR)/$(KEY_FILE) -pass $(CERT_KEY_PASS) -in $(BIN_TO_SIGN) -out "signed.exe"
	mv "signed.exe" $(BIN_TO_SIGN)
	rm $(BUILD_DIR)/$(CERT_FILE)
	rm $(BUILD_DIR)/$(KEY_FILE)
	@echo "Installer was signed with osslsigncode"

P12_FILE := Stanford-natcap-code-signing-2019-03-07.p12
signcode_windows:
	$(BASHLIKE_SHELL_COMMAND) "$(GSUTIL) cp gs://stanford_cert/$(P12_FILE) $(BUILD_DIR)/$(P12_FILE)"
	powershell.exe "& '$(SIGNTOOL)' sign /f '$(BUILD_DIR)\$(P12_FILE)' /p '$(CERT_KEY_PASS)' '$(BIN_TO_SIGN)'"
	-powershell.exe "Remove-Item $(BUILD_DIR)/$(P12_FILE)"
	@echo "Installer was signed with signtool"

deploy:
	$(GSUTIL) -m rsync -r $(DIST_DIR)/userguide $(DIST_URL_BASE)/userguide
	$(GSUTIL) -m rsync -r $(DIST_DIR)/data $(DIST_URL_BASE)/data
	$(GSUTIL) -m rsync $(DIST_DIR) $(DIST_URL_BASE)
	@echo "Binaries (if they were created) can be downloaded from:"
	@echo "  * $(DOWNLOAD_DIR_URL)/$(subst $(DIST_DIR)/,,$(WINDOWS_INSTALLER_FILE))"
    ifeq ($(BUCKET),gs://releases.naturalcapitalproject.org)  # ifeq cannot follow TABs, only spaces
		$(GSUTIL) cp "$(BUCKET)/fragment_id_redirections.json" "$(BUILD_DIR)/fragment_id_redirections.json"
		$(PYTHON) scripts/update_installer_urls.py "$(BUILD_DIR)/fragment_id_redirections.json" $(BUCKET) $(notdir $(WINDOWS_INSTALLER_FILE)) $(notdir $(patsubst "%",%,$(MAC_BINARIES_ZIP_FILE)))
		$(GSUTIL) cp "$(BUILD_DIR)/fragment_id_redirections.json" "$(BUCKET)/fragment_id_redirections.json"
    endif


# Notes on Makefile development
#
# * Use the -drR to show the decision tree (and none of the implicit rules)
#   if a task is (or is not) executing when expected.
# * Use -n to print the actions to be executed instead of actually executing them.
