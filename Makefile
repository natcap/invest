ENV = env
PIP = pip
ifeq ($(OS),Windows_NT)
	NULL := NUL
	PROGRAM_CHECK_SCRIPT := .\scripts\check_required_programs.bat
	ENV_ACTIVATE = .\$(ENV)\Scripts\activate
	CP := Copy-Item
	COPYDIR := powershell.exe Copy-Item -Recurse
	MKDIR := mkdir
	RM := powershell.exe Remove-Item -Force -Recurse -Path
	# Windows doesn't install a python2 binary, just python.
	PYTHON = python
	# Just use what's on the PATH for make.  Avoids issues with escaping spaces in path.
	MAKE := make
	SHELL := powershell.exe
	BASHLIKE_SHELL_COMMAND := cmd.exe /C
	/ := \\
else
	NULL := /dev/null
	PROGRAM_CHECK_SCRIPT := ./scripts/check_required_programs.sh
	ENV_ACTIVATE = source $(ENV)/bin/activate
	SHELL := /bin/bash
	BASHLIKE_SHELL_COMMAND := $(SHELL) -c
	CP := cp -r
	COPYDIR := $(CP)
	MKDIR := mkdir -p
	RM := rm -r
	# linux, mac distinguish between python2 and python3
	PYTHON = python2
	/ := /
endif
VERSION := $(shell $(PYTHON) setup.py --version)
PYTHON_ARCH := $(shell $(PYTHON) -c "import sys; print('x86' if sys.maxsize <= 2**32 else 'x64')")
NOSETESTS := $(PYTHON) -m nose -vsP --with-coverage --cover-package=natcap.invest --cover-erase --with-xunit --cover-tests --cover-html --logging-level=DEBUG
DEST_VERSION := $(shell hg log -r. --template="{ifeq(latesttagdistance,'0',latesttag,'develop')}")
REQUIRED_PROGRAMS := make zip pandoc $(PYTHON) svn hg pdflatex latexmk $(PIP) makensis

# Repositories managed by the makefile task tree
SVN_DATA_REPO           := svn://scm.naturalcapitalproject.org/svn/invest-sample-data
SVN_DATA_REPO_PATH      := data$(/)invest-data$(/)
SVN_DATA_REPO_REV       := 171

SVN_TEST_DATA_REPO      := svn://scm.naturalcapitalproject.org/svn/invest-test-data
SVN_TEST_DATA_REPO_PATH := data$(/)invest-test-data$(/)
SVN_TEST_DATA_REPO_REV  := 139

HG_UG_REPO              := https://bitbucket.org/jdouglass/invest.users-guide
HG_UG_REPO_PATH         := doc$(/)users-guide$(/)
HG_UG_REPO_REV          := e1d238acd5e6



# Target names.
INVEST_BINARIES_DIR := dist$(/)invest$(/)
APIDOCS_HTML_DIR := dist$(/)apidocs$(/)
USERGUIDE_HTML_DIR := dist$(/)userguide$(/)
USERGUIDE_PDF_FILE := dist$(/)InVEST_$(VERSION)_Documentation.pdf
WINDOWS_INSTALLER_FILE := dist$(/)InVEST_$(FORKNAME)$(VERSION)_$(PYTHON_ARCH)_Setup.exe
MAC_DISK_IMAGE_FILE := dist$(/)InVEST $(VERSION).dmg


# These are intended to be overridden by a jenkins build.
# When building a fork, we might set FORKNAME to <username> and DATA_BASE_URL
# to wherever the datasets will be available based on the forkname and where
# we're storing the datasets.
# These defaults assume that we're storing datasets for an InVEST release.
# DEST_VERSION is 'develop' unless we're at a tag, in which case it's the tag.
FORKNAME :=
DATA_BASE_URL := http://data.naturalcapitalproject.org/invest-data/$(DEST_VERSION)

.PHONY: fetch install binaries apidocs userguide windows_installer mac_installer sampledata test test_ui clean help check $(HG_UG_REPO_PATH) $(SVN_DATA_REPO_PATH) $(SVN_TEST_DATA_REPO_PATH) python_packages

# Very useful for debugging variables!
# $ make print-FORKNAME, for example, would print the value of the variable $(FORKNAME)
print-%:
	@echo "$* = $($*)"

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  check             to verify all needed programs and packages are installed"
	@echo "  env               to create a virtualenv with packages from requirements.txt, requirements-dev.txt"
	@echo "  fetch             to clone all managed repositories"
	@echo "  install           to build and install a wheel of natcap.invest into the active python installation"
	@echo "  binaries          to build pyinstaller binaries"
	@echo "  apidocs           to build HTML API documentation"
	@echo "  userguide         to build HTML and PDF versions of the user's guide"
	@echo "  python_packages   to build natcap.invest wheel and source distributions"
	@echo "  windows_installer to build an NSIS installer for distribution"
	@echo "  mac_installer     to build a disk image for distribution"
	@echo "  sampledata        to build sample data zipfiles"
	@echo "  test              to run nosetests on the tests directory"
	@echo "  test_ui           to run nosetests on the ui_tests directory"
	@echo "  clean             to remove temporary directories (but not dist/)"
	@echo "  help              to print this help and exit"

build data dist dist/data:
	$(MKDIR) $@

test: $(SVN_DATA_REPO_PATH) $(SVN_TEST_DATA_REPO_PATH)
	$(NOSETESTS) tests

test_ui:
	$(NOSETESTS) ui_tests

clean:
	$(PYTHON) setup.py clean
	-$(RM) build natcap.invest.egg-info

check:
	@echo Checking required applications
	@$(PROGRAM_CHECK_SCRIPT) $(REQUIRED_PROGRAMS)
	@echo ----------------------------
	@echo Checking python packages
	@$(PIP) freeze --all -r requirements.txt -r requirements-dev.txt > $(NULL)


# Subrepository management.
$(HG_UG_REPO_PATH): data
	$(BASHLIKE_SHELL_COMMAND) "hg update -r $(HG_UG_REPO_REV) -R $(HG_UG_REPO_PATH) || \
		hg clone $(HG_UG_REPO) -u $(HG_UG_REPO_REV) $(HG_UG_REPO_PATH)"

$(SVN_DATA_REPO_PATH): data
	svn checkout $(SVN_DATA_REPO) -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH)

$(SVN_TEST_DATA_REPO_PATH): data
	svn checkout $(SVN_TEST_DATA_REPO) -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH)

fetch: $(HG_UG_REPO_PATH) $(SVN_DATA_REPO_PATH) $(SVN_TEST_DATA_REPO_PATH)


# Python environment management
env:
	$(PYTHON) -m virtualenv --system-site-packages $(ENV)
	$(BASHLIKE_SHELL_COMMAND) "$(ENV_ACTIVATE) && $(PIP) install -r requirements.txt -r requirements-dev.txt"
	$(BASHLIKE_SHELL_COMMAND) "$(ENV_ACTIVATE) && $(MAKE) install"

install: dist/natcap.invest*.whl
	$(PIP) install --use-wheel --find-links=dist natcap.invest 


# Bulid python packages and put them in dist/
python_packages: dist/natcap.invest%.whl dist/natcap.invest%.zip
dist/natcap.invest%.whl: dist
	$(PYTHON) setup.py bdist_wheel

dist/natcap.invest%.zip: dist
	$(PYTHON) setup.py sdist --formats=zip


# Build binaries and put them in dist/invest
binaries: $(INVEST_BINARIES_DIR)
$(INVEST_BINARIES_DIR): dist build
	-$(RM) build/pyi-build
	-$(RM) $(INVEST_BINARIES_DIR)
	pyinstaller \
		--workpath build/pyi-build \
		--clean \
		--distpath dist \
		exe/invest.spec

# Documentation.
# API docs are copied to dist/apidocs
# Userguide HTML docs are copied to dist/userguide
# Userguide PDF file is copied to dist/InVEST_<version>_.pdf
apidocs: $(APIDOCS_HTML_DIR)
$(APIDOCS_HTML_DIR): dist
	$(PYTHON) setup.py build_sphinx -a --source-dir doc/api-docs
	$(CP) build/sphinx/html $(APIDOCS_HTML_DIR)

userguide: $(USERGUIDE_HTML_DIR) $(USERGUIDE_PDF_FILE) 
$(USERGUIDE_PDF_FILE): $(HG_UG_REPO_PATH)
	$(BASHLIKE_SHELL_COMMAND) "cd doc/users-guide && $(MAKE) BUILDDIR=../../build/userguide latex"
	$(BASHLIKE_SHELL_COMMAND) "cd build/userguide/latex && $(MAKE) all-pdf"
	$(CP) build/userguide/latex/InVEST*.pdf dist

$(USERGUIDE_HTML_DIR): $(HG_UG_REPO_PATH) dist
	$(BASHLIKE_SHELL_COMMAND) "cd doc/users-guide && $(MAKE) BUILDDIR=../../build/userguide html"
	-$(RM) $(USERGUIDE_HTML_DIR)
	$(COPYDIR) build/userguide/html dist/userguide


# Zipping up the sample data zipfiles is a little odd because of the presence
# of the Base_Data folder, where its subdirectories are zipped up separately.
# Tracking the expected zipfiles here avoids a race condition where we can't
# know which data zipfiles to create until the data repo is cloned.
# All data zipfiles are written to dist/data/*.zip
ZIPDIRS = AestheticQuality \
		  Aquaculture \
		  Freshwater \
		  Marine \
		  Terrestrial \
		  carbon \
		  CoastalBlueCarbon \
		  CoastalProtection \
		  CropProduction \
		  Fisheries \
		  forest_carbon_edge_effect \
		  globio \
		  GridSeascape \
		  HabitatQuality \
		  HabitatRiskAssess \
		  habitat_suitability \
		  Hydropower \
		  Malaria \
		  OverlapAnalysis \
		  pollination \
		  recreation \
		  ScenarioGenerator \
		  scenario_proximity \
		  ScenicQuality \
		  seasonal_water_yield \
		  storm_impact \
		  WaveEnergy \
		  WindEnergy
ZIPTARGETS = $(foreach dirname,$(ZIPDIRS),$(addprefix dist/data/,$(dirname).zip))

sampledata: $(ZIPTARGETS)
dist/data/Freshwater.zip: DATADIR=Base_Data/
dist/data/Marine.zip: DATADIR=Base_Data/
dist/data/Terrestrial.zip: DATADIR=Base_Data/
dist/data/%.zip: dist/data $(SVN_DATA_REPO_PATH)
	$(BASHLIKE_SHELL_COMMAND) "cd $(SVN_DATA_REPO_PATH) && \
		zip -r $(addprefix ../../,$@) $(subst dist/data/,$(DATADIR),$(subst .zip,,$@))"


# Installers for each platform.
# Windows (NSIS) installer is written to dist/InVEST_<version>_x86_Setup.exe
# Mac (DMG) disk image is written to dist/InVEST <version>.dmg
windows_installer: $(WINDOWS_INSTALLER_FILE)
$(WINDOWS_INSTALLER_FILE): $(INVEST_BINARIES_DIR) \
							$(USERGUIDE_HTML_DIR) \
							$(USERGUIDE_PDF_FILE) \
							build/vcredist_x86.exe \
							$(SVN_DATA_REPO_PATH) 
	makensis \
		/DVERSION=$(VERSION) \
		/DBINDIR=$(INVEST_BINARIES_DIR) \
		/DARCHITECTURE=$(PYTHON_ARCH) \
		/DFORKNAME=$(FORKNAME) \
		/DDATA_LOCATION=$(DATA_BASE_URL) \
		installer\windows\invest_installer.nsi

mac_installer: $(MAC_DISK_IMAGE_FILE)
$(MAC_DISK_IMAGE_FILE): $(INVEST_BINARIES_DIR) $(USERGUIDE_HTML_DIR)
	./installer/darwin/build_dmg.sh "$(VERSION)" "$(INVEST_BINARIES_DIR)" "$(USERGUIDE_HTML_DIR)"

build/vcredist_x86.exe: build
	powershell.exe -Command "Start-BitsTransfer -Source https://download.microsoft.com/download/5/D/8/5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x86.exe -Destination build\vcredist_x86.exe"
