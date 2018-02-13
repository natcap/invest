SVN_DATA_REPO           = svn://scm.naturalcapitalproject.org/svn/invest-sample-data
SVN_DATA_REPO_PATH      = data/invest-data
SVN_DATA_REPO_REV       = 171

SVN_TEST_DATA_REPO      = svn://scm.naturalcapitalproject.org/svn/invest-test-data
SVN_TEST_DATA_REPO_PATH = data/invest-test-data
SVN_TEST_DATA_REPO_REV  = 139

HG_UG_REPO              = https://bitbucket.org/natcap/invest.users-guide
HG_UG_REPO_PATH         = doc/users-guide
HG_UG_REPO_REV          = ae4705d8c9ad

ENV = env
PYTHON = python2
NOSETESTS = $(PYTHON) -m nose -vsP --with-coverage --cover-package=natcap.invest --cover-erase --with-xunit --cover-tests --cover-html --logging-filter=None
VERSION = $(shell $(PYTHON) setup.py --version)
PYTHON_ARCH = $(shell $(PYTHON) -c "import struct; print(8*struct.calcsize('P'))")
DEST_VERSION = $(shell hg log -r. --template="{ifeq(latesttagdistance,'0',latesttag,'develop')}")
DIRS = build data dist dist/data
REQUIRED_PROGRAMS = make zip pandoc $(PYTHON) svn hg pdflatex pip makensis

ifeq ($(OS),Windows_NT)
	NULL = NUL
	PROGRAM_CHECK_SCRIPT = .\scripts\check_required_programs.bat
	ENV_ACTIVATE = .\$(ENV)\Scripts\activate
else
	NULL = /dev/null
	PROGRAM_CHECK_SCRIPT = ./scripts/check_required_programs.sh
	ENV_ACTIVATE = source $(ENV)/bin/activate
	SHELL := /bin/bash
endif


# These are intended to be overridden by a jenkins build.
# When building a fork, we might set FORKNAME to <username> and DATA_BASE_URL
# to wherever the datasets will be available based on the forkname and where
# we're storing the datasets.
# These defaults assume that we're storing datasets for an InVEST release.
# DEST_VERSION is 'develop' unless we're at a tag, in which case it's the tag.
FORKNAME = ""
DATA_BASE_URL = "http://data.naturalcapitalproject.org/invest-data/$(DEST_VERSION)"

.PHONY: fetch install binaries apidocs userguide windows_installer mac_installer sampledata test test_ui clean help check $(HG_UG_REPO_PATH) $(SVN_DATA_REPO_PATH) $(SVN_TEST_DATA_REPO_PATH)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  check             to verify all needed programs and packages are installed"
	@echo "  env               to create a virtualenv with packages from requirements.txt, requirements-dev.txt"
	@echo "  fetch             to clone all managed repositories"
	@echo "  install           to build and install a wheel of natcap.invest into the active python installation"
	@echo "  binaries          to build pyinstaller binaries"
	@echo "  apidocs           to build HTML API documentation"
	@echo "  userguide         to build HTML and PDF versions of the user's guide"
	@echo "  windows_installer to build an NSIS installer for distribution"
	@echo "  mac_installer     to build a disk image for distribution"
	@echo "  sampledata        to build sample data zipfiles"
	@echo "  test              to run nosetests on the tests directory"
	@echo "  test_ui           to run nosetests on the ui_tests directory"
	@echo "  clean             to remove temporary directories (but not dist/)"
	@echo "  help              to print this help and exit"

env:
	$(PYTHON) -m virtualenv --system-site-packages $(ENV)
	$(ENV_ACTIVATE) && pip install -r requirements.txt -r requirements-dev.txt
	$(ENV_ACTIVATE) && $(MAKE) install

$(DIRS):
	mkdir -p $@

$(HG_UG_REPO_PATH): data
	hg update -r $(HG_UG_REPO_REV) -R $(HG_UG_REPO_PATH) || \
		hg clone $(HG_UG_REPO) -u $(HG_UG_REPO_REV) $(HG_UG_REPO_PATH)

$(SVN_DATA_REPO_PATH): data
	svn checkout $(SVN_DATA_REPO) -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH)

$(SVN_TEST_DATA_REPO_PATH): data
	svn checkout $(SVN_TEST_DATA_REPO) -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH)

fetch: $(HG_UG_REPO_PATH) $(SVN_DATA_REPO_PATH) $(SVN_TEST_DATA_REPO_PATH)


install: dist
	$(PYTHON) setup.py bdist_wheel && \
		pip install --use-wheel --find-links=dist natcap.invest 

dist/invest: dist build
	-rm -r build/pyi-build dist/invest
	pyinstaller \
		--workpath build/pyi-build \
		--clean \
		--distpath dist \
		exe/invest.spec

binaries: dist/invest

dist/apidocs:
	$(PYTHON) setup.py build_sphinx -a --source-dir doc/api-docs
	cp -r build/sphinx/html dist/apidocs

apidocs: dist/apidocs

dist/%.pdf: $(HG_UG_REPO_PATH)
	cd doc/users-guide && $(MAKE) BUILDDIR=../../build/userguide latex
	cd build/userguide/latex && make all-pdf
	cp build/userguide/latex/InVEST*.pdf dist

dist/userguide: $(HG_UG_REPO_PATH)
	cd doc/users-guide && $(MAKE) BUILDDIR=../../build/userguide html
	cp -r build/userguide/html dist/userguide

userguide: dist/userguide dist/%.pdf

SUBDIRS := $(filter-out Base_data, $(filter-out %.json,$(wildcard $(SVN_DATA_REPO_PATH)/*)))
NORMALZIPS := $(addsuffix .zip,$(subst $(SVN_DATA_REPO_PATH),dist/data,$(SUBDIRS)))
$(NORMALZIPS): $(SVN_DATA_REPO_PATH) dist/data
	cd $(SVN_DATA_REPO_PATH) && \
		zip -r $(addprefix ../../,$@) $(subst dist/data/,,$(subst .zip,,$@))

BASEDATADIRS := $(wildcard $(SVN_DATA_REPO_PATH)/Base_Data/*)
BASEDATAZIPS := $(addsuffix .zip,$(subst $(SVN_DATA_REPO_PATH)/Base_Data,dist/data,$(BASEDATADIRS)))
$(BASEDATAZIPS): $(SVN_DATA_REPO_PATH) dist/data
	cd $(SVN_DATA_REPO_PATH) && \
		zip -r $(addprefix ../../,$@) $(subst dist/data/,Base_Data/,$(subst .zip,,$@))

sampledata: $(NORMALZIPS) $(BASEDATAZIPS)

build/vcredist_x86.exe: build
	powershell.exe -command "Start-BitsTransfer -Source https://download.microsoft.com/download/5/D/8/5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x86.exe -Destination build\vcredist_x86.exe"


windows_installer: dist dist/invest dist/userguide build/vcredist_x86.exe
	makensis \
		/O=build\nsis.log \
		/DVERSION=$(VERSION) \
		/DBINDIR=dist\invest \
		/DARCHITECTURE=$(PYTHON_ARCH) \
		/DFORKNAME=$(FORKNAME) \
		/DDATA_LOCATION=$(DATA_BASE_URL)

mac_installer: dist/invest dist/userguide
	./installer/darwin/build_dmg.sh "$(VERSION)" "dist/invest" "dist/userguide"

test: $(SVN_DATA_REPO_PATH) $(SVN_TEST_DATA_REPO_PATH)
	$(NOSETESTS) tests

test_ui:
	$(NOSETESTS) ui_tests

clean:
	$(PYTHON) setup.py clean
	-rm -r build natcap.invest.egg-info

check:
	@echo Checking required applications
	@$(PROGRAM_CHECK_SCRIPT) $(REQUIRED_PROGRAMS)
	@echo ----------------------------
	@echo Checking python packages
	@pip freeze --all -r requirements.txt -r requirements-dev.txt > $(NULL)
