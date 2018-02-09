SVN_DATA_REPO           = svn://scm.naturalcapitalproject.org/svn/invest-sample-data
SVN_DATA_REPO_PATH      = data/invest-data
SVN_DATA_REPO_REV       = 171

SVN_TEST_DATA_REPO      = svn://scm.naturalcapitalproject.org/svn/invest-test-data
SVN_TEST_DATA_REPO_PATH = data/invest-test-data
SVN_TEST_DATA_REPO_REV  = 139

HG_UG_REPO              = https://bitbucket.org/natcap/invest.users-guide
HG_UG_REPO_PATH         = doc/users-guide
HG_UG_REPO_REV          = ae4705d8c9ad

VERSION = $(shell python2 setup.py --version)
PYTHON_ARCH = $(shell python2 -c "import struct; print(8*struct.calcsize('P'))")
DEST_VERSION = $(shell hg log -r. --template="{ifeq(latesttagdistance,'0',latesttag,'develop')}")

# These are intended to be overridden by a jenkins build.
# When building a fork, we might set FORKNAME to <username> and DATA_BASE_URL
# to wherever the datasets will be available based on the forkname and where
# we're storing the datasets.
# These defaults assume that we're storing datasets for an InVEST release.
# DEST_VERSION is 'develop' unless we're at a tag, in which case it's the tag.
FORKNAME = ""
DATA_BASE_URL = "http://data.naturalcapitalproject.org/invest-data/$(DEST_VERSION)"


env:
	python2 -m virtualenv --system-site-packages env
	bash -c "source env/bin/activate && pip install -r requirements.txt -r requirements-dev.txt"
	bash -c "source env/bin/activate && pip install -I nose mock"
	bash -c "source env/bin/activate && $(MAKE) install"


build:
	mkdir build

data:
	mkdir data

dist:
	mkdir dist

dist/data: dist
	mkdir dist/data


.PHONY: fetch
fetch: data
	hg update -r $(HG_UG_REPO_REV) -R $(HG_UG_REPO_PATH) || \
		hg clone $(HG_UG_REPO) -u $(HG_UG_REPO_REV) $(HG_UG_REPO_PATH)

	svn update -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH) || \
		svn checkout $(SVN_DATA_REPO) -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH)

	svn update -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH) || \
		svn checkout $(SVN_TEST_DATA_REPO) -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH)

.PHONY: install
install: dist
	python2 setup.py bdist_wheel && wheel install natcap.invest --wheel-dir=dist

.PHONY: binaries
binaries: dist build
	-rm -r build/pyi-build dist/invest
	pyinstaller \
		--workpath build/pyi-build \
		--clean \
		--distpath dist \
		exe/invest.spec

.PHONY: apidocs
apidocs:
	python setup.py build_sphinx -a --source-dir doc/api-docs

.PHONY: userguide
userguide:
	cd doc/users-guide && $(MAKE) html latex && cd build/latex && $(MAKE) all-pdf


SUBDIRS := $(filter-out Base_data, $(filter-out %.json,$(wildcard $(SVN_DATA_REPO_PATH)/*)))
NORMALZIPS := $(addsuffix .zip,$(subst $(SVN_DATA_REPO_PATH),dist/data,$(SUBDIRS)))
$(NORMALZIPS): dist/data
	cd $(SVN_DATA_REPO_PATH) && \
		zip -r $(addprefix ../../,$@) $(subst dist/data/,,$(subst .zip,,$@))

BASEDATADIRS := $(wildcard $(SVN_DATA_REPO_PATH)/Base_Data/*)
BASEDATAZIPS := $(addsuffix .zip,$(subst $(SVN_DATA_REPO_PATH)/Base_Data,dist/data,$(BASEDATADIRS)))
$(BASEDATAZIPS): dist/data
	cd $(SVN_DATA_REPO_PATH) && \
		zip -r $(addprefix ../../,$@) $(subst dist/data/,Base_Data/,$(subst .zip,,$@))


build/vcredist_x86.exe: build
	powershell.exe -command "Start-BitsTransfer -Source https://download.microsoft.com/download/5/D/8/5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x86.exe -Destination build\vcredist_x86.exe"


dist/InVEST_%_Setup.exe: dist build binaries userguide build/vcredist_x86.exe
	makensis \
		/O=build\nsis.log \
		/DVERSION=$(VERSION) \
		/DBINDIR=dist\invest \
		/DARCHITECTURE=$(PYTHON_ARCH) \
		/DFORKNAME=$(FORKNAME) \
		/DDATA_LOCATION=$(DATA_BASE_URL)

dist/InVEST%.dmg: binaries userguide
	cd installer/darwin && bash -c "./build_dmg.sh"
	cp installer/darwin/InVEST*.dmg dist

.PHONY: windows_installer
windows_installer:
	$(MAKE) dist$(/)InVEST_$(VERSION)_Setup.exe


.PHONY: sampledata
sampledata: $(NORMALZIPS) $(BASEDATAZIPS)


.PHONY: test
test:
	nosetests -vsP \
		--with-coverage \
		--cover-package=natcap.invest \
		--cover-erase \
		--with-xunit \
		--cover-tests \
		--cover-html \
		--logging-filter=None \
		tests/*.py ui_tests/*.py

.PHONY: clean
clean:
	python setup.py clean
	-rm -r build dist natcap.invest.egg-info installer/darwin/*.dmg
