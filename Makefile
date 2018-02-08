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


env:
	python2 -m virtualenv --system-site-packages env
	bash -c "source env/bin/activate && pip install -r requirements.txt -r requirements-dev.txt"
	bash -c "source env/bin/activate && pip install -I nose mock"
	bash -c "source env/bin/activate && $(MAKE) install"


data:
	mkdir data


.PHONY: fetch
fetch: data
	hg update -r $(HG_UG_REPO_REV) -R $(HG_UG_REPO_PATH) || \
		hg clone $(HG_UG_REPO) -u $(HG_UG_REPO_REV) $(HG_UG_REPO_PATH)

	svn update -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH) || \
		svn checkout $(SVN_DATA_REPO) -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH)

	svn update -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH) || \
		svn checkout $(SVN_TEST_DATA_REPO) -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH)

.PHONY: install
install:
	python2 setup.py bdist_wheel && wheel install natcap.invest --wheel-dir=dist/

.PHONY: binaries
binaries:
	rm -rf build/pyi-build dist/invest
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


dist/data:
	mkdir -p dist/data


dist/InVEST_%_Setup.exe:
	makensis

dist/InVEST%.dmg: binaries
	cd installer/darwin && bash -c "./build_dmg.sh"
	cp installer/darwin/InVEST*.dmg dist


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
