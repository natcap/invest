SVN_DATA_REPO           = svn://scm.naturalcapitalproject.org/svn/invest-data
SVN_DATA_REPO_PATH      = data/invest-data
SVN_DATA_REPO_REV       = 171

SVN_TEST_DATA_REPO      = svn://scm.naturalcapitalproject.org/svn/invest-test-data
SVN_TEST_DATA_REPO_PATH = data/invest-test-data
SVN_TEST_DATA_REPO_REV  = 139

HG_UG_REPO              = https://bitbucket.org/natcap/invest.users-guide
HG_UG_REPO_PATH         = doc/users-guide
HG_UG_REPO_REV          = ae4705d8c9ad

env:
	python2 -m virtualenv --system-site-packages test_env

fetch:
	hg update -r $(HG_UG_REPO_REV) -R $(HG_UG_REPO_PATH) || \
		hg clone $(HG_UG_REPO) -u $(HG_UG_REPO_REV) $(HG_UG_REPO_PATH)

	svn update -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH) || \
		svn checkout $(SVN_DATA_REPO) -r $(SVN_DATA_REPO_REV) $(SVN_DATA_REPO_PATH)

	svn update -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH) || \
		svn checkout $(SVN_TEST_DATA_REPO) -r $(SVN_TEST_DATA_REPO_REV) $(SVN_TEST_DATA_REPO_PATH)


install:
	python setup.py bdist_wheel && wheel install natcap.invest --wheel-dir=dist/

binaries:
	rm -rf build/pyi-build dist/invest
	pyinstaller \
		--workpath build/pyi-build \
		--clean \
		--distpath dist \
		exe/invest.spec
	# try listing available modules as a basic test.
	./dist/invest/invest --list
	# try opening up a model
	./dist/invest/invest carbon

apidocs:
	python setup.py build_sphinx -a --source-dir doc/api-docs


userguide:
	cd doc/users-guide && $(MAKE) html latex && cd build/latex && $(MAKE) all-pdf


SUBDIRS := $(filter-out %.json,$(wildcard $(SVN_DATA_REPO_PATH)/*))
ZIPS := $(addsuffix .zip,$(subst $(SVN_DATA_REPO_PATH),dist/data,$(SUBDIRS)))

$(ZIPS):
	cd $(SVN_DATA_REPO_PATH) && \
		zip -r $(addprefix ../../,$@) $(subst dist/data/,,$(subst .zip,,$@))

.PHONY: sampledata
sampledata: $(ZIPS)


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
	-rm -r build dist natcap.invest.egg-info
