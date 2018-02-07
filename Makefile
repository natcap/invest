
env:
	python2 -m virtualenv --system-site-packages test_env


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
	cd doc/users-guide && make html latex && cd build/latex && make all-pdf

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
	rm -rf build dist exe/dist exe/build natcap.invest.egg-info release_env test_env
