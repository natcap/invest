
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
	# TODO: allow this to be able to access the natcap.invest package.  Egg?
	# This works: $ PYTHONPATH=dist/natcap.invest-3.4.2.post56+n3175d6cdcf27-py2.7-linux-x86_64.egg:$PYTHONPATH python -c "import natcap.invest"
	# Note that this works as expected within an activated virtual environment.
	python setup.py build_sphinx


userguide:
	cd doc/users-guide && make html latex && cd build/latex && make all-pdf


.PHONY: clean
clean:
	python setup.py clean
	rm -rf build dist exe/dist exe/build natcap.invest.egg-info release_env test_env
