

install:
	python setup.py bdist_wheel && wheel install natcap.invest --wheel-dir=dist/

binaries:
	pyinstaller \
		--workpath pyi-dist \
		--clean \
		--onedir \
		--name invest \
		--additional-hooks-dir exe2/hooks \
		--runtime-hook exe2/hooks/rthook.py \
		--console \
		src/natcap/invest/cli.py
	
	# try listing available modules as a basic test.
	./pyi-dist/invest/invest --list

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
	rm -f build dist exe/dist exe/build natcap.invest.egg-info release_env test_env
