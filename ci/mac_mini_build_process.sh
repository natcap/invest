PYTHON_ENV=conda_env
# conda must be installed and on the path for make env
make ENV=$PYTHON_ENV env
source activate ./$PYTHON_ENV
make PYTHON=$PYTHON_ENV/bin/python python_packages
make PYTHON=$PYTHON_ENV/bin/python binaries

#27049 ERROR: Can not find path ./libshiboken2.abi3.5.13.dylib (needed by /Users/jenkins/workspace/davemfish/invest/conda_env/lib/python3.7/site-packages/PySide2/QtWidgets.abi3.so)

# To test our build:
# ./dist/invest/invest list