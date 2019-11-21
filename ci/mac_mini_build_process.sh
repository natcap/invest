make env
source activate ./conda_env
make PYTHON=conda_env/bin/python python_packages
make PYTHON=conda_env/bin/python binaries

# cp conda_env/lib/libgeos_c.1.dylib dist/invest/libgeos_c.1.dylib
# cp conda_env/lib/libpng16.16.dylib dist/invest/libpng16.16.dylib
# cp conda_env/lib/libspatialindex_c.dylib dist/invest/lib/libspatialindex_c.dylib

#27049 ERROR: Can not find path ./libshiboken2.abi3.5.13.dylib (needed by /Users/jenkins/workspace/davemfish/invest/conda_env/lib/python3.7/site-packages/PySide2/QtWidgets.abi3.so)

# To test our build:
./dist/invest/invest list