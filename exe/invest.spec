import sys
import os

# Global Variables
current_dir = os.path.join(os.getcwd(), os.path.dirname(sys.argv[1]))
app_dir = os.path.join(current_dir, 'apps')
scripts = os.listdir(app_dir)

# Analyze Scripts for Dependencies
kwargs = {
    'hookspath': [os.path.join(current_dir, 'hooks')],
    'excludes': None,
    'pathex': [os.getcwd()],
}
analysis_object_tuples = []
for script in scripts:
    fname, _ = os.path.splitext(script)
    fpath = os.path.join(app_dir, script)
    a = Analysis([fpath], **kwargs)
    analysis_object_tuples.append((a, fname, fname))
MERGE(*analysis_object_tuples)

# Compress pyc and pyo Files into ZlibArchive Objects
pyz_objects = []
for t in analysis_object_tuples:
    pyz = PYZ(t[0].pure)
    pyz_objects.append(pyz)

# Create Executable Files
exe_objects = []
for i in range(len(scripts)):
    fname, _ = os.path.splitext(scripts[i])
    exe = EXE(
        pyz_objects[i],
        analysis_object_tuples[i][0].scripts,
        name=fname,
        exclude_binaries=1,
        debug=False,
        strip=None,
        upx=False,
        console=False)
    exe_objects.append(exe)

# Collect Files into Distributable Folder/File
binaries = []
zipfiles = []
datas = []

for i in range(len(scripts)):
    binaries.append(analysis_object_tuples[i][0].binaries)
    zipfiles.append(analysis_object_tuples[i][0].zipfiles)
    datas.append(analysis_object_tuples[i][0].datas)

args = exe_objects + binaries + zipfiles + datas

dist = COLLECT(
        *args,
        name="invest_dist",
        strip=None,
        upx=True)
