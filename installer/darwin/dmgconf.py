import os
CWD = os.path.join('installer', 'darwin')

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return str(int(total_size/1024.) + 1024*50) + 'K'

size = get_size(defines['investdir'])
print('Volume size: %s' % size)
print('Packaging dirname %s' % defines['investdir'])
_invest_dirname = os.path.basename(defines['investdir'])

badge_icon = os.path.join(CWD, 'invest.icns')
symlinks = {'Applications': '/Applications'}
files = [defines['investdir']]

icon_locations = {
    _invest_dirname: (220, 290),
    'Applications': (670, 290)
}
icon_size = 100
text_size = 12

# Window Settings
window_rect = ((100, 100), (900, 660))
background = os.path.join(CWD, 'background.png')
#background = 'builtin-arrow'
default_view = 'icon-view'


