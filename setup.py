import os
import sys
from cx_Freeze import setup, Executable

PYTHON_INSTALL_DIR = os.path.dirname(sys.executable)

build_exe_options = {
    'packages': ['numpy', 'shapely', 'rasterio'],
    'includes': ['scipy.spatial.transform._rotation_groups'],
    'include_files': []
}

base = None
if sys.platform == 'win32':
    base = 'Win32GUI'
    # DLLS_FOLDER = os.path.join(PYTHON_INSTALL_DIR, 'Library', 'bin')
    # dependencies = ['mkl_core.dll', 'mkl_def.dll', 'mkl_intel_thread.dll', 'geos_c.dll']
    # for dependency in dependencies:
    #     build_exe_options['include_files'].append(os.path.join(DLLS_FOLDER, dependency))

setup(
    name='CoverTest',
    version='0.2',
    description='cx_Freeze exe',
    options={'build_exe': build_exe_options},
    executables=[Executable('main.py', base=base, target_name='CoverTest')],
)
