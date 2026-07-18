# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build spec for UniGuide.
#   Build:  pyinstaller UniGuide.spec            (run from this folder, in the venv)
#   Output: dist/UniGuide/UniGuide.exe  (a self-contained one-folder app — zip it to distribute)
# The one-FOLDER layout is used on purpose: QtWebEngine (embedded Chromium) is far more
# reliable one-folder than one-file, and it starts instantly (no per-launch unpack).
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Bundled read-only assets (found at runtime via _res_dir() -> sys._MEIPASS).
datas = [
    ('ui.html', '.'),
    ('guides.json', '.'),
    ('libs', 'libs'),
    ('modules', 'modules'),
    ('input', 'input'),
]
datas += collect_data_files('trimesh')          # trimesh ships small resource files

hiddenimports = ['manifold3d', 'networkx']
hiddenimports += collect_submodules('scipy')    # scipy pulls submodules lazily

# CONSOLE: False for the distributable (no console window behind the app). A startup
# crash is still captured — the app writes it to %LOCALAPPDATA%\UniGuide\startup_error.log
# and shows a message box. Set to True locally if you want live stdout while debugging.
CONSOLE = False

a = Analysis(
    ['uniguide_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PyQt5'],   # not used → keep the bundle smaller
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UniGuide',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='UniGuide',
)
