# -*- mode: python ; coding: utf-8 -*-
# Сборка: build.bat / build.sh / python build_project.py

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("pymodbus")
hiddenimports += [
    "config_utils",
    "paths",
    "camera_io",
    "roi_manager",
    "laser_detector",
    "laser_seam",
    "laser_ui",
    "vision_utils",
    "modbus_io",
    "tuning_controls",
    "video_recorder",
    "measure_log",
]

datas = []
for name in ("config.json", "NASTROYKI.txt"):
    if os.path.isfile(name):
        datas.append((name, "."))

try:
    datas += collect_data_files("cv2", include_py_files=False)
except Exception:
    pass

excludes = [
    "tkinter",
    "matplotlib",
    "PyQt6",
    "PyQt5",
    "PySide6",
    "scipy",
    "skimage",
    "av",
    "pandas",
    "PIL",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pipe_vision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pipe_vision",
)
