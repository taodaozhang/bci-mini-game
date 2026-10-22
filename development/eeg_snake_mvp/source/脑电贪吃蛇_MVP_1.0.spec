# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = [('C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_snake_mvp/source/assets', 'assets')]
binaries = []
datas += collect_data_files('tkinterdnd2')
binaries += collect_dynamic_libs('tkinterdnd2')


a = Analysis(
    ['C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_snake_mvp/source/snake_app.py'],
    pathex=['C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_snake_mvp/source'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['sklearn.svm._classes', 'sklearn.discriminant_analysis', 'sklearn.neural_network._multilayer_perceptron', 'sklearn.ensemble._forest', 'sklearn.pipeline', 'sklearn.preprocessing._data', 'sklearn.tree._classes'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pyedflib', 'pandas', 'IPython', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='脑电贪吃蛇_MVP_1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_snake_mvp/source/assets/app.ico'],
)
