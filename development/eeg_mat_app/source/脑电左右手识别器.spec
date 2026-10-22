# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_mat_app/source/eeg_app.py'],
    pathex=['C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_mat_app/source'],
    binaries=[],
    datas=[('C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_mat_app/source/assets', 'assets')],
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
    name='脑电左右手识别器',
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
    icon=['C:/Users/Lenovo/Documents/Codex/2026-10-15/https-bnci-horizon-2020-eu-database/outputs/eeg_mat_app/source/assets/app.ico'],
)
