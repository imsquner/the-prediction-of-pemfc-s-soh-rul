# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PEMFC_Integrated_Tool.py'],
    pathex=[],
    binaries=[],
    datas=[('PEMFC_Integrated_Tool.py', '.'), ('train.py', '.'), ('data_processing.py', '.'), ('pemfc_catboost_analysis.py', '.'), ('data', 'data'), ('processed_results', 'processed_results'), ('visualization', 'visualization'), ('catboost_results', 'catboost_results'), ('train_results_paper', 'train_results_paper'), ('train_results', 'train_results')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'tqdm'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PEMFC_Integrated_Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PEMFC_Integrated_Tool',
)
