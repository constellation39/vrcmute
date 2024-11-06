# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all required dependencies from pyproject.toml
hiddenimports = []
for package in [
    'numpy',
    'sounddevice',
    'dashscope',
    'scipy',
    'librosa',
    'python-osc',
    'discover',
    'sherpa-onnx',
]:
    hiddenimports.extend(collect_submodules(package))

# Collect additional data files
datas = []
for package in ['sherpa-onnx', 'librosa']:
    datas.extend(collect_data_files(package))

a = Analysis(
    ['src/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out any duplicate files
seen = set()
clean_datas = []
for data in a.datas:
    if data[0] not in seen:
        clean_datas.append(data)
        seen.add(data[0])
a.datas = clean_datas

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='vrcmute',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None  # You can add an icon file path here if needed
)