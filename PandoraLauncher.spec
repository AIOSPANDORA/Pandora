# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Pandora Launcher

This spec file is used to build standalone executables for:
- Windows (PandoraLauncher.exe)
- Linux (PandoraLauncher)
- macOS (PandoraLauncher.app)

Usage:
    pyinstaller PandoraLauncher.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Determine the base directory
if hasattr(sys, '_MEIPASS'):
    base_dir = Path(sys._MEIPASS)
else:
    base_dir = Path.cwd()

# Data files to include
datas = [
    ('quantum_profiles', 'quantum_profiles'),
    ('README.md', '.'),
    ('IMPLEMENTATION_SUMMARY.md', '.'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'scipy',
    'scipy.linalg',
    'scipy.sparse',
    'scipy.special',
    'quantum_profiles',
    'quantum_profiles.hamiltonian',
    'quantum_profiles.alternative',
    'quantum_profiles.castle',
    'quantum_profiles.hive',
    'quantum_profiles.empire',
    'quantum_profiles.omega',
    'quantum_profiles.ml_quantum_addon',
    'quantum_profiles.profile_manager',
    'quantum_virtual_processor',
    'test_quantum_hamiltonian',
    'demo_quantum_hamiltonian',
    'example_quantum_security_integration',
]

a = Analysis(
    ['PandoraLauncher.py'],
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PandoraLauncher',
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
    icon=None,  # Add icon file here if available: icon='pandora.ico'
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='PandoraLauncher.app',
        icon=None,  # Add icon file here if available
        bundle_identifier='com.pandora.launcher',
    )
