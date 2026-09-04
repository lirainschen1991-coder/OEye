# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


ROOT = Path.cwd()
PACKAGING_DIR = ROOT / "packaging"
PROTECTED_DIR = ROOT / "build" / "oeye_protected_sources"


datas = [
    (str(PACKAGING_DIR / "oeye_dependency_entry.py"), "."),
    (str(PACKAGING_DIR / "oeye_app_entry.py"), "."),
    (str(PACKAGING_DIR / "oeye_launcher.py"), "."),
    (str(PACKAGING_DIR / "dependency_manager.py"), "."),
    (str(ROOT / "requirements-standard.txt"), "."),
    (str(ROOT / "requirements-runtime.txt"), "."),
]

if PROTECTED_DIR.exists():
    datas.append((str(PROTECTED_DIR), "protected_sources"))

if (ROOT / "sample_data").exists():
    datas.append((str(ROOT / "sample_data"), "sample_data"))

if (ROOT / "OEye.jpeg").exists():
    datas.append((str(ROOT / "OEye.jpeg"), "."))

datas += collect_data_files("streamlit")
datas += collect_data_files("webview")
datas += copy_metadata("streamlit")
datas += copy_metadata("pywebview")
datas += copy_metadata("packaging")

hiddenimports = []
hiddenimports += [
    "streamlit",
    "streamlit.web",
    "streamlit.web.bootstrap",
    "streamlit.web.cli",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "packaging",
    "packaging.requirements",
    "packaging.specifiers",
    "packaging.markers",
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "clr",
    "clr_loader",
]

excludes = [
    "tensorflow",
    "torch",
    "catboost",
    "stable_baselines3",
    "gymnasium",
    "ray",
    "ray.rllib",
    "polars",
    "_polars_runtime_32",
    "_polars_runtime_64",
    "duckdb",
    "scipy",
    "sklearn",
    "skimage",
    "xarray",
    "numba",
    "llvmlite",
    "matplotlib",
    "plotly",
    "altair.vegalite.v5.schema.core",
    "pytest",
    "IPython",
    "sphinx",
    "statsmodels",
    "netCDF4",
    "h5py",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "tkinter",
]

a = Analysis(
    [str(PACKAGING_DIR / "oeye_launcher.py")],
    pathex=[str(ROOT), str(PACKAGING_DIR)],
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
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OEye",
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
    icon=str(ROOT / "packaging" / "OEye.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OEye",
)
