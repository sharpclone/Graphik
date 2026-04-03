# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path.cwd()
datas = [
    (str(project_root / "app.py"), "."),
    (str(project_root / "Present"), "Present"),
    (str(project_root / "data"), "data"),
    (str(project_root / "src"), "src"),
    (str(project_root / ".streamlit" / "config.toml"), ".streamlit"),
]
binaries = []
hiddenimports = []

for package_name in (
    "streamlit",
    "plotly",
    "kaleido",
    "numpy",
    "pandas",
    "scipy",
    "openpyxl",
    "odf",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports


a = Analysis(
    ["launcher.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Graphik",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Graphik",
)
