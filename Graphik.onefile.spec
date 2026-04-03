# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


project_root = Path.cwd()
datas = [
    *collect_data_files("streamlit"),
    *collect_data_files("altair"),
    (str(project_root / "app.py"), "."),
    (str(project_root / "Present"), "Present"),
    (str(project_root / "data"), "data"),
    (str(project_root / "src"), "src"),
    (str(project_root / ".streamlit" / "config.toml"), ".streamlit"),
]
for package_name in (
    "streamlit",
    "plotly",
    "matplotlib",
    "pandas",
    "numpy",
    "scipy",
    "openpyxl",
    "odfpy",
    "altair",
    "narwhals",
    "pyarrow",
):
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass
hiddenimports = [
    "app",
    "openpyxl",
    "odf.opendocument",
    "odf.table",
    "odf.text",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_mixed",
    "matplotlib.backends._backend_pdf_ps",
    "streamlit.runtime.scriptrunner.magic_funcs",
    *collect_submodules("streamlit.runtime.scriptrunner"),
]
excludes = [
    "pytest",
    "streamlit.testing",
    "IPython",
    "jupyter",
    "notebook",
    "tkinter",
]


a = Analysis(
    ["launcher.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Graphik",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
