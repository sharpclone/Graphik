@echo off
setlocal

set DIST_DIR=dist_portable

if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv was not found.
    echo Create it first with:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

echo [1/3] Installing build dependency...
.venv\Scripts\python.exe -m pip install --upgrade pyinstaller
if errorlevel 1 exit /b 1

echo [2/3] Building Windows portable bundle...
.venv\Scripts\pyinstaller.exe --noconfirm --clean --distpath %DIST_DIR% Graphik.spec
if errorlevel 1 exit /b 1

echo [3/3] Done.
echo Output folder: %DIST_DIR%\Graphik
echo Start file: %DIST_DIR%\Graphik\Graphik.exe
echo Required support files: %DIST_DIR%\Graphik\_include
