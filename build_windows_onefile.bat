@echo off
setlocal

if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv was not found.
    exit /b 1
)

echo [1/3] Installing build dependency...
.venv\Scripts\python.exe -m pip install --upgrade pyinstaller
if errorlevel 1 exit /b 1

echo [2/3] Building Windows one-file executable...
.venv\Scripts\pyinstaller.exe --noconfirm --clean Graphik.onefile.spec
if errorlevel 1 exit /b 1

echo [3/3] Done.
echo Output file: dist\Graphik.exe
