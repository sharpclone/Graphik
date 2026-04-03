@echo off
setlocal

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

echo [2/3] Building Windows bundle...
.venv\Scripts\pyinstaller.exe --noconfirm --clean Graphik.spec
if errorlevel 1 exit /b 1

echo [3/3] Done.
echo Output folder: dist\Graphik
echo Start file: dist\Graphik\Graphik.exe
