@echo off
REM Build the standalone Windows app locally with PyInstaller.
REM Output: dist\UniGuide\UniGuide.exe  (a self-contained one-folder app).
REM Zip dist\UniGuide and share it, or just let GitHub Actions build the release.
cd /d "%~dp0"
if not defined UNIGUIDE_PYTHON set "UNIGUIDE_PYTHON=%USERPROFILE%\.virtualenvs\arm-controller-I0EXsfi0\Scripts\python.exe"
if not exist "%UNIGUIDE_PYTHON%" (
    echo Could not find the build Python at %UNIGUIDE_PYTHON%
    echo Set UNIGUIDE_PYTHON to a Python that has the requirements + pyinstaller installed.
    pause & exit /b 1
)
echo Building UniGuide.exe with %UNIGUIDE_PYTHON% ...
"%UNIGUIDE_PYTHON%" -m PyInstaller UniGuide.spec --noconfirm
echo.
echo Done. The app is in  dist\UniGuide\UniGuide.exe
pause
