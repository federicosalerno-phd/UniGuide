@echo off
cd /d "%~dp0"
REM Launch UniGuide. The interpreter MUST have PyQt6 + PyQt6-WebEngine (and, for the
REM welded-guide export, numpy/trimesh/manifold3d). Those live in the arm-controller
REM virtualenv below. Override the interpreter with the UNIGUIDE_PYTHON env var if needed.
REM NOTE: a plain "python uniguide_app.py" uses the SYSTEM Python, which does NOT have
REM PyQt6 -> "ModuleNotFoundError: No module named 'PyQt6'". Always launch via this .bat.
if not defined UNIGUIDE_PYTHON set "UNIGUIDE_PYTHON=%USERPROFILE%\.virtualenvs\arm-controller-I0EXsfi0\Scripts\python.exe"
if exist "%UNIGUIDE_PYTHON%" (
    "%UNIGUIDE_PYTHON%" uniguide_app.py
) else (
    echo.
    echo Could not find the UniGuide Python ^(the one with PyQt6^) at:
    echo    %UNIGUIDE_PYTHON%
    echo.
    echo Set the UNIGUIDE_PYTHON env var to a Python that has PyQt6 + PyQt6-WebEngine,
    echo then run this launch.bat again.
    pause
    exit /b 1
)
pause
