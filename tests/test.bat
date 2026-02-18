@echo off
chcp 65001 >nul 2>&1
cd /d d:\TouchDiffusionMKII

if "%~1" neq "" (
    "d:\TouchDiffusionMKII\.venv\Scripts\python.exe" test_inference.py %*
    pause
    exit /b
)

echo.
echo Available configs:
echo.

setlocal enabledelayedexpansion
set i=0
for /r engines %%f in (config*.yaml) do (
    set "cfg[!i!]=%%f"
    echo   [!i!] %%f
    set /a i+=1
)

if %i%==0 (
    echo   No configs found. Run Build first.
    pause
    exit /b
)

echo.
set /p choice="Select config number: "
set "selected=!cfg[%choice%]!"

if "!selected!"=="" (
    echo Invalid choice.
    pause
    exit /b
)

echo.
echo Running: !selected!
echo.
"d:\TouchDiffusionMKII\.venv\Scripts\python.exe" test_inference.py "!selected!"
pause
