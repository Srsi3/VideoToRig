@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ======================================================
echo   Video2Rigify Outside‑Blender Test Runner (Windows)
echo ======================================================

REM --- Locate Blender config root ---
set "BF=%APPDATA%\Blender Foundation\Blender"
if not exist "%BF%" (
  echo [ERROR] Could not find Blender config at "%BF%"
  echo Open Blender once, or install the add-on deps first.
  goto :end
)

REM --- Find the latest Blender version folder using PowerShell ---
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-ChildItem -Path ^\"$env:APPDATA\Blender Foundation\Blender^\" -Directory ^| Sort-Object { [version]$_.Name } -Descending ^| Select-Object -First 1).FullName"`) do set "BLENDER_DIR=%%I"
if not defined BLENDER_DIR (
  echo [ERROR] Could not detect a Blender version directory under "%BF%"
  goto :end
)

set "CFG=%BLENDER_DIR%\config"
set "PY=%CFG%\video2rigify_env\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] External Python not found: "%PY%"
  echo In Blender: Edit ^> Preferences ^> Add-ons ^> Video 2 Rigify ^> Install Deps.
  goto :end
)

set "MMP=%CFG%\mmpose"
if not exist "%MMP%" (
  echo [ERROR] MMPose repo not found at "%MMP%"
  echo Run "Install Deps" in the add-on preferences to clone it.
  goto :end
)

set "MB=%CFG%\MotionBERT"
if not exist "%MB%" (
  echo [ERROR] MotionBERT repo not found at "%MB%"
  echo Run "Install Deps" in the add-on preferences to clone it.
  goto :end
)

REM --- Detect device (cuda:0 or cpu) ---
for /f "usebackq delims=" %%D in (`"%PY%" -c "import torch;print('cuda:0' if torch.cuda.is_available() else 'cpu')"`) do set "DEVICE=%%D"
if not defined DEVICE set "DEVICE=cpu"
echo Using device: %DEVICE%

REM --- Ensure test suite file is next to this .bat ---
set "SCRIPT_DIR=%~dp0"
set "TESTPY=%SCRIPT_DIR%v2r_test_suite.py"
if not exist "%TESTPY%" (
  echo [ERROR] v2r_test_suite.py not found next to this file:
  echo         "%TESTPY%"
  echo Save the suite in the same folder as this .bat and try again.
  goto :end
)

REM --- Optional: let user pick a short video (can cancel to skip E2E) ---
set "VIDEO="
for /f "usebackq delims=" %%V in (`powershell -STA -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms ^| Out-Null; $dlg = New-Object System.Windows.Forms.OpenFileDialog; $dlg.Title='Select a short test video (optional)'; $dlg.Filter='Video files|*.mp4;*.mov;*.avi;*.mkv|All files|*.*'; if($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){$dlg.FileName}"`) do set "VIDEO=%%V"
if defined VIDEO echo Using video: "%VIDEO%"

set "WORKDIR=%TEMP%\v2r_tests"
if not exist "%WORKDIR%" mkdir "%WORKDIR%" >nul 2>&1

echo.
echo ======================================================
echo   Running tests... (logs: %WORKDIR%\v2r_tests.log)
echo ======================================================
echo.

if defined VIDEO (
  "%PY%" "%TESTPY%" --python "%PY%" --mmpose "%MMP%" --motionbert "%MB%" --device %DEVICE% --workdir "%WORKDIR%" --quick --video "%VIDEO%"
) else (
  "%PY%" "%TESTPY%" --python "%PY%" --mmpose "%MMP%" --motionbert "%MB%" --device %DEVICE% --workdir "%WORKDIR%" --quick
)

set "LOGFILE=%WORKDIR%\v2r_tests.log"
if exist "%LOGFILE%" start "" "%LOGFILE%"
echo.
echo Done. Artifacts and logs in: %WORKDIR%
echo (This window can be closed.)

:pause_or_end
pause >nul
:end
exit /b
