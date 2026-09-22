@echo off
REM Double-click: start the viewer server and open the browser.
REM Run model needs PyTorch: use conda env scatteringNet, not system Python.
cd /d "%~dp0\..\.."

set "CONDA_PY=%USERPROFILE%\miniconda3\envs\scatteringNet\python.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%USERPROFILE%\anaconda3\envs\scatteringNet\python.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%LOCALAPPDATA%\miniconda3\envs\scatteringNet\python.exe"

if exist "%CONDA_PY%" (
  echo Using conda env scatteringNet:
  echo   %CONDA_PY%
  "%CONDA_PY%" -m scatteringnet.viewer.serve
) else (
  echo Conda python for env scatteringNet was not found in the usual folders.
  echo Trying: conda run -n scatteringNet
  conda run -n scatteringNet --no-capture-output python -m scatteringnet.viewer.serve
)

if errorlevel 1 (
  echo.
  echo Viewer failed. Close this window, then from Anaconda Prompt:
  echo   conda activate scatteringNet
  echo   pip install -e .
  echo   python -m scatteringnet.viewer.serve
  pause
)
