@echo off
REM Double-click: start the Gradio occupancy demo (not the Three.js viewer).
REM Needs PyTorch + gradio + plotly in conda env scatteringNet.
cd /d "%~dp0"

set "CONDA_PY=%USERPROFILE%\miniconda3\envs\scatteringNet\python.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%USERPROFILE%\anaconda3\envs\scatteringNet\python.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%LOCALAPPDATA%\miniconda3\envs\scatteringNet\python.exe"

if exist "%CONDA_PY%" (
  echo Using conda env scatteringNet:
  echo   %CONDA_PY%
  "%CONDA_PY%" -u app.py
) else (
  echo Conda python for env scatteringNet was not found in the usual folders.
  echo Trying: conda run -n scatteringNet
  conda run -n scatteringNet --no-capture-output python -u app.py
)

if errorlevel 1 (
  echo.
  echo Gradio demo failed. Close this window, then from Anaconda Prompt:
  echo   conda activate scatteringNet
  echo   pip install -r "%~dp0requirements.txt"
  echo   python "%~dp0app.py"
  pause
)
