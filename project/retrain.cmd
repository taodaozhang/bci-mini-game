@echo off
setlocal
cd /d "%~dp0"
set "EEG_PY=C:\Users\Lenovo\Documents\Codex\2026-10-15\https-bnci-horizon-2020-eu-database\work\.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "EEG_PY=%~dp0.venv\Scripts\python.exe"
"%EEG_PY%" "train_left_right_eeg.py" %*
pause
