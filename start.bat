@echo off
REM Pose Perfect - double-click launcher for Windows.
REM Runs start.ps1 without needing to change the PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
