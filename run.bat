@echo off
title GitHub Project Analyzer

REM Determine repo root
set REPO_ROOT=%~dp0

REM Bootstrap portable Python if missing
if not exist "%REPO_ROOT%python_runtime\python.exe" (
	echo [INFO] Setting up embedded Python runtime...
	powershell -ExecutionPolicy Bypass -File "%REPO_ROOT%scripts\setup_python_runtime.ps1"
)

REM Wire PYTHON_PATH for backend
set PYTHON_PATH=%REPO_ROOT%python_runtime\python.exe

REM Open frontend
start http://localhost:5501

REM Start Node backend
node "%REPO_ROOT%ºó¶Ëbackend\server.js"