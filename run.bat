REM This bat is coded in GBK to support Chinese paths
@echo off
title GitHub Project Analyzer

REM Determine repo root
set REPO_ROOT=%~dp0

REM Bootstrap portable Python if missing
if not exist "%REPO_ROOT%python\python_runtime\python.exe" (
	echo [INFO] Setting up embedded Python runtime...
	powershell -ExecutionPolicy Bypass -File "%REPO_ROOT%python\scripts\setup_python_runtime.ps1"
)

REM Open frontend
start http://localhost:5501

REM Start Node backend server
node "%REPO_ROOT%backend\server.js"