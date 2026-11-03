@echo off
setlocal
cd /d "%~dp0"
where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js is required. Install Node.js 20 or newer, then run this file again.
  pause
  exit /b 1
)
if not exist node_modules (
  call npm install
  if errorlevel 1 exit /b 1
)
call npm run dev
