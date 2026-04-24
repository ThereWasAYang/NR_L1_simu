@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup_windows.ps1" %*
if errorlevel 1 (
  echo.
  echo Windows 环境配置失败。
  exit /b 1
)

echo.
echo Windows 环境配置成功。
exit /b 0
