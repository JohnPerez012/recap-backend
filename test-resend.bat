@echo off
cd /d "%~dp0"
echo.
echo ==================================================
echo   Testing Resend API Integration
echo ==================================================
echo.
node test-resend.js
echo.
pause
