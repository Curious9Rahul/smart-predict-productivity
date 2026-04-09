@echo off
REM Smart App Predictor - Flask Backend Startup Script

echo.
echo ====================================
echo  Smart App Predictor - Backend Server
echo ====================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please download and install Python: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Navigate to project root
cd /d "%~dp0"

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Make sure you're in the smart_app_predictor directory
    pause
    exit /b 1
)

echo Installing dependencies...
call pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ====================================
echo  Starting Flask Server...
echo ====================================
echo.
echo Server will be available at: http://localhost:5000
echo.
echo To connect from your phone:
echo 1. Open Command Prompt and run: ipconfig
echo 2. Find your IPv4 Address (e.g., 192.168.X.X)
echo 3. In app Settings, enter: http://192.168.X.X:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
