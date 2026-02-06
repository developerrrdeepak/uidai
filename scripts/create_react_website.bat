@echo off
echo ============================================
echo Creating React Website for UIDAI Project
echo ============================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)

echo [1/4] Creating React app...
npx create-react-app uidai-dashboard

echo.
echo [2/4] Installing dependencies...
cd uidai-dashboard
npm install axios chart.js react-chartjs-2

echo.
echo [3/4] React app created successfully!
echo.
echo [4/4] To start the development server:
echo    cd uidai-dashboard
echo    npm start
echo.
echo ============================================
echo React Website Setup Complete!
echo ============================================
pause
