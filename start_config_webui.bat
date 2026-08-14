@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

cd scripts
echo Starting MiniMind Config WebUI...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop
echo.
streamlit run config_webui.py --server.port 8501
pause
