@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

cd scripts
echo Starting Instinct Chat WebUI...
echo Open http://localhost:8502 in your browser
echo Press Ctrl+C to stop
echo.
streamlit run web_demo.py --server.port 8502
pause
