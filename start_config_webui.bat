@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem Quoted set keeps the value free of trailing spaces (Windows cmd).
rem Note: current Windows PyTorch builds ignore expandable_segments and warn once
rem at CUDA init ("expandable_segments not supported on this platform").
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

cd scripts
echo Starting Instinct Config WebUI...
echo Open http://localhost:8500 in your browser
echo Press Ctrl+C to stop
echo.
streamlit run config_webui.py --server.port 8500
pause
