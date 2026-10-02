@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
set PYTHONUTF8=1
echo Starting Instinct Eval WebUI at http://localhost:8503
python -m streamlit run scripts/eval_webui.py --server.address 127.0.0.1 --server.port 8503
pause
