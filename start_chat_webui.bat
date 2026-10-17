@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

rem Optional: start_chat_webui.bat --https  -> serve over TLS with a
rem self-signed certificate generated at repo-root\certs\chat_webui.crt/.key
set "HTTPS_ARGS="
if /i "%~1"=="--https" (
    python scripts\https_cert.py --out-dir "%~dp0certs" --name chat_webui
    if errorlevel 1 (
        echo [ERROR] Failed to prepare the HTTPS certificate, falling back is disabled.
        pause
        exit /b 1
    )
    set "HTTPS_ARGS=--server.sslCertFile "%~dp0certs\chat_webui.crt" --server.sslKeyFile "%~dp0certs\chat_webui.key""
)

cd scripts
echo Starting Instinct Chat WebUI...
if defined HTTPS_ARGS (
    echo Open https://localhost:8502 in your browser - self-signed cert, accept the browser warning to continue
) else (
    echo Open http://localhost:8502 in your browser
)
echo Press Ctrl+C to stop
echo.
streamlit run web_demo.py --server.port 8502 %HTTPS_ARGS%
pause
