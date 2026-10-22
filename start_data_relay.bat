@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

rem Pick the newest build without overwriting an executable held by a running service.
set "INSTINCT_RELAY_ROOT=%~dp0"
set "RELAY_EXE="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "@('data_relay/bin/data_relay_manager.exe','data_relay/bin/data_relay.exe','data_relay/target/release/data_relay.exe') | ForEach-Object { Get-Item -LiteralPath (Join-Path $env:INSTINCT_RELAY_ROOT $_) -ErrorAction SilentlyContinue } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"`) do set "RELAY_EXE=%%I"

if not exist "%RELAY_EXE%" (
    echo [ERROR] Missing relay executable. Run cargo build --release in data_relay.
    pause
    exit /b 1
)

cd /d "%~dp0data_relay"
if errorlevel 1 (
    echo [ERROR] Cannot open the data_relay directory.
    pause
    exit /b 1
)

if not exist "config.toml" (
    "%RELAY_EXE%" --config config.toml serve
    if errorlevel 1 (
        echo [ERROR] Failed to generate config.toml.
        pause
        exit /b 1
    )
    echo.
    echo Edit data_relay\config.toml before starting again.
    echo Set server.api_token and AutoDL credentials; set runner.python to python on Windows.
    echo For pull uploads, also set autodl.pull_base_url to an instance-reachable URL.
    pause
    exit /b 0
)

echo Starting Data Relay / Server Training Manager...
echo Executable: %RELAY_EXE%
echo Default WebUI: http://localhost:7800
echo Press Ctrl+C to stop.
echo.
"%RELAY_EXE%" --config config.toml serve
set "RELAY_EXIT_CODE=%ERRORLEVEL%"
echo.
echo Data Relay exited with code %RELAY_EXIT_CODE%.
pause
exit /b %RELAY_EXIT_CODE%
