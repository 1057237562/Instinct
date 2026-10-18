@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
pushd "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found. Activate the training environment first.
    goto failed
)

rem Environment overrides; command-line arguments below take precedence.
if not defined INSTINCT_API_HOST set "INSTINCT_API_HOST=0.0.0.0"
if not defined INSTINCT_API_PORT set "INSTINCT_API_PORT=8998"
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help

rem Paths below are relative to the repository root.
if not defined INSTINCT_API_WEIGHT_PATH set "INSTINCT_API_WEIGHT_PATH=out\best.pth"
if not defined INSTINCT_API_CONFIG set "INSTINCT_API_CONFIG=out\best.json"

:run
echo Starting Instinct OpenAI-compatible API...
echo Default base URL: http://localhost:%INSTINCT_API_PORT%/v1
echo Endpoint: /v1/chat/completions
echo Responses endpoint: /v1/responses
echo Default weight: %INSTINCT_API_WEIGHT_PATH%
echo Press Ctrl+C to stop.
echo.
cd scripts
python serve_openai_api.py --load_from ../model --weight_path "../%INSTINCT_API_WEIGHT_PATH%" --config_path "../%INSTINCT_API_CONFIG%" --host "%INSTINCT_API_HOST%" --port "%INSTINCT_API_PORT%" %*
set "API_EXIT_CODE=%ERRORLEVEL%"
goto done

:help
cd scripts
python serve_openai_api.py --help
set "API_EXIT_CODE=%ERRORLEVEL%"
goto done

:failed
set "API_EXIT_CODE=1"

:done
popd
if not "%API_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] API server exited with code %API_EXIT_CODE%.
    pause
)
exit /b %API_EXIT_CODE%
