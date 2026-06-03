@echo off
cd /d "%~dp0"
title Courseware QA
setlocal enabledelayedexpansion

echo ========================================
echo   Courseware QA System Launcher
echo ========================================
echo.

REM === 1. Find conda ===
set CONDA_CMD=
where conda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where conda') do set CONDA_CMD=%%i
    goto :conda_found
)

for %%d in ("%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
    if exist "%%~d\condabin\conda.bat" (
        set CONDA_CMD=%%~d\condabin\conda.bat
        goto :conda_found
    )
)

echo [ERROR] conda not found
pause
exit /b 1

:conda_found
echo [OK] conda: %CONDA_CMD%

REM === 2. Find env python ===
set ENV_NAME=courseware-qa
set ENV_PYTHON=

for /f "tokens=1,2" %%a in ('call "%CONDA_CMD%" info --envs 2^>nul') do (
    if "%%a"=="%ENV_NAME%" (
        set ENV_PYTHON=%%b\python.exe
        if exist "!ENV_PYTHON!" goto :env_found
    )
)

REM Not found - create it
echo [INFO] Creating conda env "%ENV_NAME%" ...
call "%CONDA_CMD%" env create -f environment.yml
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create environment
    pause
    exit /b 1
)
echo [INFO] Installing PyTorch CUDA ...
call "%CONDA_CMD%" run -n %ENV_NAME% pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo [WARN] CUDA failed, trying CPU ...
    call "%CONDA_CMD%" run -n %ENV_NAME% pip install torch torchvision torchaudio
)

REM Find newly created env
for /f "tokens=1,2" %%a in ('call "%CONDA_CMD%" info --envs 2^>nul') do (
    if "%%a"=="%ENV_NAME%" (
        set ENV_PYTHON=%%b\python.exe
        if exist "!ENV_PYTHON!" goto :env_found
    )
)

echo [ERROR] Cannot find python.exe for "%ENV_NAME%"
pause
exit /b 1

:env_found
echo [OK] Python: %ENV_PYTHON%

REM === 3. Check API Key ===
if exist ".env" goto :check_deps
echo [INFO] No .env file found
set /p API_KEY="Enter DeepSeek API Key (Enter to skip): "
if not "!API_KEY!"=="" (
    echo DEEPSEEK_API_KEY=!API_KEY!> .env
    echo DEEPSEEK_MODEL=deepseek-v4-pro>> .env
    echo [OK] .env created
)
echo.

REM === 4. Check deps ===
:check_deps
"%ENV_PYTHON%" -c "import streamlit, chromadb, fitz, openai, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing dependencies ...
    "%ENV_PYTHON%" -m pip install -r requirements.txt
)

REM === 5. Launch ===
echo [INFO] Starting http://localhost:8501
echo        Press Ctrl+C to stop
echo ========================================
echo.

REM Skip HuggingFace network check (model already cached locally)
set HF_HUB_OFFLINE=1

"%ENV_PYTHON%" -m streamlit run app.py
pause
