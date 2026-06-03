@echo off
REM Start Courseware QA System
echo Starting Courseware QA System...
cd /d "%~dp0.."
streamlit run app.py
pause
