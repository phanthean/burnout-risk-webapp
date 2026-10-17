@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Cai thu vien (chi lan dau mat thoi gian)...
python -m pip install -r requirements.txt --quiet
echo.
start "" http://localhost:8000
python main.py --mode serve
pause
