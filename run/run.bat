@echo off
echo Starting MidiGen...
cd /d "%~dp0.."
call conda activate base 2>nul
python -m run.main
pause
