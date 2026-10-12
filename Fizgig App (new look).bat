@echo off
rem Fizgig in its own app window - the web UI, the new look (the classic app is run_fizgig.bat)
cd /d "%~dp0"
start "" "venv\Scripts\pythonw.exe" lora_trainer_gui.py --app
