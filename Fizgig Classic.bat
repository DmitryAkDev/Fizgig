@echo off
rem The classic Fizgig desktop app (Tk), with no console window. The new look is Fizgig.bat.
cd /d "%~dp0"
start "" /b wscript //nologo //b "%~dp0src\launch\run_silent.vbs"
