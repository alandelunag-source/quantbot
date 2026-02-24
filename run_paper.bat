@echo off
cd /d C:\Users\aland\quantbot
venv\Scripts\python.exe main.py paper --strategies s09,s02,s06,s10,s07 --once >> state\paper_trading.log 2>&1
