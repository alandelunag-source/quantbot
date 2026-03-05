@echo off
cd /d C:\Users\aland\quantbot

:: 12:30 PM — exit checks only (catches intraday moves)
venv\Scripts\python.exe -u main.py paper --strategies s01,s02,s03,s05,s07,s09,s10,s11,s12,s13,s14,s15,s16,s17 --once --session midday
