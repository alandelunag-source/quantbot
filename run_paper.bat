@echo off
cd /d C:\Users\aland\quantbot
venv\Scripts\python.exe -u main.py paper --strategies s01,s02,s03,s04,s05,s06,s07,s09,s10,s11,s12,s13,s14,s15,s16,s17 --once >> state\paper_trading.log 2>&1
