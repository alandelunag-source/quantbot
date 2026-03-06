@echo off
cd /d C:\Users\aland\quantbot

:: Get today's date as YYYY-MM-DD
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set TODAY=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%

:: Unset nested-session guard so claude can run from scheduler or manually
set CLAUDECODE=

%USERPROFILE%\.local\bin\claude.exe --dangerously-skip-permissions -p ^
"You are the autonomous daily reviewer for the quantbot paper trading system. Today is %TODAY%. ^
Do the following in order: ^
1. Run `venv\Scripts\python.exe main.py status` to get live MTM portfolio values. ^
2. Read state/paper_log.csv to see the P&L history across all days. ^
3. Read the last 80 lines of state/paper_trading.log for today's activity and any errors. ^
4. Write a concise daily review to state/daily_review_%TODAY%.md with these sections: ^
   ## Portfolio Summary (total value, total return, vs SPY/QQQ alpha) ^
   ## Per-Strategy Breakdown (table: strategy | value | return | positions) ^
   ## Movers (top gainers and losers today) ^
   ## Issues (any yfinance errors, scheduler failures, data gaps) ^
   ## Watch Tomorrow (1-2 lines on what to monitor) ^
   Sign off as 'Claude (autonomous reviewer) — %TODAY%'. ^
5. Append a 3-line plain-text summary (no markdown) to state/paper_trading.log." ^
--allowedTools "Bash Read Write Glob Grep"
