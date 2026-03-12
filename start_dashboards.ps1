# Start all dashboards
# Run this after reboot or when dashboards go down

Write-Host "Starting Quantbot dashboard (port 8501)..." -ForegroundColor Cyan
Start-Process -FilePath "C:\Users\aland\quantbot\venv\Scripts\streamlit.exe" `
    -ArgumentList "run C:\Users\aland\quantbot\dashboard.py --server.port 8501 --server.headless true" `
    -WindowStyle Minimized

Start-Sleep 2

Write-Host "Starting Bracketbot dashboard (port 8055)..." -ForegroundColor Cyan
Start-Process -FilePath "C:\Users\aland\bracketbot\venv\Scripts\python.exe" `
    -ArgumentList "C:\Users\aland\bracketbot\dashboard.py" `
    -WindowStyle Minimized

Start-Sleep 3

Write-Host ""
Write-Host "Dashboards:" -ForegroundColor Green
Write-Host "  Quantbot:   http://localhost:8501" -ForegroundColor White
Write-Host "  Bracketbot: http://localhost:8055" -ForegroundColor White
