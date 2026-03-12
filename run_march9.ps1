Set-Location 'C:\Users\aland\quantbot'
$output = & 'venv\Scripts\python.exe' 'run_march9.py' 2>&1
$output | Out-File 'state\march9_run.log' -Encoding utf8
Write-Host "Lines captured: $($output.Count)"
$output | Select-Object -First 80
