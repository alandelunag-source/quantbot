Set-Location 'C:\Users\aland\quantbot'
Write-Host "Starting..."
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo.FileName = 'C:\Users\aland\quantbot\venv\Scripts\python.exe'
$proc.StartInfo.Arguments = 'run_march9.py'
$proc.StartInfo.WorkingDirectory = 'C:\Users\aland\quantbot'
$proc.StartInfo.UseShellExecute = $false
$proc.StartInfo.RedirectStandardOutput = $true
$proc.StartInfo.RedirectStandardError = $true
$proc.Start() | Out-Null
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
Write-Host "Exit: $($proc.ExitCode)"
Write-Host "STDOUT ($($stdout.Length) chars):"
Write-Host $stdout
Write-Host "STDERR ($($stderr.Length) chars):"
Write-Host $stderr
