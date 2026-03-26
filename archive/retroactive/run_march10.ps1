Set-Location 'C:\Users\aland\quantbot'
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo.FileName = 'C:\Users\aland\quantbot\venv\Scripts\python.exe'
$proc.StartInfo.Arguments = 'run_march10.py'
$proc.StartInfo.WorkingDirectory = 'C:\Users\aland\quantbot'
$proc.StartInfo.UseShellExecute = $false
$proc.StartInfo.RedirectStandardOutput = $true
$proc.StartInfo.RedirectStandardError = $true
$proc.Start() | Out-Null
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
Write-Host $stdout
if ($stderr) { Write-Host "STDERR: $stderr" }
