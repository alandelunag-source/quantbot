$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File C:\Users\aland\quantbot\start_dashboard.ps1" `
    -WorkingDirectory "C:\Users\aland\quantbot"

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName "Quantbot Dashboard" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Start Streamlit dashboard on login" `
    -Force | Select-Object TaskName, State

Write-Host "Dashboard auto-start registered."
