$action = New-ScheduledTaskAction `
    -Execute "C:\Users\aland\quantbot\run_daily.bat" `
    -WorkingDirectory "C:\Users\aland\quantbot"

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "16:15"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -WakeToRun

Register-ScheduledTask `
    -TaskName "Quantbot Daily" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Daily paper trading update + autonomous Claude review at 4:15pm Mon-Fri" `
    -Force | Select-Object TaskName, State

Write-Host "Task registered. Verify with: schtasks /Query /TN 'Quantbot Daily' /FO LIST"
