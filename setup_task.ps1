# Register 3 Quantbot scheduled tasks: open (9:35), midday (12:30), close (3:45)
# Run: powershell -ExecutionPolicy Bypass -File setup_task.ps1

$workdir = "C:\Users\aland\quantbot"
$weekdays = "Monday","Tuesday","Wednesday","Thursday","Friday"

$sessions = @(
    @{ Name = "Quantbot Open";   Bat = "run_open.bat";   Time = "09:35"; Desc = "Quantbot open session: exit checks only" },
    @{ Name = "Quantbot Midday"; Bat = "run_midday.bat"; Time = "12:30"; Desc = "Quantbot midday session: exit checks only" },
    @{ Name = "Quantbot Close";  Bat = "run_close.bat";  Time = "15:45"; Desc = "Quantbot close session: full update + review" }
)

foreach ($s in $sessions) {
    $action = New-ScheduledTaskAction `
        -Execute "$workdir\$($s.Bat)" `
        -WorkingDirectory $workdir

    $trigger = New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek $weekdays `
        -At $s.Time

    $settings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -WakeToRun

    Register-ScheduledTask `
        -TaskName $s.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $s.Desc `
        -Force | Select-Object TaskName, State

    $task = Get-ScheduledTask -TaskName $s.Name
    $task.Settings.WakeToRun = $true
    $task.Settings.StartWhenAvailable = $true
    Set-ScheduledTask -InputObject $task | Out-Null

    Write-Host "Registered: $($s.Name) at $($s.Time)"
}

Unregister-ScheduledTask -TaskName "Quantbot Daily" -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Removed legacy Quantbot Daily task"
Write-Host ""
Write-Host "Verify: schtasks /Query /FO LIST | findstr /i quantbot"
