Start-Process -FilePath "C:\Program Files (x86)\cloudflared\cloudflared.exe" `
  -ArgumentList "tunnel","--url","http://localhost:8501" `
  -RedirectStandardOutput "C:\Users\aland\quantbot\state\cf_out.log" `
  -RedirectStandardError  "C:\Users\aland\quantbot\state\cf_err.log" `
  -WindowStyle Hidden

Start-Sleep 10
Get-Content "C:\Users\aland\quantbot\state\cf_err.log" | Select-String "trycloudflare|https://"
