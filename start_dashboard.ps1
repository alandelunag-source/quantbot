Start-Process -FilePath "C:\Users\aland\quantbot\venv\Scripts\streamlit.exe" `
  -ArgumentList "run","dashboard.py","--server.port","8501","--server.headless","true" `
  -WorkingDirectory "C:\Users\aland\quantbot" `
  -WindowStyle Hidden
