@echo off
cd /d d:\TouchDiffusionMKII
set GRADIO_SERVER_NAME=0.0.0.0
set GRADIO_SERVER_PORT=7861
"d:\TouchDiffusionMKII\.venv\Scripts\gradio.exe" gradio_prepare.py
pause
