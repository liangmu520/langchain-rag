@echo off
cd /d %~dp0
echo Starting RAG Web Server...
D:\ProgramData\anaconda3\envs\LDX\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause
