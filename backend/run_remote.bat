@echo off
set LTX_APP_DATA_DIR=C:\Users\bradj\AppData\Local\LTXDesktop
set LTX_PORT=8000
set LTX_REMOTE_INFERENCE=1
set LTX_LIVEPEER_SIGNER=https://192.168.1.8:8935
".venv\Scripts\python.exe" ltx2_server.py
