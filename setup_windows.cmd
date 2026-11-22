@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto install
for %%V in (3.12 3.11 3.10) do (
  py -%%V -c "import sys; print(sys.version)" >nul 2>&1
  if not errorlevel 1 (
    py -%%V -m venv .venv
    if errorlevel 1 goto failed
    goto install
  )
)
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
if errorlevel 1 (
  echo Please install Python 3.12, or send your Python version to ChatGPT.
  exit /b 1
)
python -m venv .venv
if errorlevel 1 goto failed
:install
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe check_env.py
if errorlevel 1 goto failed
echo READY. Run: .\.venv\Scripts\python.exe train.py --model gcn --epochs 200
exit /b 0
:failed
echo Setup failed. Copy the error output and send it to ChatGPT.
exit /b 1
