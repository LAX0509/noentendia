@echo off
REM Instala (la primera vez) y abre NoEntendIA en http://localhost:8000
cd /d "%~dp0"
if not exist .venv (
  echo Creando entorno virtual e instalando dependencias...
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements-dev.txt
)
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn api.index:app --port 8000
