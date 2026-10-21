@echo off
setlocal enabledelayedexpansion
title Healthcare RAG Agent - System Launcher
cd /d "%~dp0"

echo ==========================================================
echo        HEALTHCARE RAG AGENT - SYSTEM LAUNCHER
echo ==========================================================
echo.

:: 1. Check & Start Docker Container (pgvector)
echo [1/5] Checking PostgreSQL + pgvector Docker container...
docker ps -a --format "{{.Names}}" | findstr /i "healthcare-pgvector" >nul
if %errorlevel% neq 0 (
    echo Starting PostgreSQL + pgvector via Docker Compose...
    docker compose up -d
) else (
    echo Ensuring healthcare-pgvector container is running...
    docker start healthcare-pgvector >nul 2>&1
)

:: Wait briefly for PostgreSQL to become ready
echo Waiting for PostgreSQL to be ready on port 5433...
timeout /t 3 /nobreak >nul

:: 2. Check & Start Ollama
echo.
echo [2/5] Checking Ollama Service...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama is not running. Attempting to start Ollama in background...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
        echo Waiting for Ollama server to initialize...
        timeout /t 5 /nobreak >nul
    ) else (
        echo [WARNING] Ollama executable not found at default path. Please ensure 'ollama serve' is running.
    )
) else (
    echo [OK] Ollama is running and responding on port 11434.
)

:: 3. Pull required models if missing
echo.
echo [3/5] Verifying Ollama Models (nomic-embed-text & llama3.2)...
ollama list | findstr /i "nomic-embed-text" >nul 2>&1
if %errorlevel% neq 0 (
    echo Pulling nomic-embed-text embedding model...
    ollama pull nomic-embed-text
) else (
    echo [OK] nomic-embed-text model is ready.
)

ollama list | findstr /i "llama3.2" >nul 2>&1
if %errorlevel% neq 0 (
    echo Pulling llama3.2 LLM model...
    ollama pull llama3.2
) else (
    echo [OK] llama3.2 model is ready.
)

:: 4. Verify Database & Ingestion
echo.
echo [4/5] Checking pgvector Knowledge Base...
python -c "from src.db import db; count = db.get_chunk_count(); exit(0 if count > 0 else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Knowledge base is empty or uninitialized. Running automated ingestion...
    python main.py ingest
) else (
    echo [OK] Knowledge base is already populated with medication monographs.
)

:: 5. Launch Web GUI Dashboard
echo.
echo [5/5] Launching Healthcare RAG Web GUI Dashboard...
echo Dashboard will be accessible at: http://localhost:8505
echo.
echo ==========================================================
echo Press Ctrl+C in this terminal window to stop the dashboard.
echo ==========================================================
start http://localhost:8505
python -m streamlit run app.py --server.port 8505

pause
