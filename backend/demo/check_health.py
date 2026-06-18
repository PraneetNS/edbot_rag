import os
import sys
import sqlite3
from pathlib import Path
import httpx

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "edumentor.db"
ENV_PATH = BACKEND_DIR / ".env"

def print_result(label: str, status: bool, detail: str = ""):
    marker = "✓" if status else "✗"
    status_str = "OK" if status else "FAILED"
    print(f" [{marker}] {label:.<35} {status_str} {f'({detail})' if detail else ''}")

def run_diagnostics():
    print("═" * 65)
    print("               EDUBOT SYSTEM DIAGNOSTICS & HEALTH CHECK")
    print("═" * 65)

    # 1. Check Environment File
    env_ok = ENV_PATH.exists()
    detail_env = ".env file loaded" if env_ok else ".env file missing"
    print_result("Environment Config (.env)", env_ok, detail_env)
    
    # 2. Check JWT Auth Keys in Env
    jwt_ok = False
    detail_jwt = "No environment variables"
    if env_ok:
        with open(ENV_PATH, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            has_secret = "SECRET_KEY=" in content
            has_expire = "JWT_EXPIRE_HOURS=" in content
            jwt_ok = has_secret and has_expire
            if jwt_ok:
                detail_jwt = "JWT keys defined"
            elif has_secret:
                detail_jwt = "JWT_EXPIRE_HOURS missing"
            elif has_expire:
                detail_jwt = "SECRET_KEY missing"
            else:
                detail_jwt = "JWT auth configuration missing"
    print_result("JWT Configuration keys", jwt_ok, detail_jwt)

    # 3. Check SQLite Database
    db_ok = DB_PATH.exists()
    detail_db = ""
    if db_ok:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Check student count
            cursor.execute("SELECT COUNT(*) FROM students")
            student_cnt = cursor.fetchone()[0]
            
            # Check turn count
            cursor.execute("SELECT COUNT(*) FROM turns")
            turn_cnt = cursor.fetchone()[0]

            # Check session count
            cursor.execute("SELECT COUNT(*) FROM sessions")
            session_cnt = cursor.fetchone()[0]
            
            detail_db = f"{student_cnt} students, {turn_cnt} turns, {session_cnt} sessions"
            conn.close()
        except Exception as e:
            db_ok = False
            detail_db = f"DB connection failed: {e}"
    else:
        detail_db = "edumentor.db database file does not exist"
    print_result("SQLite Database (edumentor.db)", db_ok, detail_db)

    # 4. Check Ollama API Status
    ollama_url = "http://localhost:11434"
    if env_ok:
        with open(ENV_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("OLLAMA_BASE_URL="):
                    ollama_url = line.split("=")[1].strip()
                    break
    
    ollama_ok = False
    detail_ollama = ""
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)
        if r.status_code == 200:
            ollama_ok = True
            models = [m["name"] for m in r.json().get("models", [])]
            detail_ollama = f"Online, Models: {', '.join(models[:3])}"
        else:
            detail_ollama = f"HTTP status {r.status_code}"
    except Exception as e:
        detail_ollama = "Connection refused (Ollama daemon offline)"
    print_result("Ollama Local LLM service", ollama_ok, detail_ollama)

    # 5. Check FastAPI App Endpoint
    api_url = "http://127.0.0.1:8000/health"
    api_ok = False
    detail_api = ""
    try:
        r = httpx.get(api_url, timeout=2.0)
        if r.status_code == 200:
            api_ok = True
            data = r.json()
            chunks = data.get("primary_chunks", 0)
            tts_ready = "Ready" if data.get("edmentor_ready", False) else "Not Loaded"
            detail_api = f"Online, RAG Chunks: {chunks}, TTS: {tts_ready}"
        else:
            detail_api = f"HTTP status {r.status_code}"
    except Exception as e:
        detail_api = "Offline (FastAPI server is not running)"
    print_result("FastAPI App Server (/health)", api_ok, detail_api)

    print("═" * 65)

if __name__ == "__main__":
    run_diagnostics()
