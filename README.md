# EduBot RAG: Engineering Academic Mentor System

EduBot is an advanced Retrieval-Augmented Generation (RAG) system designed as an interactive Engineering Academic Mentor. It integrates semantic search, intent classification, local Large Language Models (LLM), and real-time Text-to-Speech (TTS) synthesis to assist students with engineering curricula, placements, VTU certifications, and LMS support.

## 🚀 Key Features

- **Semantic Query Engine**: Uses ChromaDB vector databases to query curated engineering mentoring datasets. Includes dual engines for both the ultra-premium and a synthetic 5k dataset.
- **Dynamic Session State**: Automatically tracks student conversation state, active topics (e.g., placements, certifications, courses), target domains, and subject weaknesses.
- **Intelligent Classification & Guardrails**: Enforces boundaries using an educational guardrail layer and intent classifier to ensure all queries relate strictly to engineering academics or career development.
- **Dual-Mode Streaming (SSE)**: Streams answers live to the frontend using Server-Sent Events (SSE), separating text to speak (TTS) from code/roadmaps blocks in the UI.
- **Local Text-to-Speech (TTS)**: Synthesizes high-quality audio utilizing Kokoro-ONNX and Kokoro TTS models with warmup on server startup.
- **JWT Authentication**: Secure endpoints with JSON Web Token (JWT) user authentication and student performance analytics dashboarding.

---

## 📁 Repository Structure

```
├── ai_core/                # Shared AI utilities and intent classification
├── backend/
│   ├── api.py              # FastAPI Web Application & API endpoints
│   ├── main.py             # Interactive Console Menu (RAG diagnostics, chat CLI)
│   ├── db.py               # SQLite database interface for students and analytics
│   ├── edmentor/           # Custom voice-mentor components (Guards, Memory, TTS, LLM)
│   ├── rag/                # RAG pipeline implementation (Index builders, retrievers)
│   ├── static/             # Frontend static files (HTML, CSS, JS Chat UI, Dashboard)
│   ├── requirements.txt    # Package dependencies
│   └── .env.example        # Environment configuration template
└── README.md               # Main project documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone & Navigate
```bash
git clone https://github.com/PraneetNS/edbot_rag.git
cd edbot_rag
```

### 2. Environment Configuration
Create a `.env` file inside the `backend/` directory based on `backend/.env.example`:
```bash
cp backend/.env.example backend/.env
```
Fill in the configuration details, including local Ollama endpoints and JWT secrets.

### 3. Install Dependencies
Make sure you are using Python 3.10+ and install requirements:
```bash
pip install -r backend/requirements.txt
```

### 4. Build/Verify Indices
Ensure ChromaDB is populated by running the indexing build script or checking status with the diagnostics dashboard:
```bash
python backend/main.py
# Select Option 1 (Rebuild Knowledge Base) or Option 4 (System Diagnostics)
```

---

## 🖥️ Running the Application

### Running the API Server
Start the FastAPI server via Uvicorn:
```bash
cd backend
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
Visit `http://127.0.0.1:8000/` in your browser to interact with the web chat client.

### Running the Console Client
For a direct terminal interface:
```bash
python backend/main.py
# Select Option 3 to chat with EduMentor in your terminal
```
