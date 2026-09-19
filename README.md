# Document Chatbot (RAG + Gemini)

A small web chatbot: upload a plain-text document, then ask questions grounded in it.
Combines conversation memory with retrieval-augmented generation (RAG).

- **Backend:** FastAPI + LangChain + Google Gemini (holds the API key, does embeddings/retrieval/chat)
- **Frontend:** one static HTML page, served by the backend (no build step)
- **State:** in-memory per session — uploaded docs and history are lost on restart (by design)

```
Browser (upload + chat)  ->  FastAPI  ->  Google Gemini API
```

## 1. Prerequisites

- Python 3.10+
- A free Gemini API key: https://aistudio.google.com/app/apikey

## 2. Set up locally

```bash
# from the project folder
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# create your .env and paste your key into it
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Open `.env` and set `GOOGLE_API_KEY=...` with your real key.

## 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser. Upload a `.txt`/`.md` file and start chatting.

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — (required) | Your Gemini API key |
| `CHAT_MODEL` | `gemini-flash-latest` | Chat model |
| `EMBED_MODEL` | `models/gemini-embedding-001` | Embedding model |

If a model name errors for your key/region, change it here.

## Files

```
app/
  main.py           # FastAPI backend: /api/upload, /api/chat, /api/reset, /api/health
  static/
    index.html      # the chat UI (served at /)
requirements.txt
.env.example        # copy to .env and add your key
```
