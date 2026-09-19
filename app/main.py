"""
Simple RAG chatbot backend.

Flow:
  upload plain-text file -> chunk -> embed -> in-memory vector store (per session)
  chat -> retrieve relevant chunks + conversation history -> Gemini -> answer

The Gemini API key stays here on the server. The browser never sees it.
State is in-memory: it is lost when the process restarts (by design).
"""

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-flash-latest")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "models/gemini-embedding-001")

if not GOOGLE_API_KEY:
    # Fail loudly at startup rather than on the first request.
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and paste your key, "
        "or set the environment variable in your host's dashboard."
    )

# One LLM + one embedding client, reused across requests.
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


# --- In-memory per-session state -------------------------------------------
# sessions[session_id] = {"history": [...], "store": InMemoryVectorStore | None,
#                         "files": [filename, ...]}
sessions: dict[str, dict] = {}


def get_session(session_id: str | None) -> tuple[str, dict]:
    """Return an existing session or create a fresh one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {"history": [], "store": None, "files": [], "suggestions": []}
    return new_id, sessions[new_id]


def generate_suggestions(text: str) -> list[str]:
    """Ask the LLM for a few starter questions answerable from the document."""
    prompt = (
        "Based on the document excerpt below, write 4 short, specific questions a reader "
        "might ask that can be answered from it. Return ONLY the questions, one per line, "
        "with no numbering, bullets, or extra text.\n\nDOCUMENT:\n" + text[:4000]
    )
    try:
        raw = get_text(llm.invoke(prompt))
    except Exception:
        return []
    questions = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("0123456789.)-•*").strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:4]


def get_text(response) -> str:
    """Return the plain text of an LLM response whether `.content` is a string
    or a list of content blocks (some Gemini models add a 'thought' block)."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# --- API --------------------------------------------------------------------
app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def all_errors_as_json(request: Request, exc: Exception):
    """Return unhandled errors as JSON so the frontend can display them
    instead of failing to parse a plain-text 'Internal Server Error'."""
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


@app.post("/api/upload")
async def upload(session_id: str | None = Form(None), file: UploadFile = File(...)):
    """Accept a plain-text file, index it into this session's vector store."""
    raw = await file.read()
    # Try UTF-8 first, then fall back to Latin-1 (common for Project Gutenberg .txt files).
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Could not read the file as text. Please upload a plain-text (.txt/.md) file.",
            )

    if not text.strip():
        raise HTTPException(status_code=400, detail="The file appears to be empty.")

    sid, sess = get_session(session_id)

    doc = Document(page_content=text, metadata={"source": file.filename})
    chunks = splitter.split_documents([doc])

    if sess["store"] is None:
        sess["store"] = InMemoryVectorStore(embeddings)

    try:
        sess["store"].add_documents(chunks)
    except Exception as e:
        # Most commonly: embedding-model name not available for this key, or a rate limit.
        raise HTTPException(
            status_code=502,
            detail=(
                f"Embedding failed ({type(e).__name__}: {e}). "
                f"The file split into {len(chunks)} chunks. If this is a rate limit, try a smaller "
                f"file; if it's a model error, change EMBED_MODEL in your .env."
            ),
        )

    sess["files"].append(file.filename)

    # Generate starter questions from this document (best-effort; never fails the upload).
    try:
        sess["suggestions"] = generate_suggestions(text)
    except Exception:
        sess["suggestions"] = []

    return {
        "session_id": sid,
        "filename": file.filename,
        "chunks": len(chunks),
        "files": sess["files"],
        "suggestions": sess["suggestions"],
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Answer a question using retrieved context + conversation history."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")

    sid, sess = get_session(req.session_id)

    # 1. Retrieve context relevant to this message (if any docs uploaded).
    context = ""
    if sess["store"] is not None:
        retriever = sess["store"].as_retriever(search_kwargs={"k": 4})
        relevant = retriever.invoke(req.message)
        context = "\n\n".join(d.page_content for d in relevant)

    # 2. Build the system instruction for this turn.
    if context:
        system = SystemMessage(content=(
            "You are a helpful assistant. Answer using the CONTEXT below when it is "
            "relevant, and use facts already established earlier in this conversation. "
            "If the answer is not in the context and was not established earlier, say you "
            "don't have that information. Do not make up details.\n\nCONTEXT:\n" + context
        ))
    else:
        system = SystemMessage(content=(
            "You are a friendly, concise assistant. No documents have been uploaded yet, "
            "so answer from general knowledge and the conversation so far."
        ))

    # 3. system + full history + new message.
    messages = [system] + sess["history"] + [HumanMessage(content=req.message)]
    reply = get_text(llm.invoke(messages))

    # 4. Record the turn.
    sess["history"].append(HumanMessage(content=req.message))
    sess["history"].append(AIMessage(content=reply))

    return {"session_id": sid, "reply": reply, "used_context": bool(context)}


@app.post("/api/reset")
async def reset(req: ChatRequest):
    """Clear a session's history and uploaded documents."""
    if req.session_id and req.session_id in sessions:
        del sessions[req.session_id]
    return {"ok": True}


@app.get("/api/session")
async def session_state(session_id: str | None = None):
    """Return the current files and chat history for a session, so a freshly
    loaded page (e.g. a new tab) can restore what the session already contains."""
    if not session_id or session_id not in sessions:
        return {"session_id": session_id, "files": [], "history": []}
    sess = sessions[session_id]
    history = [
        {"role": "user" if isinstance(m, HumanMessage) else "bot", "content": m.content}
        for m in sess["history"]
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return {
        "session_id": session_id,
        "files": sess["files"],
        "history": history,
        "suggestions": sess.get("suggestions", []),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}


# Serve the frontend (index.html) at "/". Must be mounted AFTER the API routes.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
