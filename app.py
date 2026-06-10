import os, shutil, json, time, uuid, threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utils import logger

app = FastAPI(title="RAG System API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR   = "data/uploads"
VECTOR_DB    = "vector_db"
SESSIONS_DIR = "logs/sessions"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

_agent = None
# abort: maps query_id -> threading.Event (set = aborted)
_abort_events: dict = {}

def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    from main import create_rag_system, setup_environment
    setup_environment()
    _agent = create_rag_system()
    if os.path.exists(f"{VECTOR_DB}/index.faiss"):
        _agent.rag_engine.text_processor.load_vector_store(VECTOR_DB)
        logger.info("向量库已从磁盘加载")
    if not os.path.exists(f"{VECTOR_DB}/index.faiss"):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            try:
                _agent.process_document(fpath)
                logger.info(f"启动补充入库: {fname}")
            except Exception as e:
                logger.warning(f"跳过 {fname}: {e}")
        if _agent.rag_engine.text_processor.vector_store:
            _agent.rag_engine.text_processor.save_vector_store(VECTOR_DB)
    logger.info("RAG agent initialized")
    return _agent

# ── Session helpers ───────────────────────────────────────────────────────────
def session_path(sid): return os.path.join(SESSIONS_DIR, f"{sid}.json")

def load_session(sid):
    p = session_path(sid)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

def save_session(sid, data):
    with open(session_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def list_sessions():
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        sid = fname[:-5]
        try:
            with open(os.path.join(SESSIONS_DIR, fname), encoding="utf-8") as f:
                d = json.load(f)
            sessions.append({"id": sid, "title": d.get("title", "新对话"), "updated_at": d.get("updated_at", 0)})
        except Exception:
            pass
    return sessions

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def index(): return FileResponse("static/index.html")

@app.get("/api/health")
def health(): return {"status": "ok"}

@app.get("/api/files")
def list_files():
    return {"files": [{"name": f, "size": os.path.getsize(os.path.join(UPLOAD_DIR, f))} for f in os.listdir(UPLOAD_DIR)]}

@app.get("/api/sessions")
def get_sessions(): return {"sessions": list_sessions()}

@app.post("/api/sessions")
def new_session():
    sid = uuid.uuid4().hex
    data = {"title": "新对话", "history": [], "updated_at": time.time()}
    save_session(sid, data)
    return {"id": sid, "title": data["title"], "updated_at": data["updated_at"]}

@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    d = load_session(sid)
    if d is None: raise HTTPException(404, "会话不存在")
    return {"id": sid, **d}

@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    p = session_path(sid)
    if os.path.exists(p): os.remove(p)
    return {"message": "已删除"}

# ── Query with abort support ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: str
    query_id: str = ""   # client-generated id for abort
    k: int = 5

@app.post("/api/query")
def query(req: QueryRequest):
    qid = req.query_id or uuid.uuid4().hex
    abort_event = threading.Event()
    _abort_events[qid] = abort_event
    try:
        agent = get_agent()
        if not agent.rag_engine.text_processor.vector_store:
            return {"answer": "请先上传文档，再进行提问。", "sources": [], "query_id": qid}

        d = load_session(req.session_id)
        if d is None: raise HTTPException(404, "会话不存在")
        agent.rag_engine.conversation_history = d.get("history", [])

        # Run LLM in thread so we can check abort
        result_holder = [None]
        error_holder  = [None]

        def run():
            try:
                result_holder[0] = agent.run(req.query)
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join()   # wait — abort_event checked below

        if abort_event.is_set():
            logger.info(f"查询 {qid} 已中断")
            return {"answer": "", "sources": [], "aborted": True, "query_id": qid}

        if error_holder[0]:
            raise error_holder[0]

        result = result_holder[0]
        new_history = agent.rag_engine.get_history()
        title = d.get("title", "新对话")
        if title == "新对话" and new_history:
            title = new_history[0]["content"][:20]
        save_session(req.session_id, {"title": title, "history": new_history, "updated_at": time.time()})

        sources = [
            {"text": r.get("text", "")[:300],
             "score": round(r.get("score", 0), 4),
             "file_name": r.get("metadata", {}).get("file_name", "unknown")}
            for r in result.get("search_results", [])
        ]
        return {"answer": result.get("answer", ""), "sources": sources, "title": title, "query_id": qid}
    except HTTPException:
        raise
    except Exception as e:
        cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
        logger.error(f"Query error: {cause or e}")
        raise HTTPException(500, str(e))
    finally:
        _abort_events.pop(qid, None)

@app.post("/api/abort/{query_id}")
def abort_query(query_id: str):
    ev = _abort_events.get(query_id)
    if ev:
        ev.set()
        return {"message": "已发送中断信号"}
    return {"message": "无进行中的查询"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".md", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed: raise HTTPException(400, f"不支持的文件类型: {ext}")
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        agent = get_agent()
        result = agent.process_document(dest)
        agent.rag_engine.text_processor.save_vector_store(VECTOR_DB)
        return {"message": f"文件 {file.filename} 已处理", "chunks": result["chunks_count"]}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))
