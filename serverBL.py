# ===========================================================
# serverBL.py (Chatbot engine server - LangGraph Architecture)
# ===========================================================
import os
import pandas as pd
import logging
import asyncio
import re
import threading
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Annotated, List, Dict, Any, Literal, TypedDict
from logging.handlers import RotatingFileHandler
from qdrant_client import QdrantClient

# FastAPI & Core
from fastapi import FastAPI, HTTPException, Request, Depends, Security, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from rag_core import process_and_upsert_file

# Google Sheets
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv(override=True)

# ========== CẤU HÌNH HỆ THỐNG ==========
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 60))
cors_origins = [x.strip() for x in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5500").split(",") if x.strip()]
cors_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
cors_methods = [x.strip() for x in os.getenv("CORS_ALLOW_METHODS", "GET,POST").split(",") if x.strip()]
cors_headers = [x.strip() for x in os.getenv("CORS_ALLOW_HEADERS", "*").split(",") if x.strip()]

GG_CRED_FILE = os.getenv("GOOGLE_SHEETS_CRED_FILE", "google_credentials.json")
GG_SPREADSHEET_ID = os.getenv("GOOGLE_SPREADSHEET_ID")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blu")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBENDDING_MODEL_KWARGS_DEVICE = os.getenv("EMBENDDING_MODEL_KWARGS_DEVICE", "cpu")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", 4))
DEFAULT_RERANK_TOP_N = int(os.getenv("DEFAULT_RERANK_TOP_N", 7))
SEARCH_KWARGS = int(os.getenv("SEARCH_KWARGS", 20))
MAX_CHAT_RESPONSE_TOKEN = int(os.getenv("MAX_CHAT_RESPONSE_TOKEN", 2048))
RETRIEVER_MODEL_KWARGS_DEVICE = os.getenv("RETRIEVER_MODEL_KWARGS_DEVICE", "cpu")

LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5:14b")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.1))

EXTERNAL_LLM_MODEL_NAME = os.getenv("EXTERNAL_LLM_MODEL_NAME", "gemini-3.1-flash-lite-preview")
EXTERNAL_LLM_API_BASE = os.getenv("EXTERNAL_LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY")

ASSETS_DIR = os.getenv("ASSETS_DIR", "assets")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# ========== LOGGING ==========
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

prompt_logger = logging.getLogger("PromptLogger")
prompt_logger.setLevel(logging.INFO)
prompt_logger.propagate = False 
os.makedirs(LOG_DIR, exist_ok=True)
prompt_handler = RotatingFileHandler(os.path.join(LOG_DIR, "prompt_debug.log"), maxBytes=2 * 1024 * 1024, backupCount=1, encoding="utf-8")
prompt_handler.setFormatter(logging.Formatter('%(message)s'))
if not prompt_logger.handlers:
    prompt_logger.addHandler(prompt_handler)

class PromptLoggerCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_log = f"\n{'='*50}\n[{timestamp}] CHÍNH XÁC PROMPT ĐƯỢC GỬI VÀO LLM:\n{'-'*50}\n{prompts[0]}\n{'='*50}\n"
        prompt_logger.info(new_log)

# ========== BACKGROUND TASKS & LIFESPAN ==========
class SessionData:
    def __init__(self):
        self.history = ChatMessageHistory()
        self.last_accessed = datetime.now()

store: Dict[str, SessionData] = {}
user_session = {}
sheet_lock = threading.Lock()

async def cleanup_inactive_sessions():
    while True:
        try:
            await asyncio.sleep(600)
            now = datetime.now()
            expired_sessions = [sid for sid, data in store.items() if now - data.last_accessed > timedelta(minutes=SESSION_TIMEOUT_MINUTES)]
            for sid in expired_sessions:
                del store[sid]
            if expired_sessions:
                logger.info(f"Dọn rác RAM: Đã xóa {len(expired_sessions)} session không hoạt động.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp session: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_inactive_sessions())
    yield
    task.cancel()

# ========== FASTAPI KHỞI TẠO ==========
app = FastAPI(title="BLU Chatbot API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=cors_credentials, allow_methods=cors_methods, allow_headers=cors_headers)

os.makedirs(ASSETS_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/chat-client", StaticFiles(directory="templates/chat-client"), name="chat-client")
templates = Jinja2Templates(directory="templates")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
RATE_LIMIT_ERROR_MSG = os.getenv("RATE_LIMIT_ERROR_MSG", "Vui lòng chờ trong giây lát rồi thử lại!")

async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": RATE_LIMIT_ERROR_MSG})
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

PUBLIC_API_KEY = os.getenv("PUBLIC_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_public_key(api_key: str = Security(api_key_header)):
    if api_key != PUBLIC_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Public API Key")
    return api_key

def verify_admin_key(api_key: str = Security(api_key_header)):
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Admin API Key")
    return api_key

# ========== GOOGLE SHEETS ==========
def get_or_create_monthly_sheet(sheet_prefix: str, headers: list):
    if not gg_sheet: return None
    current_month = datetime.now().strftime("%m_%Y")
    sheet_name = f"{sheet_prefix}_{current_month}"
    with sheet_lock:
        try:
            spreadsheet = gg_sheet.get(spreadsheetId=GG_SPREADSHEET_ID).execute()
            existing_sheets = [s['properties']['title'] for s in spreadsheet.get('sheets', [])]
            if sheet_name not in existing_sheets:
                req_create = {"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]}
                gg_sheet.batchUpdate(spreadsheetId=GG_SPREADSHEET_ID, body=req_create).execute()
                gg_sheet.values().update(spreadsheetId=GG_SPREADSHEET_ID, range=f"{sheet_name}!A1", valueInputOption="RAW", body={"values": [headers]}).execute()
            return sheet_name
        except Exception as e:
            logger.error(f"GG Sheets Error: {e}")
            return None

def push_to_google_sheets(sheet_prefix: str, headers: list, data_row: list):
    sheet_name = get_or_create_monthly_sheet(sheet_prefix, headers)
    if not sheet_name: return
    with sheet_lock:
        try:
            gg_sheet.values().append(spreadsheetId=GG_SPREADSHEET_ID, range=f"{sheet_name}!A:A", valueInputOption='USER_ENTERED', insertDataOption='INSERT_ROWS', body={'values': [data_row]}).execute()
        except Exception as e:
            logger.error(f"Lỗi ghi data lên GG Sheets: {e}")

def setup_ggsheet():
    try:
        creds = service_account.Credentials.from_service_account_file(GG_CRED_FILE, scopes=['https://www.googleapis.com/auth/spreadsheets'])
        return build('sheets', 'v4', credentials=creds, cache_discovery=False).spreadsheets()
    except Exception as e:
        logger.error(f"Lỗi kết nối GG Sheets: {e}")
        return None

# ========== CẤU HÌNH AI & RAG ==========
def setup_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': EMBENDDING_MODEL_KWARGS_DEVICE}, encode_kwargs={'normalize_embeddings': True})
    client = QdrantClient(url=QDRANT_URL)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings, sparse_embedding=sparse_embeddings, retrieval_mode=RetrievalMode.HYBRID)

def setup_retriever(qdrant):
    hybrid_retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": SEARCH_KWARGS})
    try:
        from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL, model_kwargs={'device': RETRIEVER_MODEL_KWARGS_DEVICE})
        return ContextualCompressionRetriever(base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=DEFAULT_RERANK_TOP_N), base_retriever=hybrid_retriever)
    except:
        hybrid_retriever.search_kwargs["k"] = DEFAULT_RERANK_TOP_N 
        return hybrid_retriever

def setup_llm():
    # Sử dụng ChatOpenAI trỏ tới chuẩn v1 của Ollama để Tool Calling chuẩn xác hơn
    local_llm = ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        openai_api_base=LLM_API_BASE,
        openai_api_key=LLM_API_KEY,                     
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_CHAT_RESPONSE_TOKEN,
        top_p=LLM_TOP_P,
        streaming=True,
        callbacks=[PromptLoggerCallback()]
    )
    if EXTERNAL_API_KEY:
        external_llm = ChatOpenAI(
            model_name=EXTERNAL_LLM_MODEL_NAME,
            openai_api_base=EXTERNAL_LLM_API_BASE,
            openai_api_key=EXTERNAL_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_CHAT_RESPONSE_TOKEN,
            streaming=True,      
            callbacks=[PromptLoggerCallback()]
        )
        return external_llm.with_fallbacks([local_llm])
    return local_llm

try:
    qdrant = setup_vector_store()
    retriever = setup_retriever(qdrant)
    main_llm = setup_llm()
    gg_sheet = setup_ggsheet()
    logger.info("✅ All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# ========== TOOL & LANGGRAPH WORKFLOW ==========
@tool
def search_admission_info(query: str) -> str:
    """Sử dụng để tìm kiếm thông tin về tuyển sinh, ngành học, điểm chuẩn, học phí, chỉ tiêu của Đại học Bạc Liêu."""
    try:
        docs = retriever.invoke(query) 
        if not docs: return "Không tìm thấy thông tin cụ thể trong cơ sở dữ liệu."
        return "\n\n".join([f"--- NGUỒN: {doc.metadata.get('source', 'Không rõ')} ---\n{doc.page_content}" for doc in docs])
    except Exception as e:
        return f"Lỗi hệ thống khi truy xuất: {str(e)}"

tools = [search_admission_info]
llm_with_tools = main_llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: AgentState):
    """Đọc toàn bộ lịch sử, quyết định gọi tool hoặc sinh câu trả lời cuối cùng."""
    sys_msg = SystemMessage(content="""Bạn là chuyên viên tư vấn tuyển sinh của Đại học Bạc Liêu (BLU).
QUY TẮC:
1. CHỈ SỬ DỤNG TIẾNG VIỆT. Trả lời ngắn gọn, đúng trọng tâm.
2. NẾU CẦN TÌM THÔNG TIN hãy sử dụng công cụ tìm kiếm được cung cấp.
3. TUYỆT ĐỐI KHÔNG thông báo hành động trước khi gọi tool (VD: Không nói "Mình sẽ tra cứu...", "Vui lòng đợi...", "Dựa theo lịch sử..."). Nếu cần dùng công cụ, HÃY CHỈ XUẤT TRỰC TIẾP CÚ PHÁP GỌI TOOL mà không kèm theo bất kỳ văn bản/câu từ nào khác.
4. Nếu dữ liệu có sự mâu thuẫn, hãy sữ ưu tiên dữ liệu có version cao hơn.
5. Nếu không tìm thấy thông tin từ công cụ, hãy nói rõ: "Hiện tại chưa có thông tin về vấn đề này". Tuyệt đối không tự bịa thông tin, URL, SĐT.
6. Nếu câu hỏi không liên quan đến vấn đề tuyển sinh, việc làm sau tốt nghiệp mà liên quan đến các chủ đề khác như cuộc sống, thời tiết, nấu nướng, tâm lý,... thì nên từ chối trả lời lịch sự và không cần gọi tool tìm kiếm.
7. Sử dụng cách gọi "Mình" - "Bạn".
""")
    # Đưa toàn bộ lịch sử (bao gồm cả kết quả từ tool) vào LLM
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Kiểm tra xem LLM có muốn gọi tool không."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
# Nếu có tool -> sang node tools. Nếu không -> KẾT THÚC
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
# Sau khi tool chạy xong, BẮT BUỘC quay lại agent để tổng hợp câu trả lời cho user
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()

# ========== API ROUTES ==========
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store: store[session_id] = SessionData()
    session_data = store[session_id]
    session_data.last_accessed = datetime.now()
    if len(session_data.history.messages) > MAX_CHAT_HISTORY * 2:
        session_data.history.messages = session_data.history.messages[-(MAX_CHAT_HISTORY * 2):]
    return session_data.history

class ChatRequest(BaseModel):
    session_id: str = Field(..., max_length=100)
    chat_request: str = Field(..., max_length=500)

class UserInfo(BaseModel):
    name: str
    phone: str
    email: str
    address: str

@app.post("/chat_stream")
@limiter.limit("15/minute")
async def chat_stream_endpoint(request: Request, chat_req: ChatRequest, api_key: str = Depends(verify_public_key)):
    session_id = chat_req.session_id
    user_msg = chat_req.chat_request

    if not user_msg.strip():
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống")

    session_history = get_session_history(session_id)
    
    # Nạp lịch sử vào Graph Input
    inputs = {"messages": session_history.messages + [HumanMessage(content=user_msg)]}

    async def stream_generator():
        full_answer = ""
        try:
            async for event in app_graph.astream_events(inputs, version="v2"):
                if await request.is_disconnected():
                    break

                kind = event["event"]
                name = event["name"]

                if LANGCHAIN_DEBUG:
                    # ---  IN LOG RA CONSOLE ---
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Event: {kind} | Node: {name}")
                    if kind == "on_chat_model_stream":
                        chunk_content = event["data"]["chunk"].content
                        if chunk_content:
                            print(f"Chunk ({name}): {chunk_content}", end="", flush=True)
                    # ---------------------------------------------

                # 1. BẮT SỰ KIỆN ĐỂ PHÁT TRẠNG THÁI
                if kind == "on_chain_start":
                    if name == "agent":
                        yield "[[STATUS:Đang phân tích...]]"
                    elif name == "tools":
                        yield "[[STATUS:Đang tra cứu...]]"

                # 2. PHÁT NỘI DUNG VĂN BẢN TRẢ LỜI
                if kind == "on_chat_model_stream" and event["metadata"].get("langgraph_node") == "agent":
                    chunk = event["data"]["chunk"]

                    if chunk.tool_call_chunks:
                        continue

                    if chunk.content: # Chỉ lấy nội dung văn bản trực tiếp
                        full_answer += chunk.content
                        yield chunk.content
                        await asyncio.sleep(0.01)

            if LANGCHAIN_DEBUG: 
                print("\n--- KẾT THÚC LUỒNG STREAM ---")
            
            if full_answer.strip() and not await request.is_disconnected():
                session_history.add_user_message(user_msg)
                session_history.add_ai_message(full_answer)
            else:
                logger.warning(f"Bỏ qua lưu lịch sử do AI trả về chuỗi rỗng. Session: {session_id}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"\n[Lỗi hệ thống: {str(e)}]"

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.post("/user_info")
@limiter.limit("3/minute")
async def save_user_info(request: Request, user_info: UserInfo, background_tasks: BackgroundTasks):
    if not user_info.email: raise HTTPException(status_code=400, detail="Email không được để trống")
    user_session[user_info.email] = user_info.model_dump()
    row_data = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_info.name, user_info.phone, user_info.email, user_info.address]
    background_tasks.add_task(push_to_google_sheets, "User", ["Thời gian", "Họ Tên", "Số ĐT", "Email", "Địa chỉ"], row_data)
    return {"message": "Lưu thông tin thành công!"}

@app.get("/", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("templates/favicon.ico")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")