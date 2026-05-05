import os
import pandas as pd
import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_classic.chains import LLMChain
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import File, UploadFile, BackgroundTasks, Request
from fastapi.templating import Jinja2Templates
from fastapi import Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi import Security, Depends
from fastapi.security import APIKeyHeader
from fastapi import BackgroundTasks

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import time
import re
import json
from functools import lru_cache
import asyncio
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
import threading
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from contextlib import asynccontextmanager


import logging
from logging.handlers import RotatingFileHandler

load_dotenv(override=True)

# Thời gian tối đa giữ session trong RAM nếu không có tương tác (phút)
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 60))

cors_origins = [x.strip() for x in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5500").split(",") if x.strip()]
cors_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
cors_methods = [x.strip() for x in os.getenv("CORS_ALLOW_METHODS", "GET,POST").split(",") if x.strip()]
cors_headers = [x.strip() for x in os.getenv("CORS_ALLOW_HEADERS", "*").split(",") if x.strip()]

set_llm_cache(SQLiteCache(database_path="logs/llm_cache.db"))


# --- GOOGLE SHEETS SETUP ---
GG_CRED_FILE = os.getenv("GOOGLE_SHEETS_CRED_FILE", "google_credentials.json")
GG_SPREADSHEET_ID = os.getenv("GOOGLE_SPREADSHEET_ID")

# --- THÔNG SỐ VECTOR STORE DATABASE ---
DATABASE_PATH = os.getenv("DATABASE_PATH", "database/collection")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blu")

# --- THÔNG SỐ EMBEDDING ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBENDDING_MODEL_KWARGS_DEVICE = os.getenv("EMBENDDING_MODEL_KWARGS_DEVICE", "cpu")

# --- THÔNG SỐ RETRIEVER ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", 2))
DEFAULT_RERANK_TOP_N = int(os.getenv("DEFAULT_RERANK_TOP_N", 5))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", 0.4))
SEARCH_KWARGS = int(os.getenv("SEARCH_KWARGS", 20))
MAX_CHAT_RESPONSE_TOKEN = int(os.getenv("MAX_CHAT_RESPONSE_TOKEN", 1024))
RETRIEVER_MODEL_KWARGS_DEVICE = os.getenv("RETRIEVER_MODEL_KWARGS_DEVICE", "cpu")

# --- THÊM CÁC BIẾN LLM TỪ ENV ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")

# --- THÔNG TIN API CHO MÔ HÌNH BÊN THỨ 3 ---
EXTERNAL_LLM_MODEL_NAME = os.getenv("EXTERNAL_LLM_MODEL_NAME", "gpt-4o-mini")
EXTERNAL_LLM_API_BASE = os.getenv("EXTERNAL_LLM_API_BASE", "https://api.openai.com/v1")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY")

# --- THÔNG TIN CÀI ĐẶT CHUNG CHO AI ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.1))
LLM_MAX_TOKEN_RESPONSE = int(os.getenv("LLM_MAX_TOKEN_RESPONSE", 1024))

# --- THÔNG TIN CHUNG ---
ASSETS_DIR = os.getenv("ASSETS_DIR", "assets")
LOG_DIR = os.getenv("LOG_DIR", "logs")


async def cleanup_inactive_sessions():
    """Tác vụ chạy ngầm định kỳ xóa các session hết hạn để giải phóng RAM"""
    while True:
        try:
            await asyncio.sleep(600)  # Chạy kiểm tra mỗi 10 phút
            now = datetime.now()
            
            # Tìm các session đã quá hạn
            expired_sessions = [
                sid for sid, data in store.items()
                if now - data.last_accessed > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
            ]
            
            # Xóa khỏi bộ nhớ
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
    # Khởi chạy task dọn dẹp khi Server start
    task = asyncio.create_task(cleanup_inactive_sessions())
    yield
    # Hủy task an toàn khi Server tắt
    task.cancel()

# FastAPI setup
app = FastAPI(title="CUSC Chatbot API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_credentials,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
)

os.makedirs(ASSETS_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/chat-client", StaticFiles(directory="templates/chat-client"), name="chat-client")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("templates/favicon.ico")

# 1. Thiết lập Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# tự định nghĩa câu thông báo lỗi
RATE_LIMIT_ERROR_MSG = os.getenv("RATE_LIMIT_ERROR_MSG", "Vui lòng chờ trong giây lát rồi thử lại!")

async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": RATE_LIMIT_ERROR_MSG}
    )

app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

# 2. Thiết lập 2 loại API Key (1 cho Chatbot nhúng ở web, 1 cho Quản trị viên)
PUBLIC_API_KEY = os.getenv("PUBLIC_API_KEY", "QA67V0pMzb2kLlSYQe7mPyJxGlKnbWaKVya2EZNFxtF16Nt29WNYPjQrNShtD6ft")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "XN39rEZfJj9g6wRZuHHm9Qcqk8Zy0H35VAzfDKfi5sGfphZj3nlsNPIVTR5ptuGZ")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_public_key(api_key: str = Security(api_key_header)):
    if api_key != PUBLIC_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Public API Key")
    return api_key

def verify_admin_key(api_key: str = Security(api_key_header)):
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Admin API Key")
    return api_key


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Cấu hình Logger chuyên dụng cho Prompt ---
prompt_logger = logging.getLogger("PromptLogger")
prompt_logger.setLevel(logging.INFO)
# Ngăn log của prompt in rác ra màn hình console của uvicorn
prompt_logger.propagate = False 

os.makedirs(LOG_DIR, exist_ok=True)
prompt_log_file = os.path.join(LOG_DIR, "prompt_debug.log")

# Dùng RotatingFileHandler: Giới hạn file ở mức 2MB (tương đương khoảng 1000-2000 dòng log prompt).
# Khi vượt quá 2MB, nó tự động đổi tên file cũ thành .log.1 và tạo file .log mới.
prompt_handler = RotatingFileHandler(
    prompt_log_file, 
    maxBytes=2 * 1024 * 1024, # 2MB
    backupCount=1,            # Giữ lại 1 file cũ
    encoding="utf-8"
)

# Set format chỉ in ra nội dung tin nhắn, bỏ qua các tiền tố thời gian mặc định của logger
prompt_handler.setFormatter(logging.Formatter('%(message)s'))

# Chống add trùng handler khi reload FastAPI
if not prompt_logger.handlers:
    prompt_logger.addHandler(prompt_handler)

# ========== Lưu user trong phiên ==========
user_session = {}

def get_or_create_monthly_sheet(sheet_prefix: str, headers: list):
    """
    Hàm này kiểm tra xem Tab của tháng hiện tại đã có chưa (VD: Chat_04_2026).
    Nếu chưa có, tự động tạo Tab mới và bôi đậm Header.
    """
    if not gg_sheet:
        return None
        
    current_month = datetime.now().strftime("%m_%Y")
    sheet_name = f"{sheet_prefix}_{current_month}"
    
    with sheet_lock:
        try:
            # 1. Lấy danh sách các tab hiện có
            spreadsheet = gg_sheet.get(spreadsheetId=GG_SPREADSHEET_ID).execute()
            existing_sheets = [s['properties']['title'] for s in spreadsheet.get('sheets', [])]
            
            # 2. Nếu tab chưa tồn tại thì tạo mới
            if sheet_name not in existing_sheets:
                # Yêu cầu tạo Tab mới
                req_create = {"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]}
                gg_sheet.batchUpdate(spreadsheetId=GG_SPREADSHEET_ID, body=req_create).execute()
                
                # Điền Header vào dòng đầu tiên
                gg_sheet.values().update(
                    spreadsheetId=GG_SPREADSHEET_ID,
                    range=f"{sheet_name}!A1",
                    valueInputOption="RAW",
                    body={"values": [headers]}
                ).execute()
                logger.info(f"Đã tự động tạo trang tính mới: {sheet_name}")
                
            return sheet_name
        except Exception as e:
            logger.error(f"GG Sheets Error (Tab Management): {e}")
            return None

def push_to_google_sheets(sheet_prefix: str, headers: list, data_row: list):
    """Hàm chuẩn bị dữ liệu đẩy lên Google Sheets (để các worker pool gọi)"""
    sheet_name = get_or_create_monthly_sheet(sheet_prefix, headers)
    if not sheet_name:
        return
        
    with sheet_lock:
        try:
            gg_sheet.values().append(
                spreadsheetId=GG_SPREADSHEET_ID,
                range=f"{sheet_name}!A:A",
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body={'values': [data_row]}
            ).execute()
        except Exception as e:
            logger.error(f"Lỗi ghi data lên Google Sheets: {e}")
# ========================================================
def setup_embeddings():
    """Setup embeddings with error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBENDDING_MODEL_KWARGS_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"✅ Embeddings model {EMBEDDING_MODEL} loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {str(e)}")
        raise

def setup_vector_store(embeddings):
    """Setup vector store with HYBRID capabilities"""
    try:
        if not os.path.exists(DATABASE_PATH) or not os.listdir(DATABASE_PATH):
            error_msg = f"LỖI NGHIÊM TRỌNG: Không tìm thấy Vector Database tại '{DATABASE_PATH}'. Vui lòng upload tài liệu theo mẫu soạn thảo và chạy cập nhật Vector DB trước khi khởi động Chatbot."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        from qdrant_client import QdrantClient
        client = QdrantClient(path=DATABASE_PATH)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
        logger.info(f"✅ Setup vector store done.")
        return qdrant
    except Exception as e:
        logger.error(f"Failed to setup vector store: {str(e)}")
        raise
        
    
def setup_retriever(qdrant):
    """Setup retriever combining Hybrid Search and Cross-Encoder Reranking"""
    try:
        # 1. Thu thập sâu (Broad Recall) bằng Hybrid Search
        # Qdrant sẽ tự động gộp điểm BM25 và Dense bằng thuật toán RRF (Reciprocal Rank Fusion)
        hybrid_retriever = qdrant.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": SEARCH_KWARGS} # Kéo lên {SEARCH_KWARGS} kết quả thô cực kỳ sát từ khóa và ngữ nghĩa
        )
        
        # 2. Lọc tinh (Fine Ranking) bằng Cross-Encoder
        try:
            from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
            from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            
            cross_encoder = HuggingFaceCrossEncoder(
                model_name=RERANKER_MODEL,
                model_kwargs={'device': RETRIEVER_MODEL_KWARGS_DEVICE}
            )


            compressor = CrossEncoderReranker(model=cross_encoder, top_n=DEFAULT_RERANK_TOP_N)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=hybrid_retriever
            )

            logger.info(f"✅ Setup compression retriever done.")
            return compression_retriever
            
        except Exception as rerank_error:
            logger.error(f"Failed to setup reranker: {str(rerank_error)}")
            hybrid_retriever.search_kwargs["k"] = DEFAULT_RERANK_TOP_N 
            return hybrid_retriever
    except Exception as e:
        logger.error(f"Failed to setup retriever: {str(e)}")
        raise
        
class PromptLoggerCallback(BaseCallbackHandler):
    """Callback ghi log prompt an toàn luồng (Thread-safe)"""
    def on_llm_start(self, serialized, prompts, **kwargs):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Chỉ tạo string, KHÔNG thao tác file I/O ở đây
        new_log = (
            f"\n{'='*50}\n"
            f"[{timestamp}] CHÍNH XÁC PROMPT ĐƯỢC GỬI VÀO LLM:\n"
            f"{'-'*50}\n"
            f"{prompts[0]}\n"
            f"{'='*50}\n"
        )
        
        # Giao việc ghi file cho Thread-safe Logger
        prompt_logger.info(new_log)
            
def setup_llm():
    """Setup LLM with fallback mechanism (External -> Local)"""
    try:
        # 1. Khởi tạo Local LLM (Luôn có để dự phòng và dùng cho các task phụ)
        local_llm = ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            openai_api_base=LLM_API_BASE, 
            openai_api_key=LLM_API_KEY,                     
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_CHAT_RESPONSE_TOKEN,
            top_p=LLM_TOP_P,
            callbacks=[PromptLoggerCallback()]
        )
        
        # 2. Khởi tạo External LLM (Gemini)
        if EXTERNAL_API_KEY:
            external_llm = ChatOpenAI(
                model_name=EXTERNAL_LLM_MODEL_NAME,
                openai_api_base=EXTERNAL_LLM_API_BASE,
                openai_api_key=EXTERNAL_API_KEY,
                temperature=LLM_TEMPERATURE,
                max_tokens=MAX_CHAT_RESPONSE_TOKEN,
                streaming=True,      
                stream_usage=False,  
                callbacks=[PromptLoggerCallback()]
            )
            
            # Sử dụng Gemini làm chính, vLLM làm dự phòng
            main_llm = external_llm.with_fallbacks([local_llm])
            logger.info(f"🌟 RAG is using Gemini ({EXTERNAL_LLM_MODEL_NAME}) with Fallback vLLM.")
        else:
            main_llm = local_llm
            logger.info("🏠 It runs entirely using Local LLM (vLLM).")
            
        # Trả về cả 2 object
        return main_llm, local_llm
    except Exception as e:
        logger.error(f"Failed to setup LLM: {str(e)}")
        raise

def log_qa_to_file(question: str, answer: str):
    """Ghi log Câu hỏi của User và Câu trả lời của Bot ra file logs/log.txt"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "log.txt")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Mở file ở chế độ "a" (append - ghi nối tiếp)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"🧑 User : {question}\n")
        f.write(f"🤖 Bot  : {answer}\n")
        f.write("-" * 60 + "\n")        

def create_rag_chain(main_llm, local_llm, retriever):
    try:
        # ====================================================================
        # PROMPT VIẾT LẠI CÂU HỎI
        # ====================================================================
        contextualize_q_system_prompt = """
Bạn là một công cụ xử lý ngôn ngữ tự nhiên tự động.
NHIỆM VỤ ĐỘC QUYỀN CỦA BẠN: Đọc "Lịch sử trò chuyện" (nếu có) và "Câu hỏi mới nhất" của người dùng. Sau đó, CHỈ xuất ra một câu hỏi duy nhất đã được làm rõ ngữ cảnh.

CÁC QUY TẮC CỐT LÕI (PHẢI TUÂN THỦ TUYỆT ĐỐI):
1. KHÔNG BAO GIỜ ĐƯỢC TRẢ LỜI CÂU HỎI. Không giải thích, không thêm kiến thức cá nhân, không cung cấp thông tin.
2. NẾU câu hỏi mới nhất đã rõ ràng, có đầy đủ chủ ngữ và đối tượng, BẠN PHẢI GIỮ NGUYÊN CÂU HỎI ĐÓ.
3. CHỈ khi câu hỏi bị thiếu chủ ngữ, dùng đại từ (nó, cái đó, ngành này) hoặc mang tính nối tiếp, bạn mới được dùng thông tin từ Lịch sử trò chuyện để điền vào cho rõ nghĩa.
4. BẮT BUỘC phải bọc kết quả cuối cùng trong thẻ <rewritten_question> và </rewritten_question>.

CÁC VÍ DỤ:

[Ví dụ 1 - Câu hỏi nối tiếp cần làm rõ]
Lịch sử: User: Ngành CNTT học gì? - AI: Ngành này học lập trình.
Câu hỏi mới: Học phí bao nhiêu?
Output: <rewritten_question>Học phí ngành Công nghệ thông tin là bao nhiêu?</rewritten_question>

[Ví dụ 2 - Câu hỏi ĐÃ RÕ RÀNG (Giữ nguyên, TUYỆT ĐỐI KHÔNG TRẢ LỜI)]
Lịch sử: Trống
Câu hỏi mới: Hồ sơ xét tuyển ĐGNL gồm những gì?
Output: <rewritten_question>Hồ sơ xét tuyển ĐGNL gồm những gì?</rewritten_question>

[Ví dụ 3 - Câu hỏi ĐÃ RÕ RÀNG (Giữ nguyên, TUYỆT ĐỐI KHÔNG TRẢ LỜI)]
Lịch sử: Trống
Câu hỏi mới: Chuẩn đầu ra ngành CNTT gồm những kỹ năng gì?
Output: <rewritten_question>Chuẩn đầu ra ngành Công nghệ thông tin gồm những kỹ năng gì?</rewritten_question>
"""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # LLM Chain chỉ để tạo ra String câu hỏi mới
        rewrite_chain = contextualize_q_prompt | local_llm | (lambda x: x.content)

        # HÀM XỬ LÝ TRUY VẤN KÉP (MULTI-QUERY)
        def dual_retrieval(inputs: dict) -> list[Document]:
            original_q = inputs["input"]
            chat_history = inputs.get("chat_history", [])
            
            # Nếu không có lịch sử, chỉ search câu gốc
            if not chat_history:
                return retriever.invoke(original_q)
            
            # Nếu có lịch sử, tạo câu hỏi mới
            rewritten_q = rewrite_chain.invoke(inputs)
            logger.info(f"🔄 Original Q: {original_q}")
            logger.info(f"🔄 Rewritten Q: {rewritten_q}")
            
            # Lấy vector thô (Chưa qua Rerank) từ cả 2 câu
            # Lưu ý: retriever.base_retriever là truy cập thẳng vào Hybrid Search bỏ qua Compressor
            base_retriever = retriever.base_retriever 
            
            docs_original = base_retriever.invoke(original_q)
            docs_rewritten = base_retriever.invoke(rewritten_q)
            
            # Gộp và loại bỏ các chunk trùng lặp (dựa vào page_content)
            unique_docs = {}
            for doc in docs_original + docs_rewritten:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
                    
            combined_docs = list(unique_docs.values())
            
            # Chạy Reranker lần cuối trên tập tổng hợp, so với CÂU HỎI VIẾT LẠI (hoặc câu gốc tùy bạn)
            # retriever.base_compressor chính là CrossEncoderReranker bạn đã định nghĩa
            final_docs = retriever.base_compressor.compress_documents(
                documents=combined_docs, 
                query=rewritten_q # Dùng câu đã có ngữ cảnh để chấm điểm
            )
            
            return final_docs
            
        # ====================================================================
        # PROMPT CHÍNH TRẢ LỜI CÂU HỎI RAG
        # ====================================================================
        qa_system_prompt = """
<role>BẠN LÀ CHUYÊN VIÊN TƯ VẤN TUYỂN SINH CỦA ĐẠI HỌC BẠC LIÊU (BLU)</role>
<core-mission>Nhiệm vụ của bạn là tư vấn và giải đáp thắc mắc về TUYỂN SINH dựa trên thông tin được cung cấp trong <context>.</core-mission>
<rules>
1. TUYỆT ĐỐI CHỈ sử dụng thông tin từ <context> để trả lời. Không sử dụng kiến thức tự có bên ngoài.
2. Nếu <context> trống hoặc không chứa đủ thông tin để trả lời, BẮT BUỘC phản hồi: "Hiện tại mình chưa có thông tin chính xác về vấn đề này. Bạn có thể liên hệ hotline hoặc fanpage của trường để được hỗ trợ nhé." - Tuyệt đối không tự suy luận.
3. TỪ CHỐI lịch sự các câu hỏi không liên quan đến chủ đề tuyển sinh của BLU.
</rules>

<context-format-definition>
1. Hình ảnh kèm theo tài liệu có dạng: ![Image](/assets/...) >**[HÌNH TÓM TẮT]: Hình ảnh ...**
2. Bảng biểu từ <context> có dạng Markdown chuẩn.
3. Cấu trúc của một CHUNK trong <context> bắt đầu từ dòng chữ được bao bọc bằng một cặp ngoặc vuông.
</context-format-definition>

<context>
{context}
</context>

<answer-format-rules>
1. Trình bày bằng định dạng Markdown (md), KHÔNG bao gồm bất kỳ thẻ XML nào trong đầu ra.
2. Trả lời CHÍNH XÁC, đúng trọng tâm. Không giải thích dông dài nếu không được yêu cầu.
3. Giọng điệu tự nhiên, thân thiện. Luôn xưng hô là "Mình" và gọi người dùng là "Bạn".
4. KHÔNG lặp lại câu hỏi của người dùng,KHÔNG đưa nguồn vào câu trả lời.
5. Công thức toán/tổ hợp môn (nếu có) phải viết bằng văn bản thuần túy kết hợp Markdown.
</answer-format-rules>
"""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"), 
        ])
        
        # Lắp ráp luồng RAG hoàn chỉnh
        rag_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(dual_retrieval) | (lambda docs: "\n\n".join(d.page_content for d in docs))
            )
            | qa_prompt
            | main_llm
            | StrOutputParser()
        )

        logger.info("✅ RAG chain created successfully with Dual-Query Reranking")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {str(e)}")
        raise

def clean_response(response: str) -> str:
    """Clean response text: remove markdown formatting, trim whitespace, and remove duplicate URLs."""
    if not response:
        return ""
    
    cleaned = response;

    cleaned = re.sub(r'<context>', 'dữ liệu', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'</context>', '', cleaned, flags=re.IGNORECASE)

    url_pattern = re.compile(r'https?://[^\s\[\]()]+')
    urls_with_positions = []
    for match in url_pattern.finditer(cleaned):
        urls_with_positions.append((match.start(), match.end(), match.group()))
    
    if not urls_with_positions:
        return ' '.join(cleaned.split()).strip()
    
    seen_urls = set()
    result_parts = []
    last_end = 0
    
    for start, end, url in urls_with_positions:
        result_parts.append(cleaned[last_end:start])
        if url not in seen_urls:
            result_parts.append(url)
            seen_urls.add(url)
        last_end = end
    
    result = ''.join(result_parts)
    result = ' '.join(result.split()).strip()
    return result

# Tạo một Khóa (Lock) để chống lỗi SSL khi có nhiều user chat cùng lúc
sheet_lock = threading.Lock()
def setup_ggsheet():
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = service_account.Credentials.from_service_account_file(GG_CRED_FILE, scopes=scopes)
        gg_service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
        gg_sheet = gg_service.spreadsheets()
        logger.info("✅ Google Sheets API kết nối thành công!")
        return gg_sheet
    except Exception as e:
        logger.error(f"Lỗi kết nối Google Sheets: {e}")
        raise

# Initialize components with error handling
try:
    logger.info("Initializing application components...")
    embeddings = setup_embeddings()
    qdrant = setup_vector_store(embeddings)
    retriever = setup_retriever(qdrant)
    main_llm, local_llm = setup_llm()
    rag_chain = create_rag_chain(main_llm, local_llm, retriever)
    gg_sheet = setup_ggsheet()
    
    logger.info("✅ All components initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize components. Exiting: {str(e)}")
    raise

# Chat history management
class SessionData:
    def __init__(self):
        self.history = ChatMessageHistory()
        self.last_accessed = datetime.now()

# Đổi kiểu dữ liệu của store sang SessionData
store: Dict[str, SessionData] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Lấy hoặc tạo session, đồng thời cập nhật thời gian truy cập cuối"""
    if session_id not in store:
        store[session_id] = SessionData()
    
    session_data = store[session_id]
    session_data.last_accessed = datetime.now() # Cập nhật mốc tương tác mới nhất
    
    # Cắt tỉa lịch sử theo MAX_CHAT_HISTORY
    if len(session_data.history.messages) > MAX_CHAT_HISTORY:
        session_data.history.messages = session_data.history.messages[-MAX_CHAT_HISTORY:]
    
    return session_data.history

# Create conversational RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str = Field(..., max_length=100)
    chat_request: str = Field(..., max_length=500) # Tối đa 500 ký tự
    metadata_filter: Optional[Dict[str, str]] = None
    user_info: Optional[dict] = None

class ChatResponse(BaseModel):
    chat_response: str
    success: bool = True
    error_message: Optional[str] = None

class UserInfo(BaseModel):
    name: str
    phone: str
    email: str
    address: str

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Store for user information
user_store: Dict[str, Dict] = {}

@app.post("/chat_stream")
@limiter.limit("5/minute")
async def chat_stream_endpoint(
    request: Request,
    chat_req: ChatRequest,
    api_key: str = Depends(verify_public_key)
):
    start_time = time.time()
    try:
        logger.info(f"Processing streaming chat request from session {chat_req.session_id}")
        
        if not chat_req.chat_request.strip():
            raise ValueError("Empty chat request")
        if not chat_req.session_id.strip():
            raise ValueError("Empty session ID")
        
        original_question = chat_req.chat_request.strip()
        logger.info(f"Original Question: {original_question}")

        SMALL_TALK_KEYWORDS = ["đói", "ăn gì", "buồn ngủ", "mệt", "chán", "vui", 
                               "trời mưa", "trời nắng", "trời lạnh", "trời nóng",
                               "hôm nay ăn gì", "ngủ ngon", "mệt mỏi", "hết pin"]
        def is_small_talk(message: str) -> bool:
            return any(keyword in message.lower() for keyword in SMALL_TALK_KEYWORDS)

        if is_small_talk(original_question):
            friendly_response = "Nghe có vẻ bạn đang cần thư giãn một chút! Nếu bạn cần tôi tư vấn về các ngành ở BLU, cứ hỏi nhé."
            async def small_talk_stream():
                for word in friendly_response.split(" "):
                    yield word + " "
                    await asyncio.sleep(0.05)
            return StreamingResponse(small_talk_stream(), media_type="text/plain")

        history = get_session_history(chat_req.session_id)
        async def generate_chat_stream():
            full_answer = ""
            try:
                async for chunk in conversational_rag_chain.astream(
                    {"input": original_question},
                    config={"configurable": {"session_id": chat_req.session_id}},
                ):
                    if isinstance(chunk, str) and chunk:
                        full_answer += chunk
                        yield chunk
                        await asyncio.sleep(0)

                cleaned_response = clean_response(full_answer.strip())
                if not cleaned_response:
                    cleaned_response = "Xin lỗi, tôi không thể tạo ra câu trả lời cho câu hỏi này."
                
                formatted_response = cleaned_response
                logger.info(f"Final Response (streaming finished). Time: {time.time() - start_time:.2f}s")

                # ==== LƯU LỊCH SỬ LÊN GOOGLE SHEETS ====
                try:
                    user = chat_req.user_info or {}  
                    name = user.get("name", "Khách")
                    email = user.get("email", "N/A")
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    row_data = [timestamp, chat_req.session_id, name, email, original_question, formatted_response]
                    headers = ["Thời gian", "Session ID", "Tên", "Email", "Câu hỏi", "Câu trả lời AI"]
                    
                    # Ném tác vụ ghi sheet sang Threadpool của asyncio
                    # Không chặn luồng chính, tự động dọn dẹp khi xong
                    await asyncio.to_thread(push_to_google_sheets, "Chat", headers, row_data)
                        
                    logger.info("✅ Đã ghi log Chat lên Google Sheets.")
                except Exception as log_error:
                    logger.warning(f"Lỗi hệ thống ghi log: {str(log_error)}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"\n[Lỗi Hệ Thống: {str(e)}]"

        return StreamingResponse(generate_chat_stream(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error initializing chat stream: {e}")
        async def error_stream():
            yield "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại sau."
        return StreamingResponse(error_stream(), media_type="text/plain")

@app.post("/user_info")
@limiter.limit("3/minute")
async def save_user_info(
    request: Request, 
    user_info: UserInfo,
    background_tasks: BackgroundTasks
):
    if not user_info.email:
        raise HTTPException(status_code=400, detail="Email không được để trống")

    user_session[user_info.email] = user_info.model_dump()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row_data = [timestamp, user_info.name, user_info.phone, user_info.email, user_info.address]
    headers = ["Thời gian", "Họ Tên", "Số điện thoại", "Email", "Địa chỉ"]

    # ==== ĐẨY LÊN GOOGLE SHEETS  ====
    # BackgroundTasks sẽ tự động đưa tác vụ này vào threadpool để chạy ngầm
    background_tasks.add_task(push_to_google_sheets, "User", headers, row_data)
    
    return {"message": "Lưu thông tin user thành công!", "email": user_info.email}

# Khởi tạo template folder
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def home(request: Request):
    """Render trang chatbot HTML từ thư mục templates"""
    return templates.TemplateResponse(request=request, name="chat.html")

@app.get("/debug_search")
@limiter.limit("10/minute")
async def debug_search(request: Request, q: str, admin_key: str = Depends(verify_admin_key)):
    """API kiểm tra xem chunk nào được nạp vào LLM"""
    docs = qdrant.similarity_search(q, k=DEFAULT_RERANK_TOP_N)
    return {
        "query": q,
        "context_chunks_sent_to_llm": [doc.page_content for doc in docs]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Chatbot server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
