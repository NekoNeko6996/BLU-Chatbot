import os
import re
import shutil
import unicodedata
import subprocess
import sys
import base64
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import uvicorn
import asyncio
from dotenv import load_dotenv
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from contextlib import asynccontextmanager
from rag_core import process_and_upsert_file
from typing import List
from fastapi.exceptions import RequestValidationError
from rag_core import DENSE_MODEL, SPARSE_MODEL, COLLECTION_NAME, GLOBAL_QDRANT_CLIENT
from langchain_qdrant import QdrantVectorStore, RetrievalMode
import json

load_dotenv(override=True)

import subprocess
import sys

# client config
FRAGMENT_FILE = Path("templates/chat-client/top-chat-message-fragment.html")

# env config
ENV_FILE = Path(".env")

# Khởi tạo thư mục từ ENV
DOC_DIR = Path(os.getenv("DOC_DIR", "documentBL"))
RAW_DIR = Path(os.getenv("RAW_DIR", "documentBL_original"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets"))
HISTORY_FILE = LOG_DIR / "upload_history.json"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blu")

LOG_FILE = LOG_DIR / "history.xlsx"
PROMPT_LOG_FILE = LOG_DIR / "prompt_debug.log"

# Cấu hình đường dẫn Database & Backup
DB_DIR = Path(os.getenv("DATABASE_PATH", "database/collection"))
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", "database/backup/blu"))

# Tự động kiểm tra và tạo thư mục nếu chưa tồn tại
for d in [DOC_DIR, RAW_DIR, LOG_DIR, ASSETS_DIR, DB_DIR, BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cấu hình Gemini Vision từ .env
ENABLE_GEMINI_VISION = os.getenv("ENABLE_GEMINI_VISION", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent")

# Quản lý tiến trình và đường dẫn Log
PROCESSES = {
    "chatbot": None,
    "updater": None
}
LOG_FILES = {
    "chatbot": LOG_DIR / "chatbot_server_log.txt",
    "updater": LOG_DIR / "updater_log.txt"
}

# Tạo sẵn file log nếu chưa tồn tại
for path in LOG_FILES.values():
    path.touch(exist_ok=True)

# --- THƯ VIỆN CỦA DOCLING ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import ImageRefMode

# 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup trước khi app chạy (có thể để trống)
    yield
    
    # Cleanup khi app chuẩn bị tắt (Nhấn Ctrl+C)
    print("\n[HỆ THỐNG] Đang dọn dẹp các tiến trình con trước khi tắt máy chủ...")
    for service_name, process in PROCESSES.items():
        if process is not None and process.returncode is None:
            print(f"- Đang ép dừng tiến trình: {service_name}")
            try:
                process.terminate()
                # Chờ một chút để tiến trình con kịp tắt an toàn
                await asyncio.wait_for(process.wait(), timeout=3.0) 
            except Exception as e:
                print(f"  > Không thể tắt {service_name}: {e}")
                # Ép giết (kill) nếu terminate thất bại
                try:
                    process.kill()
                except:
                    pass

app = FastAPI(title="Admin Document Manager", lifespan=lifespan)

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
templates = Jinja2Templates(directory="templates")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("\n" + "="*50)
    print("❌ LỖI 422 UNPROCESSABLE CONTENT ❌")
    print("Dữ liệu gửi lên không đúng định dạng mà FastAPI yêu cầu.")
    print("Chi tiết lỗi từ Pydantic:")
    for error in exc.errors():
        print(f" - Trường (Field): {error.get('loc')}")
        print(f" - Lỗi (Msg): {error.get('msg')}")
    print("="*50 + "\n")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Sai định dạng dữ liệu gửi lên"}
    )

# CẤU HÌNH DOCLING ĐỂ TRÍCH XUẤT ẢNH
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# CẤU TRÚC NHẬN DỮ LIỆU
class MDContent(BaseModel):
    folder: str
    filename: str
    content: str

class ServiceAction(BaseModel):
    action: str
    
class BackupRequest(BaseModel):
    name: str
    description: str
    
class FragmentData(BaseModel):
    content: str
    
class EnvData(BaseModel):
    content: str

class TestSearchQuery(BaseModel):
    query: str
    top_k: int = 5
    
# --- UTILS ---
def slugify(text: str) -> str:
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def get_directory_tree(path: Path, relative_to: Path):
    tree = []
    for item in sorted(path.iterdir()):
        if item.is_dir() and item.name != "assets": 
            rel_path = str(item.relative_to(relative_to)).replace("\\", "/")
            tree.append({
                "id": rel_path,
                "text": item.name,
                "children": get_directory_tree(item, relative_to),
                "state": {"opened": True}
            })
    return tree

def log_action(action: str, filename: str, folder: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[now, action, filename, folder]], 
                            columns=["Thời gian", "Hành động", "Tên file", "Thư mục"])
    if LOG_FILE.exists():
        df = pd.read_excel(LOG_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_excel(LOG_FILE, index=False)

# --- AI IMAGE CONVERT TO TEXT
def analyze_image_with_gemini(b64_data, ext):
    fallback_text = "Hình ảnh/Sơ đồ minh họa tài liệu" 
    
    # 1. Kiểm tra công tắc từ .env. Nếu tắt (false) -> Trả về Fallback luôn, không tốn tiền API
    if not ENABLE_GEMINI_VISION:
        return fallback_text
        
    # 2. Nếu bật (true) nhưng quên điền API Key -> Báo lỗi console và trả về Fallback
    if not GEMINI_API_KEY:
        print("CẢNH BÁO: Đã bật ENABLE_GEMINI_VISION nhưng chưa điền GEMINI_API_KEY trong file .env!")
        return fallback_text

    # 3. Tiến hành gọi API
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    mime_type = f"image/{ext}" if ext.lower() in ['jpeg', 'jpg', 'png'] else "image/jpeg"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": "Mô tả ngắn gọn nội dung của biểu đồ/hình ảnh này trong tối đa 2 câu để làm chú thích tài liệu RAG."},
                {"inline_data": {"mime_type": mime_type, "data": b64_data}}
            ]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status() # Bắt lỗi HTTP (400, 401, 500...)
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"Lỗi gọi Gemini API: {e}")
        return fallback_text


# --- API ENDPOINTS ---
@app.get("/upload-history", response_class=HTMLResponse)
async def upload_history_page(request: Request):
    # Sửa lỗi unhashable dict cho bản FastAPI mới
    return templates.TemplateResponse(request=request, name="upload-history.html")

@app.get("/api/upload-history")
async def get_upload_history():
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/test-search", response_class=HTMLResponse)
async def test_search_page(request: Request):
    return templates.TemplateResponse(request=request, name="test-search.html")
    
@app.post("/api/test-search")
async def test_vector_search(data: TestSearchQuery):
    try:
        # Khởi tạo lại kết nối từ collection có sẵn
        vectorstore = QdrantVectorStore(
            client=GLOBAL_QDRANT_CLIENT,
            collection_name=COLLECTION_NAME,
            embedding=DENSE_MODEL,
            sparse_embedding=SPARSE_MODEL,
            retrieval_mode=RetrievalMode.HYBRID
        )

        # Thực hiện truy vấn
        results = vectorstore.similarity_search_with_score(data.query, k=data.top_k)

        # Trích xuất và định dạng kết quả
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "score": round(score, 4),
                "metadata": doc.metadata,
                "content": doc.page_content
            })

        return {"query": data.query, "results": formatted_results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def admin_ui(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html")

@app.get("/api/tree")
async def fetch_tree():
    return JSONResponse(content=[{
        "id": "root",
        "text": "documentBL (Gốc)",
        "state": {"opened": True, "selected": True},
        "children": get_directory_tree(DOC_DIR, DOC_DIR)
    }])

@app.post("/api/folder")
async def create_folder(parent: str = Form(...), name: str = Form(...)):
    slug_name = slugify(name)
    if not slug_name: raise HTTPException(status_code=400, detail="Tên thư mục không hợp lệ")

    doc_path = DOC_DIR / slug_name if parent == "root" else DOC_DIR / parent / slug_name
    raw_path = RAW_DIR / slug_name if parent == "root" else RAW_DIR / parent / slug_name

    if doc_path.exists(): raise HTTPException(status_code=400, detail="Thư mục đã tồn tại")

    doc_path.mkdir(parents=True, exist_ok=True)
    raw_path.mkdir(parents=True, exist_ok=True)

    log_action("Tạo thư mục", slug_name, parent)
    return {"message": "Tạo thư mục thành công", "path": str(doc_path)}

@app.delete("/api/folder")
async def delete_folder(folder: str):
    if not folder or folder == "root": raise HTTPException(status_code=400, detail="Không thể xóa thư mục gốc")
    
    target_doc_dir = DOC_DIR / folder
    target_raw_dir = RAW_DIR / folder

    if target_raw_dir.exists():
        for file in target_raw_dir.iterdir():
            if file.is_file():
                slug_name = slugify(file.stem)
                asset_prefix = f"img_{slug_name[:10]}_"
                for asset in ASSETS_DIR.glob(f"{asset_prefix}*"):
                    asset.unlink(missing_ok=True)

    deleted = False
    if target_raw_dir.exists():
        shutil.rmtree(target_raw_dir)
        deleted = True
    if target_doc_dir.exists():
        shutil.rmtree(target_doc_dir)
        deleted = True
        
    if not deleted: raise HTTPException(status_code=404, detail="Thư mục không tồn tại")
    log_action("Xóa thư mục", "", folder)
    return {"message": "Đã xóa thư mục và toàn bộ tài nguyên bên trong"}

@app.post("/api/upload")
# QUAY VỀ SIGNATURE GỐC CỦA BẠN: Nhận từng file một, KHÔNG dùng List, KHÔNG bị lỗi 422
async def upload_file(folder: str = Form(...), file: UploadFile = File(...)):
    target_doc_dir = DOC_DIR if folder == "root" else DOC_DIR / folder
    target_raw_dir = RAW_DIR if folder == "root" else RAW_DIR / folder

    if not target_doc_dir.exists() or not target_raw_dir.exists():
        raise HTTPException(status_code=400, detail="Thư mục đích không tồn tại")

    # 1. Lưu file gốc
    raw_file_path = target_raw_dir / file.filename
    with open(raw_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_stem = Path(file.filename).stem
    md_file_path = target_doc_dir / f"{slugify(file_stem)}.md"

    try:
        # 2. Convert bằng Docling (GIỮ NGUYÊN LOGIC GỐC CỦA BẠN)
        result = doc_converter.convert(str(raw_file_path))
        md_content = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        
        def process_and_replace_image(match):
            alt_text = match.group(1) or "image"
            ext = match.group(2) 
            b64_data = match.group(3)
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            img_name = f"img_{slugify(file_stem)[:10]}_{timestamp}_{uuid.uuid4().hex[:8]}.{ext}"
            img_full_path = ASSETS_DIR / img_name
            
            with open(img_full_path, "wb") as fh:
                fh.write(base64.b64decode(b64_data))
                
            caption = analyze_image_with_gemini(b64_data, ext)
            return f"![{alt_text}](/assets/{img_name})\n\n>**[HÌNH TÓM TẮT]: {caption}**\n\n"

        pattern = r'!\[([^\]]*)\]\(data:image/([a-zA-Z0-9]+);base64,([A-Za-z0-9+/=]+)\)'
        md_content = re.sub(pattern, process_and_replace_image, md_content)
        
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        # 3. TÍCH HỢP RAG MỚI (UPDATE VECTOR TỰ ĐỘNG)
        from rag_core import process_and_upsert_file
        new_version = process_and_upsert_file(str(md_file_path))
            
    except Exception as e:
        if raw_file_path.exists():
            raw_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

    log_action("Upload & Indexing", file.filename, f"Version {new_version}")
    return {"message": "Upload và xử lý thành công", "version": new_version}

@app.get("/api/files")
async def list_files(folder: str):
    target_raw_dir = RAW_DIR if folder == "root" else RAW_DIR / folder
    target_doc_dir = DOC_DIR if folder == "root" else DOC_DIR / folder
    
    raw_files = []
    md_files = []
    
    if target_raw_dir.exists():
        raw_files = [f.name for f in target_raw_dir.iterdir() if f.is_file()]
                
    if target_doc_dir.exists():
        for f in target_doc_dir.iterdir():
            if f.is_file():
                if f.suffix.lower() == '.md':
                    md_files.append(f.name)
                else:
                    raw_files.append(f.name) # Bắt cả các file .txt cũ
                    
    return JSONResponse(content={"raw": list(set(raw_files)), "md": md_files})

@app.get("/api/file/content")
async def get_md_content(folder: str, filename: str):
    target_doc_dir = DOC_DIR if folder == "root" else DOC_DIR / folder
    md_path = target_doc_dir / filename
    
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại")
        
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    slug_name = slugify(Path(filename).stem)
    asset_prefix = f"img_{slug_name[:10]}_"
    images = []
    
    if ASSETS_DIR.exists():
        for asset in ASSETS_DIR.glob(f"{asset_prefix}*"):
            images.append(f"/assets/{asset.name}")
            
    return JSONResponse(content={"content": content, "images": images})

@app.put("/api/file/content")
async def save_md_content(data: MDContent):
    target_doc_dir = DOC_DIR if data.folder == "root" else DOC_DIR / data.folder
    md_path = target_doc_dir / data.filename
    
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại")
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(data.content)
        
    # --- THÊM PHẦN NÀY ĐỂ AUTO UPDATE KHI SỬA FILE ---
    try:
        from rag_core import process_and_upsert_file
        new_version = process_and_upsert_file(str(md_path))
        log_action("Sửa file MD & Đồng bộ Vector", data.filename, f"Version {new_version}")
    except Exception as e:
        print(f"Lỗi cập nhật Vector: {e}")
        # Vẫn cho qua vì file vật lý đã lưu thành công
    # ---------------------------------------------------

    return {"message": "Lưu file và đồng bộ Vector thành công!"}

@app.get("/api/download")
async def download_file(folder: str, filename: str):
    target_raw_dir = RAW_DIR if folder == "root" else RAW_DIR / folder
    target_doc_dir = DOC_DIR if folder == "root" else DOC_DIR / folder
    
    if (target_raw_dir / filename).exists(): return FileResponse(path=(target_raw_dir / filename), filename=filename)
    elif (target_doc_dir / filename).exists(): return FileResponse(path=(target_doc_dir / filename), filename=filename)
    else: raise HTTPException(status_code=404, detail="File không tồn tại")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    # Cập nhật lại API xóa file để dùng hàm filter chính xác
    try:
        client = GLOBAL_QDRANT_CLIENT
        
        # Xóa trong Qdrant bằng metadata.source (tên file chính xác)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=filename))]
            )
        )
        
        # Xóa file vật lý
        file_path = DOC_DIR / filename
        if file_path.exists():
            file_path.unlink()
            
        log_action("Xóa file", filename, "Success")
        return {"message": "Xóa thành công"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-vector")
async def update_vector():
    """Chạy script update_new_data.py và stream output trực tiếp về frontend"""
    async def generate():
        try:
            # Dùng -u (unbuffered) để ép Python xả log ra ngay lập tức, không gom cục
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "update_new_data.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                # Trả từng dòng log về client
                yield line.decode('utf-8')
            
            await process.wait()
            
            if process.returncode == 0:
                log_action("Cập nhật Vector", "update_new_data.py", "Hệ thống")
                yield "\nKẾT THÚC: CẬP NHẬT VECTOR THÀNH CÔNG!\n"
            else:
                log_action("Lỗi Cập nhật Vector", "update_new_data.py", "Hệ thống")
                yield f"\nLỖI: TIẾN TRÌNH KẾT THÚC VỚI MÃ LỖI {process.returncode}\n"
                
        except Exception as e:
            yield f"\nLỖI HỆ THỐNG: {str(e)}\n"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/api/history")
async def get_history():
    if not LOG_FILE.exists(): return JSONResponse(content=[])
    try:
        df = pd.read_excel(LOG_FILE)
        df = df.fillna("").astype(str)
        return JSONResponse(content=df.to_dict(orient="records"))
    except: return JSONResponse(content=[])


# TREAM LOG CONSOLE
@app.get("/admin/console", response_class=HTMLResponse)
async def get_admin_console(request: Request):
    return templates.TemplateResponse(request=request, name="admin_chatbot_console.html")

# -- Task chạy ngầm để đọc log stream và tự động xoay vòng (Rotate Log) ---
async def service_logger_task(service_name: str, process: asyncio.subprocess.Process, script: str):
    log_path = LOG_FILES[service_name]
    
    def trim_log_file():
        """Cắt file log giữ lại đúng 1000 dòng mới nhất bằng file tạm để tránh xung đột đọc/ghi."""
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > 1000:
                temp_path = log_path.with_suffix('.tmp')
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-1000:])
                temp_path.replace(log_path)  # Ghi đè nguyên tử (Atomic replace) an toàn

    # Cắt file dọn dẹp trước khi khởi động
    trim_log_file()
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> KHỞI ĐỘNG {script} <<<\n")

    line_count = 0
    while True:
        line = await process.stdout.readline()
        if not line:
            break  # Thoát vòng lặp khi tiến trình bị dừng
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line.decode('utf-8', errors='replace'))
        
        line_count += 1
        # Cứ mỗi khi ghi được 50 dòng thì kiểm tra và cắt log 1 lần (Tối ưu I/O đĩa cứng)
        if line_count >= 50:
            trim_log_file()
            line_count = 0
            
    # Khi tiến trình bị dừng hoặc hoàn tất
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> ĐÃ DỪNG {script} <<<\n")
    trim_log_file()

@app.post("/api/service/{service_name}")
async def manage_service(service_name: str, payload: ServiceAction):
    global PROCESSES
    if service_name not in PROCESSES:
        raise HTTPException(status_code=404, detail="Service không tồn tại")

    script = "serverBL.py" if service_name == "chatbot" else "update_new_data.py"

    if payload.action == "start":
        # Với Asyncio Process, kiểm tra trạng thái bằng "returncode" thay vì "poll()"
        if PROCESSES[service_name] and PROCESSES[service_name].returncode is None:
            return {"message": "Service đang chạy"}

        # Mở subprocess bằng cơ chế Bất đồng bộ (Asyncio)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        PROCESSES[service_name] = process
        
        # Kích hoạt task đọc log ngầm để xử lý xả log và xoay vòng
        asyncio.create_task(service_logger_task(service_name, process, script))

        log_action("Khởi động", script, "Console")
        return {"message": f"Đã khởi động {service_name}"}

    elif payload.action == "stop":
        if PROCESSES[service_name] and PROCESSES[service_name].returncode is None:
            PROCESSES[service_name].terminate()
            # Task ngầm `service_logger_task` sẽ tự động bắt được tín hiệu dừng này
            # và ghi dòng ">>> ĐÃ DỪNG <<<", bạn không cần ghi file thủ công ở đây nữa.
            
            log_action("Dừng", script, "Console")
            return {"message": f"Đã dừng {service_name}"}
        return {"message": "Service không chạy"}

@app.get("/api/service/{service_name}/status")
async def service_status(service_name: str):
    if service_name not in PROCESSES:
        raise HTTPException(status_code=404)
    # Tương tự, kiểm tra trạng thái hoạt động bằng returncode
    is_running = PROCESSES[service_name] is not None and PROCESSES[service_name].returncode is None
    return {"running": is_running}


@app.get("/api/service/{service_name}/logs")
async def service_logs(service_name: str):
    log_path = LOG_FILES.get(service_name)
    if not log_path or not log_path.exists():
        return {"logs": "Chưa có dữ liệu log."}
    
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Vẫn trả về tối đa 1000 dòng để đề phòng (phòng hờ lúc đang chuẩn bị cắt file)
        return {"logs": "".join(lines[-1000:])}


@app.post("/api/backup")
async def create_backup(payload: BackupRequest):
    backup_path = BACKUP_DIR / payload.name
    if backup_path.exists():
        raise HTTPException(status_code=400, detail="Tên bản sao lưu đã tồn tại. Vui lòng chọn tên khác.")
    
    try:
        # 1. Gọi Qdrant API để tạo Snapshot trong container Docker
        create_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots"
        res = requests.post(create_url)
        res.raise_for_status()
        snapshot_name = res.json()["result"]["name"]
        
        # 2. Tải file snapshot (.snapshot) từ Docker về máy chủ Host
        download_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}"
        snap_res = requests.get(download_url, stream=True)
        snap_res.raise_for_status()
        
        backup_path.mkdir(parents=True, exist_ok=True)
        snapshot_file_path = backup_path / snapshot_name
        
        with open(snapshot_file_path, "wb") as f:
            for chunk in snap_res.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # 3. Ra lệnh xóa Snapshot tạm trên Docker để tránh đầy ổ cứng ảo
        requests.delete(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}")
        
        # 4. Lưu Metadata
        meta_data = {
            "name": payload.name,
            "description": payload.description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_filename": snapshot_name
        }
        with open(backup_path / "backup_info.json", "w", encoding="utf-8") as f:
            import json
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
            
        log_action("Sao lưu DB (Docker)", payload.name, "Backup")
        return {"message": "Sao lưu qua Docker Snapshot thành công!"}
        
    except Exception as e:
        if backup_path.exists():
            shutil.rmtree(backup_path) # Dọn dẹp nếu lỗi giữa chừng
        raise HTTPException(status_code=500, detail=f"Lỗi tạo snapshot: {str(e)}")


@app.get("/api/backup")
async def list_backups():
    backups = []
    if BACKUP_DIR.exists():
        for item in BACKUP_DIR.iterdir():
            if item.is_dir():
                info_file = item / "backup_info.json"
                if info_file.exists():
                    import json
                    with open(info_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        backups.append(meta)
                        
    backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return JSONResponse(content=backups)


@app.delete("/api/backup/{name}")
async def delete_backup(name: str):
    backup_path = BACKUP_DIR / name
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy bản sao lưu")
    shutil.rmtree(backup_path)
    log_action("Xóa Backup DB", name, "Backup")
    return {"message": "Đã xóa bản sao lưu"}


@app.post("/api/backup/{name}/restore")
async def restore_backup(name: str):
    backup_path = BACKUP_DIR / name
    info_file = backup_path / "backup_info.json"
    
    if not backup_path.exists() or not info_file.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy bản sao lưu hợp lệ (chuẩn Snapshot).")

    import json
    with open(info_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    snapshot_filename = meta.get("snapshot_filename")
    snapshot_file_path = backup_path / snapshot_filename
    
    if not snapshot_file_path.exists():
        raise HTTPException(status_code=404, detail="Mất file snapshot vật lý (.snapshot). Không thể khôi phục.")

    try:
        # Upload file snapshot thẳng lên Qdrant thông qua REST API (priority=snapshot để ghi đè)
        upload_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/upload?priority=snapshot"
        
        with open(snapshot_file_path, "rb") as f:
            files = {'snapshot': (snapshot_filename, f, 'application/tar')}
            res = requests.post(upload_url, files=files)
            res.raise_for_status()
            
        log_action("Khôi phục DB (Docker)", name, "Backup")
        return {"message": "Khôi phục thành công! Dữ liệu đã được nạp lại vào Docker."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khôi phục: {str(e)}")
        
        
# phần config UI
@app.get("/api/fragment")
async def get_fragment():
    if not FRAGMENT_FILE.exists():
        return {"content": ""}
    with open(FRAGMENT_FILE, "r", encoding="utf-8") as f:
        return {"content": f.read()}

@app.post("/api/fragment")
async def save_fragment(data: FragmentData):
    try:
        # Tạo thư mục nếu chưa tồn tại
        FRAGMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FRAGMENT_FILE, "w", encoding="utf-8") as f:
            f.write(data.content)
        log_action("Cập nhật Top Message Fragment", "top-chat-message-fragment.html", "UI")
        return {"message": "Lưu Fragment thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# phần config env
@app.get("/api/env")
async def get_env():
    if not ENV_FILE.exists():
        return {"content": ""}
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        return {"content": f.read()}

@app.post("/api/env")
async def save_env(data: EnvData):
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(data.content)
        log_action("Cập nhật cấu hình", ".env", "Config")
        return {"message": "✅ Đã lưu file .env!\n⚠️ LƯU Ý: Bạn cần KHỞI ĐỘNG LẠI server-admin.py để áp dụng cấu hình mới."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

# log view
@app.get("/api/logs/prompt")
async def get_prompt_logs():
    if not PROMPT_LOG_FILE.exists():
        return {"logs": "Chưa có dữ liệu prompt log."}

    try:
        with open(PROMPT_LOG_FILE, "r", encoding="utf-8") as f:
            # Trả về toàn bộ nội dung (vì file đã được giới hạn 1000 dòng ở serverBL)
            return {"logs": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/logs/prompt")
async def clear_prompt_logs():
    try:
        if PROMPT_LOG_FILE.exists():
            PROMPT_LOG_FILE.write_text("", encoding="utf-8")
        log_action("Xóa Prompt Log", "prompt_debug.log", "System")
        return {"message": "Đã xóa trắng file log thành công"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/file")
async def delete_file(folder: str, filename: str):
    target_doc_dir = DOC_DIR if folder == "root" else DOC_DIR / folder
    target_raw_dir = RAW_DIR if folder == "root" else RAW_DIR / folder
    
    md_file_path = target_doc_dir / filename
    
    # 1. Xóa file Markdown (Nếu không có vẫn bỏ qua để tiếp tục xóa Vector)
    if md_file_path.exists():
        md_file_path.unlink()
    else:
        print(f"Cảnh báo: Không tìm thấy file vật lý {md_file_path}, tiếp tục dọn dẹp DB...")

    # 2. Xóa file Raw gốc (Thử tìm theo tên không có đuôi)
    file_stem = Path(filename).stem
    for ext in [".pdf", ".doc", ".docx", ".txt", ".md"]:
        raw_file = target_raw_dir / f"{file_stem}{ext}"
        if raw_file.exists():
            raw_file.unlink()

    # 3. TỰ ĐỘNG XÓA VECTOR TRONG QDRANT
    try:
        from rag_core import GLOBAL_QDRANT_CLIENT, COLLECTION_NAME
        from qdrant_client.http import models
        
        # Xóa tất cả các chunk có metadata.source khớp với tên file
        GLOBAL_QDRANT_CLIENT.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
        )
    except Exception as e:
        print(f"Lỗi khi xóa vector của {filename}: {e}")

    return {"message": f"Đã xóa hoàn toàn {filename} và dữ liệu AI liên quan!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("templates/favicon.ico")

if __name__ == "__main__":
    print("Khởi chạy Admin Service tại http://localhost:8081")
    uvicorn.run(app, host="0.0.0.0", port=8081)
