import os
import hashlib
import re
from datetime import datetime
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from dotenv import load_dotenv
import uuid
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
load_dotenv(override=True)

DOC_DIR = os.getenv("DOC_DIR", "documentBL")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
HISTORY_FILE = LOG_DIR / "upload_history.json"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blu")

# Khởi tạo model 1 lần (Global) để tránh load lại mỗi khi upload
GLOBAL_QDRANT_CLIENT = QdrantClient(url=QDRANT_URL)
DENSE_MODEL = OpenAIEmbeddings(
    openai_api_key="empty", # Infinity chạy local không cần key
    openai_api_base=os.getenv("INFINITY_API_BASE", "http://localhost:7997"),
    model=os.getenv("DENSE_EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3"),
    check_embedding_ctx_length=False
)
SPARSE_MODEL = FastEmbedSparse(model_name=os.getenv("SPARSE_EMBEDDINGS_MODEL_NAME", "Qdrant/bm25"))

def track_changes(file_name, old_chunks, new_chunks, version):
    """So sánh và lưu lịch sử thay đổi"""
    old_set = set(old_chunks)
    new_set = set(new_chunks)

    added = list(new_set - old_set)
    deleted = list(old_set - new_set)
    
    # Chỉ lưu nếu có sự thay đổi hoặc là file mới hoàn toàn
    if not added and not deleted and version > 1:
        return

    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "version": version,
        "stats": {
            "total": len(new_chunks),
            "added": len(added),
            "deleted": len(deleted)
        },
        "changes": {
            "added": added,
            "deleted": deleted
        }
    }

    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except: history = []
    
    history.insert(0, history_entry)
    # Giữ tối đa 50 bản ghi gần nhất
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[:50], f, ensure_ascii=False, indent=4)

def get_qdrant_client():
    return GLOBAL_QDRANT_CLIENT

def get_semantic_source(file_path, base_dir=DOC_DIR):
    try:
        rel_path = os.path.relpath(file_path, base_dir)
        path_str = os.path.splitext(rel_path)[0]
        return " > ".join([re.sub(r'^\d+(-\d+)*-', '', p).replace('-', ' ').capitalize() for p in path_str.split(os.sep)])
    except Exception:
        return os.path.basename(file_path)

def split_text_preserve_tables(text, max_chunk_size=1000):
    blocks = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for block in blocks:
        is_table = bool(re.search(r'\|.*\|.*\n\s*\|[-:\s|]+\|', block))
        if is_table:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(block.strip())
        else:
            if len(current_chunk) + len(block) < max_chunk_size:
                current_chunk += "\n\n" + block if current_chunk else block
            else:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = block
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def advanced_clean_text(text: str) -> str:
    """Đã sửa: Chỉ xóa các dòng trống và trùng lặp liền kề, KHÔNG xóa dữ liệu ngắn/lặp ngẫu nhiên"""
    if not text: return ""
    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if cleaned_lines and line == cleaned_lines[-1]: continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def delete_old_vectors_and_get_version(file_name: str) -> int:
    """Xóa vector cũ của file này nếu có và trả về Version mới"""
    client = get_qdrant_client()
    try:
        # Tách kết quả thành 2 biến: results và next_offset
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=file_name))]
            ),
            limit=1,
            with_payload=["metadata"]
        )
        
        # Nếu đã có vector cũ
        if results:
            current_version = results[0].payload.get("metadata", {}).get("version", 0)
            new_version = current_version + 1
            
            # Xóa point cũ
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=file_name))]
                )
            )
            return new_version
            
        return 1 # Chưa có file, version bắt đầu = 1
    except Exception as e:
        return 1

def process_and_upsert_file(file_path: str):
    """Xử lý 1 file: Cắt chunk, nhúng version vào text, và Upsert vào DB"""
    path_obj = Path(file_path)
    file_name = path_obj.name # CHỈ lấy tên file, không lấy đường dẫn
    client = get_qdrant_client()
    
    # --- LẤY DỮ LIỆU CŨ TRƯỚC KHI XÓA ---
    old_chunks = []
    try:
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=file_name))]),
            limit=100, # Lấy tối đa 100 chunk cũ
            with_payload=True
        )
        old_chunks = [res.payload.get("page_content", "") for res in results if res.payload]
    except: pass
    
    # 1. Load nội dung
    if path_obj.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif path_obj.suffix.lower() in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif path_obj.suffix.lower() in [".md", ".txt"]:
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Định dạng không được hỗ trợ")
    
    docs = loader.load()
    raw_text = "\n".join([doc.page_content for doc in docs])
    cleaned_text = advanced_clean_text(raw_text)

    # 2. Chunking
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = markdown_splitter.split_text(cleaned_text)

    # 3. Lấy Version & Xóa dữ liệu cũ
    new_version = delete_old_vectors_and_get_version(file_name)
    current_date = datetime.now().strftime("%d/%m/%Y")
    semantic_source = get_semantic_source(file_path)

    # 4. Gắn Metadata & Header vào từng Chunk
    final_chunks = []
    custom_ids = []
    chunk_index = 0

    for md_doc in md_docs:
        heading_str = " > ".join([f"{v}" for k, v in md_doc.metadata.items() if k.startswith("Header")])
        sub_chunks = split_text_preserve_tables(md_doc.page_content)
        
        for chunk_text in sub_chunks:
            if not chunk_text.strip(): continue
            
            context_path = semantic_source
            if heading_str: context_path += f" > {heading_str}"
            
            # Header chuẩn hóa theo ngữ nghĩa
            header = f"[TÀI LIỆU: {file_name} | NGỮ NGHĨA: {context_path} | VERSION: {new_version} | CẬP NHẬT: {current_date}]\n\n"
            injected_content = header + chunk_text
            
            new_metadata = md_doc.metadata.copy()
            new_metadata.update({
                "source": file_name,
                "version": new_version,
                "updated_at": current_date,
                "semantic_source": semantic_source
            })
            
            final_chunks.append(Document(page_content=injected_content, metadata=new_metadata))
            
            # Tạo ID tĩnh
            hash_str = hashlib.md5((injected_content + file_name + str(chunk_index)).encode('utf-8')).hexdigest()
            custom_ids.append(str(uuid.UUID(hash_str)))
            chunk_index += 1

    # 5. Upsert vào Qdrant
    vectorstore = QdrantVectorStore(
        client=GLOBAL_QDRANT_CLIENT,
        collection_name=COLLECTION_NAME,
        embedding=DENSE_MODEL,
        sparse_embedding=SPARSE_MODEL,
        retrieval_mode=RetrievalMode.HYBRID
    )
    vectorstore.add_documents(documents=final_chunks, ids=custom_ids)
    
    # --- GHI LOG LỊCH SỬ ---
    new_texts = [d.page_content for d in final_chunks]
    track_changes(file_name, old_chunks, new_texts, new_version)
    return new_version