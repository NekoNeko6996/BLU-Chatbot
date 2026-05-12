import logging
import os
import re
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from dotenv import load_dotenv

load_dotenv(override=True)

DENSE_EMBEDDINGS_MODEL_NAME = os.getenv("DENSE_EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3")
DENSE_EMBEDDINGS_MODEL_KWARGS_DEVICE = os.getenv("DENSE_EMBEDDINGS_MODEL_KWARGS_DEVICE", "cpu")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blu")
DOC_DIR = os.getenv("DOC_DIR", "documentBL")
SPARSE_EMBEDDINGS_MODEL_NAME = os.getenv("SPARSE_EMBEDDINGS_MODEL_NAME", "Qdrant/bm25")
EMBEDDINGS_BATCH_SIZE = int(os.getenv("EMBEDDINGS_BATCH_SIZE", 500))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def advanced_clean_text(text: str) -> str:
    """Đã sửa: Chỉ xóa dòng trống và dòng trùng lặp liền kề để bảo toàn sđt, địa chỉ"""
    if not text: return ""
    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if cleaned_lines and line == cleaned_lines[-1]: continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

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

def load_and_process_documents():
    """CHỈ tải file .md vì Docling đã convert sẵn ở bước Upload"""
    documents = []
    for root, _, files in os.walk(DOC_DIR):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    text_content = TextLoader(file_path, encoding='utf-8').load()[0].page_content
                    clean_text = advanced_clean_text(text_content)
                    semantic_source = get_semantic_source(file_path)
                    documents.append(Document(page_content=clean_text, metadata={"source": file, "semantic_source": semantic_source}))
                except Exception as e:
                    logger.error(f"Lỗi đọc file {file}: {e}")
    return documents

def split_and_inject_version(documents):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    final_chunks = []
    current_date = datetime.now().strftime("%d/%m/%Y")

    for doc in documents:
        md_docs = markdown_splitter.split_text(doc.page_content)
        file_name = doc.metadata.get('source')
        for md_doc in md_docs:
            heading_str = " > ".join([f"{v}" for k, v in md_doc.metadata.items() if k.startswith("Header")])
            sub_chunks = split_text_preserve_tables(md_doc.page_content)
            
            for chunk_text in sub_chunks:
                if not chunk_text.strip(): continue
                
                context_path = doc.metadata.get('semantic_source', '')
                if heading_str: context_path += f" > {heading_str}"
                
                # INJECT VERSION VÀO TEXT
                header = f"[TÀI LIỆU: {file_name} | NGỮ NGHĨA: {context_path} | VERSION: 1 | CẬP NHẬT: {current_date}]\n\n"
                injected_content = header + chunk_text
                
                new_metadata = md_doc.metadata.copy()
                new_metadata.update({
                    "source": file_name,
                    "version": 1,
                    "updated_at": current_date
                })
                final_chunks.append(Document(page_content=injected_content, metadata=new_metadata))
    return final_chunks

def main():
    logger.info("1. Tải và xử lý các file Markdown...")
    documents = load_and_process_documents()
    splits = split_and_inject_version(documents)
    logger.info(f"-> Tạo thành công {len(splits)} chunks.")

    logger.info("2. Load Model Embedding...")
    dense = HuggingFaceEmbeddings(model_name=DENSE_EMBEDDINGS_MODEL_NAME, model_kwargs={"device": DENSE_EMBEDDINGS_MODEL_KWARGS_DEVICE})
    sparse = FastEmbedSparse(model_name=SPARSE_EMBEDDINGS_MODEL_NAME)
    
    # Recreate Collection thay vì xóa rỗng thư mục (An toàn & sạch sẽ hơn)
    client = QdrantClient(url=QDRANT_URL)

    logger.info("3. Bắt đầu Upsert vào Qdrant...")
    for i in range(0, len(splits), EMBEDDINGS_BATCH_SIZE):
        batch = splits[i:i+EMBEDDINGS_BATCH_SIZE]
        ids = [str(uuid.UUID(hashlib.md5((doc.page_content + doc.metadata['source'] + str(idx)).encode('utf-8')).hexdigest())) for idx, doc in enumerate(batch)]
        
        is_first_batch = (i == 0)
        
        vectorstore = QdrantVectorStore.from_documents(
            documents=batch,
            embedding=dense,
            sparse_embedding=sparse,
            ids=ids,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
            force_recreate=is_first_batch
        )
        
        vectorstore.add_documents(documents=batch, ids=ids)
        
        logger.info(f"-> Đã nạp {min(i+EMBEDDINGS_BATCH_SIZE, len(splits))}/{len(splits)} chunks...")

    client.close()
    logger.info("✅ CẬP NHẬT TOÀN BỘ DỮ LIỆU THÀNH CÔNG!")

if __name__ == "__main__":
    main()