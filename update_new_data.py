import logging
import os
import re
import uuid
import hashlib
import shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from dotenv import load_dotenv
load_dotenv(override=True)

# env
DENSE_EMBEDDINGS_MODEL_NAME = os.getenv("DENSE_EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3")
DENSE_EMBEDDINGS_MODEL_KWARGS_DEVICE = os.getenv("DENSE_EMBEDDINGS_MODEL_KWARGS_DEVICE", "cpu")
DB_PATH = os.getenv("DATABASE_PATH", "database/collection")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blu")
DOC_DIR = os.getenv("DOC_DIR", "documentBL")
SPARSE_EMBEDDINGS_MODEL_NAME = os.getenv("SPARSE_EMBEDDINGS_MODEL_NAME", "Qdrant/bm25")
EMBEDDINGS_BATCH_SIZE = os.getenv("EMBEDDINGS_BATCH_SIZE", 500)

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 1. Hàm làm sạch text cơ bản ===
def advanced_clean_text(text: str) -> str:
    if not text: 
        return ""
    lines = text.split('\n')
    cleaned_lines = []
    seen_lines_count = {}
    for line in lines:
        line = line.strip()
        if len(line) < 5 and not any(c in line for c in ['|', '!', '[', ']']):
            continue
        if cleaned_lines and line == cleaned_lines[-1]:
            continue
        is_table_row = line.startswith('|') and line.endswith('|')
        if not is_table_row:
            seen_lines_count[line] = seen_lines_count.get(line, 0) + 1
            if seen_lines_count[line] > 3:
                continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# === 2. HÀM MỚI: Trích xuất đường dẫn thành Nguồn ngữ nghĩa ===
def get_semantic_source(file_path, base_dir="documentBL"):
    """
    Biến đổi 'documentBL/1-tuyen-sinh/5-1-gioi-thieu-nganh-cntt.md' 
    Thành: 'Tuyển sinh > Giới thiệu ngành cntt'
    """
    try:
        rel_path = os.path.relpath(file_path, base_dir)
        path_str = os.path.splitext(rel_path)[0]
        parts = path_str.split(os.sep)
        clean_parts = []
        for part in parts:
            # Xóa các số thứ tự phân cấp ở đầu (vd: 0-, 1-, 5-1-)
            part = re.sub(r'^\d+(-\d+)*-', '', part)
            part = part.replace('-', ' ')
            part = part.capitalize()
            clean_parts.append(part)
        return " > ".join(clean_parts)
    except Exception:
        return os.path.basename(file_path)

# === 3. HÀM MỚI: Cắt Chunk nhưng Tuyệt đối KHÔNG cắt Bảng biểu ===
def split_text_preserve_tables(text, max_chunk_size=1000):
    """
    Thuật toán tách văn bản theo đoạn (\n\n). 
    Nếu đoạn đó là Bảng Markdown, giữ nguyên vẹn thành 1 chunk độc lập dù dung lượng lớn.
    """
    blocks = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for block in blocks:
        # Nhận diện bảng Markdown (có chứa | và dòng gạch ngang |---| )
        is_table = bool(re.search(r'\|.*\|.*\n\s*\|[-:\s|]+\|', block))
        
        if is_table:
            # Đẩy phần text đang tích lũy vào trước
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Bảng được đưa vào thành 1 chunk duy nhất, không bị chẻ đôi
            chunks.append(block.strip())
        else:
            # Cộng gộp text thường nếu chưa vượt quá max_chunk_size
            if len(current_chunk) + len(block) < max_chunk_size:
                current_chunk += "\n\n" + block if current_chunk else block
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = block
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# === 4. Load File & Gắn Nguồn Ngữ Nghĩa ===
def load_documents(directory_path: str):
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                text_content = ""
                if file.endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                    text_content = loader.load()[0].page_content
                elif file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    text_content = "\n".join([page.page_content for page in loader.load()])
                elif file.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(file_path)
                    text_content = "\n".join([page.page_content for page in loader.load()])
                
                if text_content:
                    clean_text = advanced_clean_text(text_content)
                    semantic_source = get_semantic_source(file_path, directory_path)
                    
                    documents.append(Document(
                        page_content=clean_text, 
                        metadata={"source": file_path, "semantic_source": semantic_source}
                    ))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    return documents

# === 5. Cắt văn bản & Tiêm Nguồn vào nội dung ===
def process_and_split_documents(documents):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    final_chunks = []
    for doc in documents:
        # Tách theo Heading trước
        md_docs = markdown_splitter.split_text(doc.page_content)
        
        for md_doc in md_docs:
            # Lọc lấy tất cả Heading gom lại thành 1 chuỗi
            headings = [f"{v}" for k, v in md_doc.metadata.items() if k.startswith("Header")]
            heading_str = " > ".join(headings)
            
            # Tách nội dung bên trong Heading, giữ nguyên Bảng
            sub_chunks = split_text_preserve_tables(md_doc.page_content, max_chunk_size=1000)
            
            for chunk_text in sub_chunks:
                if not chunk_text.strip():
                    continue
                    
                # => TIÊM ĐƯỜNG DẪN NGỮ NGHĨA VÀO ĐẦU CHUNK <=
                semantic_source = doc.metadata.get('semantic_source', '')
                context_path = f"{semantic_source}"
                if heading_str:
                    context_path += f" > {heading_str}"
                    
                injected_content = f"[{context_path}]\n\n{chunk_text}"
                
                new_metadata = md_doc.metadata.copy()
                new_metadata.update({
                    "source": doc.metadata.get('source', ''),
                    "semantic_source": semantic_source,
                    "context_path": context_path
                })
                
                final_chunks.append(Document(page_content=injected_content, metadata=new_metadata))
                
    return final_chunks

# === 6. Main Process ===
def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    
    logger.info(f"Đang tải dữ liệu từ {DOC_DIR}...")
    documents = load_documents(DOC_DIR)
    
    logger.info("Đang xử lý tách chunk và tiêm siêu dữ liệu...")
    splits = process_and_split_documents(documents)
    logger.info(f"Tổng số chunk sau khi tách: {len(splits)}")

    logger.info("Đang tải Embedding Models...")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=DENSE_EMBEDDINGS_MODEL_NAME,
        model_kwargs={"device": DENSE_EMBEDDINGS_MODEL_KWARGS_DEVICE}
    )
    
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDINGS_MODEL_NAME)
    
    if os.path.exists(DB_PATH):
        logger.info(f"Đang xóa Vector Database cũ tại '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)
    
    os.makedirs(DB_PATH, exist_ok=True)
    logger.info("Đang nạp dữ liệu vào Vector Database mới...")

    first_batch = splits[0:EMBEDDINGS_BATCH_SIZE]
    custom_ids = [str(uuid.UUID(hashlib.md5(doc.page_content.encode('utf-8')).hexdigest())) for doc in first_batch]
    
    qdrant_vectorstore = QdrantVectorStore.from_documents(
        documents=first_batch,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        ids=custom_ids,
        collection_name=COLLECTION_NAME,
        path=DB_PATH, 
        retrieval_mode=RetrievalMode.HYBRID
    )

    for i in range(EMBEDDINGS_BATCH_SIZE, len(splits), EMBEDDINGS_BATCH_SIZE):
        batch_docs = splits[i:i+EMBEDDINGS_BATCH_SIZE]
        custom_ids = [str(uuid.UUID(hashlib.md5(doc.page_content.encode('utf-8')).hexdigest())) for doc in batch_docs]
        qdrant_vectorstore.add_documents(documents=batch_docs, ids=custom_ids)
        logger.info(f"Đã nạp {min(i+EMBEDDINGS_BATCH_SIZE, len(splits))}/{len(splits)} chunks...")

    logger.info("✅ Cập nhật dữ liệu thành công!")

if __name__ == "__main__":
    main()