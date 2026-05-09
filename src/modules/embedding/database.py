import time
import json
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_and_embed(new_docs, db_dir):
    """Cắt nhỏ và nhúng dữ liệu vào Vector DB theo từng đợt"""
    print(f"DEBUG: Số lượng docs nhận vào: {len(new_docs)}")
    if not new_docs:
        return True

    # 1. Phân loại tài liệu
    splittable_docs = []
    non_splittable_docs = []
    
    for doc in new_docs:
        source_path = doc.metadata.get("source", "")
        # Lấy filename gốc để inject vào metadata
        filename = Path(source_path).name if source_path else "Unknown"
        doc.metadata["filename"] = filename
        
        lower_source = source_path.lower()
        if lower_source.endswith((".csv", ".xlsx", ".xls")):
            non_splittable_docs.append(doc)
        else:
            splittable_docs.append(doc)

    # 2. Xử lý Chunking (Băm) có chủ đích cho file văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(splittable_docs)
    
    # Gộp chung với dữ liệu bảng biểu (không băm)
    all_chunks = chunks + non_splittable_docs
    
    print(f"DEBUG: Số lượng chunks tổng cộng tạo ra: {len(all_chunks)}")
    
    if not all_chunks:
        return True

    batch_size = 10
    vector_db = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
        persist_directory=str(db_dir)
    )

    # XOÁ CHUNKS CŨ ĐỂ CẬP NHẬT MỚI
    source_path = all_chunks[0].metadata.get("source", "")
    if source_path:
        try:
            existing_docs = vector_db.get(where={"source": source_path})
            if existing_docs and existing_docs.get("ids"):
                vector_db.delete(ids=existing_docs["ids"])
                print(f"🗑️ Đã xoá {len(existing_docs['ids'])} chunks cũ của {Path(source_path).name}.")
        except Exception as e:
            print(f"⚠️ Không thể xoá chunk cũ (có thể chưa tồn tại): {e}")

    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    try:
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            vector_db.add_documents(batch)
            print(f"✅ Đã nhúng đợt {i//batch_size + 1}/{total_batches}")
                
            if i + batch_size < len(all_chunks):
                time.sleep(3) # Tránh Rate Limit
        return True
    except Exception as e:
        print(f"❌ Lỗi trong quá trình nhúng Chroma: {e}")
        return False