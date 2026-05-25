import time
import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
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
    
    # Khởi tạo ElasticsearchStore thay vì Chroma
    vector_db = ElasticsearchStore(
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        index_name="humg_documents",
        es_url="http://localhost:9200"
    )

    # XOÁ CHUNKS CŨ ĐỂ CẬP NHẬT MỚI
    source_path = all_chunks[0].metadata.get("source", "")
    if source_path:
        try:
            # ElasticsearchStore cung cấp phương thức delete dựa trên metadata nếu ta dùng query
            # Hoặc ta có thể xoá qua Elasticsearch client. Tuy nhiên trong langchain ElasticsearchStore:
            # Ta dùng es_connection để thực thi delete by query.
            client = vector_db.client
            query = {
                "query": {
                    "match": {
                        "metadata.source.keyword": source_path
                    }
                }
            }
            # Cần chắc chắn index tồn tại trước khi xoá
            if client.indices.exists(index="humg_documents"):
                res = client.delete_by_query(index="humg_documents", body=query, ignore_unavailable=True)
                deleted_count = res.get('deleted', 0)
                if deleted_count > 0:
                    print(f"🗑️ Đã xoá {deleted_count} chunks cũ của {Path(source_path).name}.")
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