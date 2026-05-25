import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# --- 1. THIẾT LẬP ĐƯỜNG DẪN TƯƠNG ĐỐI ---
# File này đang nằm ở: src/modules/models.py
# Cần lùi lại 2 cấp để ra thư mục gốc NCKH/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Tải .env từ thư mục gốc
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Thư mục chứa dữ liệu từ Google Drive (đã đồng bộ về máy)
# Thư mục này nên nằm cùng cấp với thư mục dự án hoặc bên trong dự án
DATA_DIR = BASE_DIR / "Data"

# Thư mục chứa Database và cấu hình
CONFIG_DIR = BASE_DIR / "config"
DB_DIR = CONFIG_DIR / "vector_db"

# Tự động tạo thư mục nếu chưa có để tránh lỗi
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

env_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if env_key:
    key_path = BASE_DIR / env_key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)


@st.cache_resource
def init_models():
    # Khởi tạo Embeddings chạy trên Ollama (nomic-embed-text)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Khởi tạo LLM chính chạy trên Ollama (qwen2.5:7b)
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.3,
    )

    # Kết nối Vector DB bằng ElasticsearchStore
    vector_db = ElasticsearchStore(
        embedding=embeddings,
        index_name="humg_documents",
        es_url="http://localhost:9200"
    )
    
    # Tăng base_retriever k lên 15 để lưới quét rộng hơn
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    
    system_prompt = (
        "Bạn là Trợ lý Học liệu số chuyên nghiệp và thân thiện của nhóm NCKH HUMG (Đại học Mỏ - Địa chất).\n\n"
        "VAI TRÒ VÀ NHIỆM VỤ:\n"
        "Dựa vào các phần tài liệu được trích xuất bên dưới, hãy trả lời câu hỏi của sinh viên một cách chính xác. Yêu cầu tối thượng là bạn phải giải thích các khái niệm học thuật một cách ĐƠN GIẢN HOÁ, sử dụng từ ngữ phổ thông để một người KHÔNG CÓ kiến thức chuyên ngành cũng có thể hiểu ngay lập tức. Nội dung phải ngắn gọn, đi thẳng vào trọng tâm, không lan man nhưng phải đầy đủ ý chính.\n\n"
        "ĐÂY LÀ CHỈ LỆNH TỐI CAO:\n"
        "- Nếu các tài liệu được cung cấp KHÔNG CHỨA ĐỦ thông tin trả lời, BẮT BUỘC trả lời: 'Xin lỗi, thông tin bạn hỏi hiện không có trong học liệu của khoa.'\n"
        "- BẠN KHÔNG ĐƯỢC PHÉP SÁNG TẠO HOẶC DÙNG KIẾN THỨC BÊN NGOÀI ĐỂ SUY DIỄN HAY TRẢ LỜI THAY TÀI LIỆU.\n\n"
        "NGỮ CẢNH (TÀI LIỆU THAM KHẢO):\n"
        "{context}\n\n"
        "ĐỊNH DẠNG TRẢ LỜI:\n"
        "- Viết thành các đoạn văn ngắn gọn.\n"
        "- Nếu có nhiều ý, hãy cấu trúc bằng Bullet Points (Gạch đầu dòng).\n"
        "- Nhớ in đậm (**gây chú ý**) các từ khóa hoặc kết luận cốt lõi.\n"
        "- Luôn giữ thái độ của một gia sư tận tâm.\n"
        "- ĐƯỜNG DẪN TRÍCH DẪN: Kết thúc nội dung trả lời, bạn BẮT BUỘC phải đối chiếu nội dung với nguồn và ghi trích dẫn lấy từ file nào (ví dụ: `[Trích từ tài liệu: abc.pdf]`). Nguồn tên file được kẹp ở đuôi mỗi đoạn ngữ cảnh."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Định dạng lại Context truyền cho LLM để ghim dữ kiện Metadata: filename
    document_prompt = PromptTemplate(
        input_variables=["page_content", "filename"],
        template="{page_content}\n[Nguồn gốc: {filename}]\n"
    )
    
    # Tạo Chain
    qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    return llm, rag_chain

# Khởi tạo model
llm, rag_chain = init_models()

# --- 2. HÀM PHÂN LOẠI Ý ĐỊNH (ROUTER) ---
def classify_intent(user_query, messages):
    # LỌC NHANH (Rule-based): Bỏ qua LLM nếu là câu hỏi ngắn/giao tiếp
    chat_keywords = ["chào", "hello", "hi", "tác dụng", "là ai", "giúp gì", "cảm ơn", "tạm biệt", "ok", "dạ", "vậy bạn"]
    if len(user_query.split()) < 4 or any(k in user_query.lower() for k in chat_keywords):
        return "CHAT"

    history_context = ""
    for m in messages[-6:]:
        role = "Sinh viên" if m["role"] == "user" else "Trợ lý"
        history_context += f"{role}: {m['content']}\n"

    classification_prompt = f"""
    Bạn là bộ phận định tuyến thông minh cho Trợ lý HUMG. 
    Dựa vào lịch sử hội thoại và câu hỏi mới nhất, hãy phân loại ý định người dùng.

    LỊCH SỬ HỘI THOẠI:
    {history_context}

    CÂU HỎI MỚI: "{user_query}"

    QUY TẮC PHÂN LOẠI:
    - Trả về 'RAG': Nếu câu hỏi cần tra cứu kiến thức từ tài liệu.
    - Trả về 'CHAT': Nếu chỉ là chào hỏi, cảm ơn hoặc tán gẫu.

    Chỉ trả ra đúng 1 từ duy nhất: 'RAG' hoặc 'CHAT'.
    """
    
    # SỬ DỤNG CHUNG LLM CHÍNH CHO PHÂN LOẠI
    response = llm.invoke(classification_prompt)
    
    if isinstance(response.content, list):
        content = "".join([item.get("text", "") for item in response.content if isinstance(item, dict)])
    else:
        content = str(response.content)
        
    return content.strip().upper()