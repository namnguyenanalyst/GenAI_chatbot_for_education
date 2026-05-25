import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import glob

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
        num_ctx=32768, # Tăng trí nhớ lên 32k token để đọc trọn vẹn 50 chunks từ k=50
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
        "Bạn là Trợ lý Học liệu số chuyên nghiệp của nhóm NCKH HUMG.\n\n"
        "THÔNG TIN NGƯỜI DÙNG VÀ YÊU CẦU CẤP BÁCH:\n"
        "- Trình độ sinh viên: {student_profile}\n"
        "- Chế độ yêu cầu: {ai_mode}\n\n"
        "LUẬT DÀNH RIÊNG CHO TỪNG CHẾ ĐỘ (BẮT BUỘC TUÂN THỦ 100%):\n"
        "1. Nếu Chế độ là 'Tra cứu bình thường': Trả lời chính xác, sát nghĩa với tài liệu gốc nhất có thể, KHÔNG suy diễn dài dòng.\n"
        "2. Nếu Chế độ là 'Giải đáp trực tiếp': Trả lời thẳng vào câu hỏi một cách đầy đủ và chi tiết nhất dựa trên tài liệu.\n"
        "3. Nếu Chế độ là 'Gia sư Socratic': TUYỆT ĐỐI KHÔNG đưa ra đáp án trực tiếp. Bạn chỉ được phép đặt các câu hỏi gợi mở, hướng dẫn từng bước để sinh viên tự tìm ra câu trả lời từ tài liệu.\n"
        "4. Nếu Chế độ là 'Tạo bài tập': KHÔNG trả lời câu hỏi trực tiếp. Dựa vào tài liệu, hãy TỰ ĐỘNG SINH RA 3-5 CÂU HỎI TRẮC NGHIỆM (A, B, C, D) có đáp án ẩn ở cuối để kiểm tra kiến thức sinh viên.\n\n"
        "LUẬT DÀNH CHO TRÌNH ĐỘ (BẮT BUỘC TUÂN THỦ 100%):\n"
        "- Nếu là 'Bình thường': Không cần thay đổi văn phong, hãy báo cáo đúng kiến thức gốc.\n"
        "- Nếu là 'Năm 1' hoặc 'Người ngoài ngành': BẮT BUỘC dùng từ ngữ cực kỳ đơn giản, giải thích bằng các VÍ DỤ ĐỜI THƯỜNG.\n"
        "- Nếu là 'Năm cuối': Dùng từ ngữ hàn lâm, chuyên ngành, sâu sắc.\n\n"
        "ĐÂY LÀ CHỈ LỆNH TỐI CAO:\n"
        "- Nếu các tài liệu được cung cấp KHÔNG CHỨA ĐỦ thông tin trả lời, BẮT BUỘC trả lời: 'Xin lỗi, thông tin bạn hỏi hiện không có trong học liệu của khoa.'\n"
        "- BẠN KHÔNG ĐƯỢC PHÉP SÁNG TẠO HOẶC DÙNG KIẾN THỨC BÊN NGOÀI ĐỂ SUY DIỄN.\n\n"
        "NGỮ CẢNH (TÀI LIỆU THAM KHẢO):\n"
        "{context}\n\n"
        "ĐỊNH DẠNG TRẢ LỜI:\n"
        "- Trình bày mạch lạc bằng Bullet Points.\n"
        "- ĐƯỜNG DẪN TRÍCH DẪN: Ở cuối câu trả lời, BẮT BUỘC phải ghi trích dẫn lấy từ file nào (ví dụ: `[Trích từ tài liệu: abc.pdf]`). Nguồn tên file được kẹp ở đuôi mỗi đoạn ngữ cảnh."
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
    
    qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    
    return llm, prompt, document_prompt, vector_db

# Khởi tạo model cơ sở
llm, prompt, document_prompt, vector_db = init_models()

def get_rag_chain(target_filename=None):
    """Tạo chain RAG có khả năng focus vào 1 file cụ thể nếu cần"""
    if target_filename and target_filename != "Tất cả tài liệu":
        # Lọc chặt chẽ trong Elasticsearch chỉ lấy đúng chunk của file này
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 50,
                "filter": [{"term": {"metadata.filename.keyword": target_filename}}]
            }
        )
    else:
        # Tìm kiếm trên toàn bộ database
        retriever = vector_db.as_retriever(search_kwargs={"k": 50})
        
    qa_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    return create_retrieval_chain(retriever, qa_chain)

# --- 2. TẠO DATA AGENT ĐỂ XỬ LÝ CSV/EXCEL ---
def get_data_agent(target_filename=None):
    # Tìm tất cả file CSV và Excel
    csv_files = glob.glob(str(DATA_DIR / "**/*.csv"), recursive=True)
    xlsx_files = glob.glob(str(DATA_DIR / "**/*.xlsx"), recursive=True)
    all_data_files = csv_files + xlsx_files
    
    if not all_data_files:
        return None
    
    # Nếu người dùng chọn đích danh 1 file trong Vùng Tìm Kiếm
    target_file_path = None
    if target_filename and target_filename != "Tất cả tài liệu":
        for f in all_data_files:
            if Path(f).name == target_filename:
                target_file_path = f
                break
                
    # Nếu không tìm thấy hoặc người dùng chọn "Tất cả tài liệu", lấy file mới nhất
    if not target_file_path:
        target_file_path = max(all_data_files, key=os.path.getmtime)
    
    try:
        if target_file_path.endswith('.csv'):
            df = pd.read_csv(target_file_path)
        else:
            df = pd.read_excel(target_file_path)
    except Exception as e:
        print(f"Lỗi đọc file {target_file_path}: {e}")
        return None
        
    # Tạo Pandas Agent chỉ tập trung xử lý đúng file này
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True
    )
    return agent

# --- 3. HÀM PHÂN LOẠI Ý ĐỊNH (ROUTER) ---
def classify_intent(user_query, messages):
    # LỌC NHANH (Rule-based): Bỏ qua LLM nếu là câu hỏi ngắn/giao tiếp
    chat_keywords = ["chào", "hello", "hi", "tác dụng", "là ai", "giúp gì", "cảm ơn", "tạm biệt", "ok", "dạ", "vậy bạn"]
    if len(user_query.split()) < 4 or any(k in user_query.lower() for k in chat_keywords):
        return "CHAT"

    # LỌC DỮ LIỆU BẢNG: Đưa sang luồng Data Agent nếu hỏi về tính toán, giá cả
    data_keywords = ["giá", "đắt nhất", "rẻ nhất", "thấp nhất", "cao nhất", "trung bình", "thống kê", "tổng số", "tổng cộng", "file csv", "file excel", "bảng", "tính toán", "sản phẩm"]
    if any(k in user_query.lower() for k in data_keywords):
        return "DATA"

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

    QUY TẮC PHÂN LOẠI CỰC KỲ NGHIÊM NGẶT:
    1. Trả về 'DATA' NẾU VÀ CHỈ NẾU câu hỏi yêu cầu tính toán, đếm số lượng, liên quan đến giá tiền, tìm giá trị cao nhất/thấp nhất từ bảng biểu.
    2. Trả về 'CHAT' NẾU VÀ CHỈ NẾU câu hỏi đơn thuần là giao tiếp xã giao (ví dụ: xin chào, cảm ơn, tạm biệt, khen ngợi).
    3. Trả về 'RAG' CHO TẤT CẢ CÁC TRƯỜNG HỢP CÒN LẠI (khi người dùng hỏi kiến thức, định nghĩa, "là gì", "tại sao", "như thế nào"). ĐÂY LÀ MẶC ĐỊNH.

    Chỉ trả ra đúng 1 từ duy nhất: 'DATA', 'RAG' hoặc 'CHAT'. Không giải thích thêm.
    """
    
    # SỬ DỤNG CHUNG LLM CHÍNH CHO PHÂN LOẠI
    response = llm.invoke(classification_prompt)
    
    if isinstance(response.content, list):
        content = "".join([item.get("text", "") for item in response.content if isinstance(item, dict)])
    else:
        content = str(response.content)
        
    return content.strip().upper()