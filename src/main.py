import unicodedata
import time as tm
import streamlit as st
import os

from datetime import datetime
from pathlib import Path
from modules.models import llm, rag_chain, classify_intent
from modules.utils import load_all_chats, save_chat, get_folder_map
from tokenAmount import extract_tokens, log_token_usage, TokenCallbackHandler
from modules.embedding.processors import get_loader
from modules.embedding.database import process_and_embed

# --- 1. THIẾT LẬP ĐƯỜNG DẪN TƯƠNG ĐỐI ---
# Xác định thư mục gốc của dự án (NCKH/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn tới Logo và Dữ liệu cục bộ
LOGO_PATH = BASE_DIR / "assets" / "Images" / "Logo ĐH Mỏ - Địa Chất - HUMG.png"
DATA_DIR = BASE_DIR / "Data"
CONFIG_DIR = BASE_DIR / "config"
DB_DIR = CONFIG_DIR / "vector_db"

st.set_page_config(
    page_title="HUMG AI Assistant", 
    page_icon=LOGO_PATH, 
    layout="wide")


# --- 2. GIAO DIỆN SIDEBAR ---
with st.sidebar:
     # --- TÍNH NĂNG 1: NEW CHAT & LỊCH SỬ ---
    if st.button("➕ Bắt đầu Chat mới", use_container_width=True):
        st.session_state.messages = []
        # Tạo ID duy nhất cho cuộc chat mới dựa trên thời gian
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()
    
    st.divider()

    st.title("📂 Quản lý tài liệu")
    
    # --- TÍNH NĂNG 2: UPLOAD & PHÂN LOẠI THÔNG MINH ---
    st.subheader("📤 Tải lên học liệu")
    uploaded_files = st.file_uploader(
        "Chọn file (PDF, Word, Excel...)", 
        accept_multiple_files=True, 
        type=['pdf', 'docx', 'xlsx', 'pptx', 'csv', 'jpg', 'png'],
        label_visibility="collapsed"
    )
    
    allow_overwrite = st.checkbox("Cho phép ghi đè nếu file đã tồn tại", value=False)

    if st.button("🚀 Xử lý & Phân loại", use_container_width=True):
        if uploaded_files:
            folder_map = get_folder_map()
            
            # Biến để kiểm tra xem có file nào thành công không
            any_success = False
            
            for f in uploaded_files:
                file_ext = f.name.split('.')[-1].lower()
                sub_folder = folder_map.get(file_ext, "Others")
                
                # 2. ĐƯỜNG DẪN ĐÍCH: Thư mục Data cục bộ + Sub_folder
                target_path = DATA_DIR / sub_folder
                
                try:
                    # 3. LỆNH QUAN TRỌNG: Tạo thư mục con nếu chưa có
                    target_path.mkdir(parents=True, exist_ok=True)
                    
                    full_file_path = target_path / f.name
                    
                    if full_file_path.exists() and not allow_overwrite:
                        st.warning(f"⚠️ '{f.name}' đã tồn tại. Hãy tích chọn 'Cho phép ghi đè' nếu muốn cập nhật.")
                    else:
                        # 4. GHI FILE THỰC TẾ LÊN LOCAL
                        with open(full_file_path, "wb") as file:
                            file.write(f.getbuffer())
                        
                        # 5. TIẾN HÀNH NHÚNG TRỰC TIẾP VÀO DATABASE
                        with st.spinner(f"Đang xử lý và nhúng dữ liệu file {f.name}..."):
                            loader = get_loader(full_file_path)
                            if loader:
                                try:
                                    pages = loader.load()
                                    success = process_and_embed(pages, DB_DIR)
                                    if success:
                                        st.success(f"✅ Đã lưu và nhúng {f.name} vào hệ thống!")
                                        any_success = True
                                    else:
                                        st.error(f"❌ Nhúng {f.name} vào database thất bại.")
                                except Exception as e:
                                    st.error(f"❌ Lỗi khi đọc file {f.name}: {e}")
                            else:
                                st.warning(f"⚠️ Đã lưu {f.name} nhưng hệ thống chưa hỗ trợ trích xuất chữ cho định dạng này.")
                                any_success = True
                        
                except Exception as e:
                    st.error(f"❌ Lỗi ghi file {f.name}: {e}")
            
            if any_success:
                tm.sleep(1.5) # Để user kịp nhìn thấy thông báo xanh
                st.rerun()
        else: 
            st.warning("⚠️ Bạn chưa chọn tài liệu!")

    st.divider()
    
    st.subheader("🎯 Cá nhân hóa (Personalization)")
    student_level = st.selectbox(
        "Trình độ của bạn:", 
        [
            "Sinh viên năm 1 (Cần giải thích đơn giản, ví dụ dễ hiểu)",
            "Sinh viên năm cuối (Cần giải thích chuyên sâu, hàn lâm)",
            "Người ngoài ngành (Cần giải thích bằng ngôn ngữ đời thường)"
        ]
    )
    
    ai_mode = st.selectbox(
        "Chế độ Trợ lý:", 
        [
            "Giải đáp trực tiếp (Nhanh chóng, đi thẳng vấn đề)",
            "Gia sư Socratic (Gợi mở, không đưa đáp án ngay để SV tự nghĩ)",
            "Tạo bài tập (Tự động sinh câu hỏi trắc nghiệm từ tài liệu)"
        ]
    )

    st.divider()
    
    st.subheader("📊 Token Phiên Trò Chuyện Mở")
    
    # Đảm bảo state lưu trữ token
    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = {"input": 0, "output": 0, "total": 0}
        
    c1, c2 = st.columns(2)
    c1.metric("Gửi Đi (In)", st.session_state.session_tokens["input"])
    c2.metric("Trả Về (Out)", st.session_state.session_tokens["output"])
    st.caption(f"Tổng gánh nặng: **{st.session_state.session_tokens['total']} Tokens**")
    
    st.divider()
    
    st.subheader("📜 Lịch sử trò chuyện")
    all_chats = load_all_chats()
    
    if all_chats:
        sorted_chats = sorted(all_chats.items(), key=lambda x: x[1]['updated_at'], reverse=True)
        
        # TẠO VÙNG TRƯỢT RIÊNG CHO LỊCH SỬ (Chiều cao khoảng 150px)
        with st.container(height=150, border=False):
            for c_id, c_data in sorted_chats:
                first_msg = c_data['messages'][0]['content'] if c_data['messages'] else "Cuộc trò chuyện trống"
                chat_title = (first_msg[:25] + '...') if len(first_msg) > 25 else first_msg
                
                if st.button(f"💬 {chat_title}", key=f"btn_{c_id}", use_container_width=True):
                    st.session_state.messages = c_data['messages']
                    st.session_state.chat_id = c_id
                    st.rerun()
    else:
        st.caption("Chưa có lịch sử trò chuyện.")

    st.divider()

# --- 3. GIAO DIỆN CHAT ---
st.title("💬 Trợ lý Học liệu số HUMG")

# Đảm bảo luôn có chat_id trong session để lưu trữ
if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat từ session_state
for m in st.session_state.messages:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])

if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    # 1. Hiển thị và lưu tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    # 2. Tạo cửa sổ ngữ cảnh (Context Window)
    context_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])

    with st.chat_message("assistant"):
        # Phân loại ý định (truyền lịch sử vào để AI hiểu ngữ cảnh)
        try:
            intent = classify_intent(prompt, st.session_state.messages[:-1])
        except Exception as e:
            intent = "CHAT" # Mặc định trả lời trò chuyện nếu classify_intent lỗi
        
        full_query = f"Lịch sử trò chuyện:\n{context_history}\n\nCâu hỏi hiện tại: {prompt}"

        if "RAG" in intent:
            # Dùng Custom Callback để nghe ngóng hoạt động tiêu thụ token
            cb = TokenCallbackHandler()
            
            def rag_generator():
                for chunk in rag_chain.stream(
                    {
                        "input": full_query,
                        "student_profile": student_level,
                        "ai_mode": ai_mode
                    }, 
                    config={"callbacks": [cb]}
                ):
                    if "answer" in chunk:
                        ans = chunk["answer"]
                        if isinstance(ans, list):
                            yield "".join([item.get("text", "") for item in ans if isinstance(item, dict)])
                        else:
                            yield str(ans)
            
            response = st.write_stream(rag_generator)
            
            # Ghi nhận Token từ Callback
            current_tokens = {
                "input": cb.inputs,
                "output": cb.outputs,
                "total": cb.total
            }
        else:
            cb = TokenCallbackHandler()
            
            # Hàm sinh stream cho LLM cơ bản
            def chat_generator():
                # Thêm System Prompt để định hướng AI là trợ lý học tập chung có cá nhân hóa
                messages = [
                    (
                        "system", 
                        f"Bạn là Trợ lý Học liệu số HUMG. Đối tượng: {student_level}. Chế độ: {ai_mode}. "
                        "Hãy trả lời thân thiện, ngắn gọn và hữu ích. Tránh tự nhận mình là chuyên gia lịch sử."
                    ),
                    ("human", full_query)
                ]
                for chunk in llm.stream(messages, config={"callbacks": [cb]}):
                    if isinstance(chunk.content, list):
                        yield "".join([item.get("text", "") for item in chunk.content if isinstance(item, dict)])
                    else:
                        yield str(chunk.content)
                    
            response = st.write_stream(chat_generator)
                
            current_tokens = {
                "input": cb.inputs,
                "output": cb.outputs,
                "total": cb.total
            }
        
        # 4. Hiển thị và lưu số liệu Tốn Kém Token
        if current_tokens["total"] > 0:
            st.caption(f"🪙 *Mức độ tiêu thụ của câu trả lời này: `{current_tokens['total']}` Token* (Gửi: {current_tokens['input']} | Nhận: {current_tokens['output']})")
            
            # Lưu ra file data gốc báo cáo
            log_token_usage(st.session_state.chat_id, intent, prompt, current_tokens)
            
            # Cộng dồn lại để thanh Menu bên trái chạy theo
            st.session_state.session_tokens["input"] += current_tokens["input"]
            st.session_state.session_tokens["output"] += current_tokens["output"]
            st.session_state.session_tokens["total"] += current_tokens["total"]
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # --- TÍNH NĂNG MỚI: LƯU VÀO FILE JSON ---
        try:
            save_chat(st.session_state.chat_id, st.session_state.messages)
        except Exception as e:
            st.error(f"Lỗi lưu lịch sử: {e}")