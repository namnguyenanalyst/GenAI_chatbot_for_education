import json
from datetime import datetime
from pathlib import Path
from langchain_core.callbacks.base import BaseCallbackHandler

# Thư mục gốc dự án (NCKH)
BASE_DIR = Path(__file__).resolve().parent.parent

# Tệp tin dùng để ghi log báo cáo (config/token_history.json)
TOKEN_FILE = BASE_DIR / "config" / "token_history.json"

def extract_tokens(ai_message):
    """
    Hàm bốc tách dữ liệu MetaData ẩn bên trong câu trả lời từ Google Gemini thông qua Langchain.
    Trả về Dictionary: {'input': X, 'output': Y, 'total': Z}
    """
    input_tok = 0
    output_tok = 0
    total_tok = 0
    
    # Langchain Model đôi khi bọc metadata dưới biến usage_metadata
    if hasattr(ai_message, "usage_metadata") and ai_message.usage_metadata is not None:
        input_tok = ai_message.usage_metadata.get("input_tokens", 0)
        output_tok = ai_message.usage_metadata.get("output_tokens", 0)
        total_tok = ai_message.usage_metadata.get("total_tokens", 0)
    
    # Langchain cũng có khi trả về dưới dạng response_metadata (tùy version và chain)
    elif hasattr(ai_message, "response_metadata") and ai_message.response_metadata is not None:
        token_usage = ai_message.response_metadata.get("token_usage", {})
        input_tok = token_usage.get("prompt_tokens", 0)
        output_tok = token_usage.get("completion_tokens", 0)
        total_tok = token_usage.get("total_tokens", input_tok + output_tok)
        
    return {
        "input": input_tok,
        "output": output_tok,
        "total": total_tok
    }

class TokenCallbackHandler(BaseCallbackHandler):
    """
    Callback function để đếm token tổng hợp cho tất cả các cuộc gọi LLM
    nằm bên trong một Chain phức tạp (ví dụ: RAG Chain kết hợp MultiQueryRetriever).
    """
    def __init__(self):
        self.inputs = 0
        self.outputs = 0
        self.total = 0

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, 'message', None)
                if msg:
                    usage = getattr(msg, 'usage_metadata', None)
                    if usage:
                        self.inputs += usage.get('input_tokens', 0)
                        self.outputs += usage.get('output_tokens', 0)
                        self.total += usage.get('total_tokens', 0)

def log_token_usage(chat_id, intent, user_query, tokens_dict):
    """
    Hàm tự động dồn đuôi (append) thông số tiêu thụ ra File thống kê để báo cáo
    """
    try:
        # Nếu thư mục config chưa có thì mọc ra
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Mở file cũ lấy lại danh sách
        data = []
        if TOKEN_FILE.exists():
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                    
        # Tao bản ghi mới
        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chat_id": chat_id,
            "intent": intent,
            "query_preview": user_query[:50] + ("..." if len(user_query) > 50 else ""), # Trích tối đa 50 ký tự đầu làm xem trước
            "input_tokens": tokens_dict["input"],
            "output_tokens": tokens_dict["output"],
            "total_tokens": tokens_dict["total"]
        }
        
        # Thêm vào mảng và ghi đè JSON
        data.append(new_entry)
        
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"Lỗi hệ thống khi lưu Token: {e}")
