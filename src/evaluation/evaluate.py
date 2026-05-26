import sys
import os
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Tải bộ công cụ tách từ của nltk (nếu chưa có)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Cấu hình đường dẫn để import từ thư mục src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Import RAG pipeline và model từ mã nguồn chính
from src.modules.models import llm, get_rag_chain

import warnings

# Tắt cảnh báo DeprecationWarning để log sạch hơn
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("--- BẮT ĐẦU ĐÁNH GIÁ (EVALUATION) ---")
    
    # 1. Load Golden Dataset
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    user_inputs = []
    references = []
    responses = []
    retrieved_contexts = []
    bleu_scores = []

    print(f"Tìm thấy {len(data)} câu hỏi. Đang tiến hành hỏi Chatbot ngầm...")
    
    rag_chain = get_rag_chain()
    
    # 2. Sinh câu trả lời từ hệ thống RAG
    for item in data:
        q = item["question"]
        gt = item["ground_truth"]
        
        # Đưa câu hỏi vào hệ thống
        try:
            rag_response = rag_chain.invoke({"input": q})
        except Exception as e:
            if "index_not_found_exception" in str(e):
                print("\n❌ LỖI NGHIÊM TRỌNG: Kho dữ liệu Elasticsearch của bạn ĐANG TRỐNG!")
                print("👉 Vui lòng mở trang web Chatbot lên, bấm nút 'Đồng bộ hóa' hoặc 'Nạp dữ liệu' để AI đọc file vào kho trước khi chạy bài thi đánh giá.")
                sys.exit(1)
            else:
                raise e
        
        ans = rag_response.get("answer", "")
        # Lọc kết quả trả về
        if isinstance(ans, list):
            ans = "".join([i.get("text", "") for i in ans if isinstance(i, dict)])
        elif hasattr(ans, "content"):
            ans = ans.content
            
        ctx_docs = rag_response.get("context", [])
        ctx_texts = [doc.page_content for doc in ctx_docs]
        
        user_inputs.append(q)
        references.append(gt)
        responses.append(str(ans))
        retrieved_contexts.append(ctx_texts)
        
        # Tính BLEU score
        ref_tokens = nltk.word_tokenize(gt.lower())
        ans_tokens = nltk.word_tokenize(str(ans).lower())
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([ref_tokens], ans_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
        print(f"Q: {q}")
        print(f"A: {ans[:100]}...")
        print(f"BLEU: {bleu:.4f}\n")

    print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ---")
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"Điểm BLEU trung bình: {avg_bleu:.4f}")
    
    # 3. Lưu ma trận ra Excel/CSV để phân tích
    df = pd.DataFrame({
        "user_input": user_inputs,
        "reference": references,
        "response": responses,
        "bleu_score": bleu_scores
    })
    
    out_path = Path(__file__).parent / "evaluation_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu kết quả chi tiết tại: {out_path}")

if __name__ == "__main__":
    main()
