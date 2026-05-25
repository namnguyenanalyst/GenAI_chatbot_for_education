import sys
import os
import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv

# Cấu hình đường dẫn để import từ thư mục src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Import RAG pipeline và model từ mã nguồn chính
from src.modules.models import llm, rag_chain

# Import Ragas 
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaEmbeddings
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

    print(f"Tìm thấy {len(data)} câu hỏi. Đang tiến hành hỏi Chatbot ngầm...")
    
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
        
        print(f"Q: {q}")
        print(f"A: {ans[:100]}...\n")

    # 3. Chuẩn bị cấu trúc Dataset cho Ragas phiên bản mới (v0.2+)
    data_dict = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    print("--- ĐANG ĐO LƯỜNG MATRIX BẰNG RAGAS ---")
    
    # Ragas 0.4.x yêu cầu sử dụng OpenAI client qua llm_factory để hỗ trợ JSON/Structured output.
    # Vì dùng Ollama, ta giả lập OpenAI client trỏ vào cổng 11434 của máy chủ Ollama.
    ollama_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Bọc mô hình LLM giám khảo
    ragas_llm = llm_factory("qwen2.5:7b", client=ollama_client)
    
    # Bọc Embedding giám khảo bằng wrapper tương thích bản cũ
    eval_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)

    # 4. Chạy Metric
    # - Faithfulness: Đo lường mức độ Ảo giác (Hallucination). AI có bịa thêm ý ngoài tài liệu không?
    # - AnswerRelevancy: AI trả lời có đi đúng trọng tâm câu hỏi không?
    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        raise_exceptions=False
    )
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ---")
    print(result)
    
    # 5. Lưu ma trận ra Excel/CSV để phân tích
    df = result.to_pandas()
    out_path = Path(__file__).parent / "evaluation_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu kết quả chi tiết từng nhãn tại: {out_path}")

if __name__ == "__main__":
    main()
