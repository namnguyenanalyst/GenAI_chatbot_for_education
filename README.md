# HUMG AI Assistant - Trợ Lý Học Liệu Số NCKH 🤖📚

Chào mừng bạn đến với **HUMG AI Assistant** – một sản phẩm trí tuệ nhân tạo được thiết kế đặc biệt để giúp sinh viên Đại học Mỏ - Địa chất (HUMG) tự động hóa việc đọc, phân loại và truy vấn học liệu. Hệ thống chạy **100% Offline (Local)** với tốc độ cao, đảm bảo tuyệt đối an toàn dữ liệu và không lo giới hạn API.

---

## 🌟 Những Năng Lực Cốt Lõi

1. **🧠 Chạy 100% Offline (Ollama):** 
   Sử dụng mô hình ngôn ngữ lớn (LLM) `Qwen 2.5` và mô hình nhúng `Nomic-Embed-Text` chạy trực tiếp trên máy của bạn thông qua Ollama. Khả năng đọc hiểu tiếng Việt xuất sắc mà không tốn một đồng phí API nào.
2. **🔎 Tìm Kiếm Lai (Hybrid Search RAG):** 
   Lưu trữ dữ liệu bằng **Elasticsearch** cho phép kết hợp tìm kiếm theo ngữ nghĩa (Vector) và tìm kiếm theo từ khóa chính xác (BM25 Keyword), giúp lục lọi tài liệu cực kỳ chính xác.
3. **🎭 Đa Sắc Thái (Router phân luồng):**
   AI tự động phân loại xem bạn đang muốn tra cứu tài liệu chuyên ngành hay chỉ đang muốn nói chuyện, tán gẫu thông thường để đưa ra phương án trả lời phù hợp nhất.
4. **📖 Quản lý Upload Trực Tiếp:**
   Tất cả tài liệu PDF, Word, Excel tải lên qua giao diện Web sẽ tự động được lưu trữ và nhúng thẳng vào hệ thống Database ngay tức thì. Không cần phải thao tác với Google Drive hay các ứng dụng chạy ngầm phức tạp.

---

## 🛠 Hướng Dẫn Cài Đặt Ban Đầu

Vì hệ thống chạy Local 100%, bạn cần chuẩn bị đầy đủ Môi trường Python, Docker và Ollama.

### Bước 1: Cài đặt Docker và Khởi chạy Elasticsearch
Hệ thống sử dụng Elasticsearch làm Vector Database. Chạy lệnh sau để khởi động (máy cần có Docker):
```bash
# Đi vào thư mục gốc của dự án
cd /home/nam/GenAI_chatbot_for_education

# Khởi chạy Elasticsearch ngầm
sudo docker compose up -d
```

### Bước 2: Cài đặt Ollama và Tải Mô hình AI
1. Tải và cài đặt Ollama từ [https://ollama.com](https://ollama.com) (Hoặc chạy `curl -fsSL https://ollama.com/install.sh | sh` trên Linux).
2. Tải mô hình ngôn ngữ và mô hình nhúng (Chỉ làm 1 lần, dung lượng ~5GB):
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Bước 3: Cài đặt thư viện Python
Kích hoạt môi trường ảo (nếu có) và cài đặt các gói cần thiết:
```bash
pip install -r requirements.txt
```

---

## 🚀 Cách Khởi Chạy Ứng Dụng Hàng Ngày

Khi mọi thứ đã được cài đặt xong, mỗi ngày bạn chỉ cần mở Terminal tại thư mục dự án và chạy đúng 1 lệnh duy nhất:

```bash
streamlit run src/main.py
```

> *Giao diện Web sẽ tự động bật lên. Lúc này bạn có thể thỏa thích tải lên các file giáo trình và bắt đầu đặt câu hỏi!*

---

## 📂 Các Định Dạng Hỗ Trợ 
Hệ thống sẵn sàng phân tích và trích xuất rất nhiều nguồn tài nguyên:
- **Hệ Văn Bản:** `.pdf` (rất mạnh), `.docx`
- **Hệ Dữ Liệu Bảng:** `.xlsx`, `.csv`
- **Hệ Trình Chiếu:** `.pptx`

*(Lưu ý: Bạn có thể chọn tùy chọn "Cho phép ghi đè" trên giao diện khi tải lại các file đã chỉnh sửa để hệ thống tự động cập nhật kiến thức mới vào Database).*
