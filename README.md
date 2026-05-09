# HUMG AI Assistant - Trợ Lý Học Liệu Số NCKH 🤖📚

Chào mừng bạn đến với **HUMG AI Assistant** – một sản phẩm trí tuệ nhân tạo được thiết kế đặc biệt để giúp bạn tự động hóa việc đọc, phân loại và truy vấn học liệu. Hãy tưởng tượng thay vì phải lật tung hàng trăm trang PDF giáo trình để tìm một công thức, bạn chỉ việc gõ câu hỏi, AI sẽ tự tìm, trích xuất và giải thích dựa trên chính xác tài liệu bạn cung cấp.

---

## 🌟 Những Năng Lực Cốt Lõi

1. **🧠 Nạp "Não Bộ" Tự Động (Auto-Embedding):** 
   Một "robot tàng hình" luôn túc trực, mỗi khi bạn tải một file tài liệu mới lên, hệ thống sẽ tự động đem đi đọc và nén vào trí nhớ mà không cần bạn làm gì thêm!
2. **🔎 Đọc Tên Bắt Hình Dong (Smart RAG):** 
   AI tự động hiểu câu hỏi, lục lọi thẳng vào bộ tài liệu (PDF, Word, Excel, Slide) và chỉ lôi ra những đáp án sát nhất với văn bản gốc.
3. **🎭 Đa Sắc Thái (Router phân luồng):**
   Bạn có thể hỏi nó kiến thức hàn lâm, nhưng cũng có thể vào chào nó buổi sáng. AI thông minh tự động tách biệt để phục vụ theo đúng kiểu mẫu.
4. **📖 Quản lý như một Thư Viện Thực Thụ:**
   Lịch sử chat cũng như tệp tin bạn thả vào sẽ tự động lưu lại phân loại vào "ngăn kéo" (Folder) theo đuôi của nó gọn gàng.

---

## 🛠 Hướng Dẫn Tinh Chỉnh & Cài Đặt Ban Đầu

Nếu bạn tải dự án này về một máy tính mới, hãy làm 3 bước sau:

1. **Kích hoạt môi trường (Virtual Environment):**
   ```bash
   source .venv/bin/activate
   ```
2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Môi trường kết nối (API & Cấu Hình):**
   Đảm bảo bạn có file `.env` ở thư mục gốc chứa dòng mã:
   `GOOGLE_API_KEY=AI...`
   Và thư mục `config/vertex-ai-key.json` đã chứa khóa JSON kết nối mô hình.

---

## 🚀 Cách Sử Dụng Ứng Dụng Hàng Ngày

Để trải nghiệm mọi công năng mượt mà nhất, hãy mở **2 cửa sổ Terminal** (cửa sổ lệnh) song song cùng lúc nhé:

### Bước 1: Gọi Trợ Lý Trông Coi (Watchdog) 🐕
Ở cửa sổ Terminal thứ nhất, hãy kích hoạt người giữ cửa làm nhiệm vụ nhúng Text:
```bash
python src/autoEmbed.py
```
> *Lúc này Ứng dụng ngầm sẽ chạy và thông báo "Robot HUMG đang canh gác tại Drive...". Hãy treo nó ở đó và để nó chạy ẩn.*

### Bước 2: Bật Giao Diện Tương Tác 💬
Ở cửa sổ Terminal thứ hai, kích hoạt màn hình trò chuyện Web bằng lệnh:
```bash
streamlit run src/main.py
```
> *Tự động một trang Web sẽ nhảy lên. Giờ thì bạn tha hồ Tải Tài Liệu Lên và Bắt Đầu Đặt Câu Hỏi cho mọi môn học rồi!*

---

## 📂 Các Định Dạng Bạn Hỗ Trợ 
Hệ thống sẵn sàng nhai nát và thẩm thấu rất nhiều nguồn tài nguyên:
- Hệ Văn Bản: `.pdf` (rất mạnh), `.docx`
- Hệ Dữ Liệu Bảng: `.xlsx`, `.csv`
- Hệ Trình Chiếu: `.pptx`

*(Hình ảnh đuôi ảnh vẫn có thể truyền vào để lưu nhanh qua thư mục, dù AI chưa chính thức bóc chữ trên mặt ảnh để giữ nhẹ ram).*

