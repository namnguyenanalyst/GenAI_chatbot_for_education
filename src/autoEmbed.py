import os
import time
import hashlib
import json
import sys
import threading
import logging
from pathlib import Path
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from modules.embedding.processors import get_loader
from modules.embedding.database import process_and_embed

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
# Đây là đường dẫn file autoEmbed.py đang chạy, dùng để xác định vị trí gốc của dự án NCKH
CURRENT_FILE = Path(__file__).resolve()

# Vị trí hiện tại ở file NCKH
PROJECT_ROOT = CURRENT_FILE.parent.parent 

# Tải biến môi trường từ file .env cùng cấp với PROJECT_ROOT (chứa thông tin nhạy cảm như API keys)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ĐƯỜNG DẪN GOOGLE DRIVE để file autoEmbed.py theo dõi
google_drive_env = os.getenv("GOOGLE_DRIVE_PATH")
if not google_drive_env:
    print("⚠️ Lỗi: Không tìm thấy biến môi trường GOOGLE_DRIVE_PATH trong file .env!")
    sys.exit(1)
GOOGLE_DRIVE_DIR = Path(google_drive_env)

# Đường dẫn thư mục cấu hình, nơi lưu trữ database vector và file theo dõi registry
CONFIG_DIR = PROJECT_ROOT / "config"

# Đường dẫn xuống thư mục vector_db và file theo dõi registry
DB_DIR = CONFIG_DIR / "vector_db"

# Đường dẫn file theo dõi registry, lưu trữ thông tin hash của các file đã được nhúng để tránh nhúng lại nếu không có thay đổi
TRACKING_FILE = CONFIG_DIR / "data_registry.json"

# Đường dẫn file JSON key của Vertex AI (nếu có), để thiết lập xác thực cho việc nhúng dữ liệu lên Vertex AI
KEY_FILE = CONFIG_DIR / "vertex-ai-key.json"

# Nếu file JSON key tồn tại, thiết lập biến môi trường để Google Cloud SDK có thể sử dụng nó. Nếu không, in cảnh báo và robot sẽ cố gắng sử dụng quyền mặc định từ gcloud SDK (nếu đã cấu hình). Điều này cho phép linh hoạt trong việc triển khai trên các môi trường khác nhau (local hoặc cloud).
if KEY_FILE.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_FILE)
else:
    print(f"⚠️ Cảnh báo: Không tìm thấy file JSON key tại {KEY_FILE}. Robot sẽ dùng quyền mặc định từ gcloud SDK.")

# Cấu hình Logging
LOG_FILE = PROJECT_ROOT / "error.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("autoEmbed")

# Hàm tính hash của file để kiểm tra xem nội dung có thay đổi hay không, giúp tránh việc nhúng lại những file đã được nhúng mà không có thay đổi nào.
def get_file_hash(file_path):
    """Kiểm tra thay đổi nội dung file an toàn chống tràn RAM"""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path} (Xem error.log để biết chi tiết)")
        logger.error(f"Lỗi khi đọc file {file_path}: {e}", exc_info=True)
        return None
    

# --- 2. LOGIC ĐỒNG BỘ HÓA ---
def run_sync():
    print(f"\n🔍 Bắt đầu quét thư mục: {GOOGLE_DRIVE_DIR}")
    
    if not GOOGLE_DRIVE_DIR.exists():
        print(f"❌ LỖI: Không tìm thấy đường dẫn Drive! Kiểm tra xem bạn đã bật App Google Drive chưa.")
        return 

    # Tải registry hiện tại nếu tồn tại, nếu không thì khởi tạo một registry mới. Registry này sẽ lưu trữ thông tin về các file đã được nhúng (dựa trên hash) để tránh việc nhúng lại những file không có thay đổi nào.
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    else:
        registry = {}

    #
    updated_registry = registry.copy()
    found_files_count = 0
    
    # Quét đệ quy
    for full_path in GOOGLE_DRIVE_DIR.rglob('*'):
        # Chỉ xử lý các file có phần mở rộng phù hợp (pdf, docx, xlsx, pptx, csv, jpg, png).
        if full_path.is_file() and full_path.suffix.lower() in ['.pdf', '.docx', '.xlsx', '.pptx', '.csv', '.jpg', '.png']:
            found_files_count += 1
            rel_path = str(full_path.relative_to(GOOGLE_DRIVE_DIR))
            file_hash = get_file_hash(full_path)
            
            if not file_hash: continue

            # Kiểm tra nếu file mới hoặc hash thay đổi
            if rel_path not in registry or registry[rel_path] != file_hash:
                print(f"📄 Đang xử lý file mới: {rel_path}")
                loader = get_loader(full_path)
                if loader:
                    try:
                        pages = loader.load()
                        print(f"   ✅ Đã đọc thành công {len(pages)} trang.")
                        # Gọi nhánh trực tiếp nhúng cho CÁC TÀI LIỆU CỦA FILE NÀY
                        success = process_and_embed(pages, DB_DIR)
                        if success:
                            updated_registry[rel_path] = file_hash
                            # Lưu vào file JSON ngay khi tài liệu này nhúng xong thành công để bảo toàn tiến trình
                            with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
                                json.dump(updated_registry, f, indent=4, ensure_ascii=False)
                            print(f"   ✅ Đã nhúng và cập nhật Registry cho {rel_path}")
                        else:
                            print(f"   ❌ Lỗi khi nhúng {rel_path}, hệ thống sẽ tự thử lại lần sau.")
                    except Exception as e:
                        print(f"   ❌ Lỗi khi nạp nội dung (Xem error.log để biết chi tiết): {e}")
                        logger.error(f"Lỗi nạp file {rel_path}: {e}", exc_info=True)
                else:
                    print(f"   ⚠️ Không tìm thấy Loader phù hợp (Hoặc là Tệp Ảnh): {full_path.suffix}")

    print(f"📊 Thống kê: Tìm thấy {found_files_count} file phù hợp.")
    print("✅ Đồng bộ hoàn tất định kì.")

# --- 3. THIẾT LẬP WATCHDOG ---

sync_timer = None

def trigger_sync():
    """Hàm debounce để tránh gọi run_sync quá nhiều lần khi copy paste nhiều file"""
    global sync_timer
    if sync_timer:
        sync_timer.cancel()
    # Nếu trong vòng 3 giây không có file mới được thả vào thì mới tiến hành quét và nhúng
    sync_timer = threading.Timer(3.0, run_sync)
    sync_timer.start()

class DataHandler(FileSystemEventHandler):
    def triger_conditions(self, event):
        if event.is_directory:
            return False
        # Chuyển ext sang đuôi viết thường và bổ sung đủ đuôi
        ext = Path(event.src_path).suffix.lower()
        return ext in ['.pdf', '.docx', '.xlsx', '.pptx', '.csv', '.jpg', '.png']

    def on_modified(self, event):
        if self.triger_conditions(event):
            print(f"\n⚡ Phát hiện thay đổi: {Path(event.src_path).name}")
            trigger_sync()

    def on_created(self, event):
        if self.triger_conditions(event):
            print(f"\n⚡ Phát hiện file mới: {Path(event.src_path).name}")
            trigger_sync()

if __name__ == "__main__":
    run_sync()
    
    event_handler = DataHandler()
    observer = Observer()
    observer.schedule(event_handler, str(GOOGLE_DRIVE_DIR), recursive=True)
    observer.start()
    
    print(f"\n🤖 Robot HUMG đang canh gác tại Drive...")
    print("Bấm Ctrl+C để dừng.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Robot nghỉ ngơi.")
    observer.join()