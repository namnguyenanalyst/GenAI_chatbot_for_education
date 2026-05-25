from pathlib import Path
from src.modules.embedding.processors import get_loader
from src.modules.embedding.database import process_and_embed
import traceback

def test_file(file_path_str):
    try:
        p = Path(file_path_str)
        loader = get_loader(p)
        if loader is None:
            print(f"[{file_path_str}] Không có loader.")
            return
        docs = loader.load()
        print(f"[{file_path_str}] Tải thành công {len(docs)} tài liệu thô.")
        process_and_embed(docs, None)
    except Exception as e:
        print(f"[{file_path_str}] LỖI: {e}")
        traceback.print_exc()

test_file("Data/Data_Files/thongtintruong.csv")
test_file("Data/PDF_Files/TmhiuvBlockchain.pdf")
test_file("Data/Others/Nàng thơ Amee.jpeg")
test_file("Data/Audio_Files/20260522094344_cache_b9e06ab1-8683-44b9-aec6-fad541b5fdf3.wav")
