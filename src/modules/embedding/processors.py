import pandas as pd
import pytesseract
import whisper
import warnings
from PIL import Image
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader
)

# --- GIAI ĐOẠN 1: BẢNG -> VĂN BẢN (Textualization) ---
class PandasTextualizationLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = str(file_path)
        
    def load(self):
        ext = Path(self.file_path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(self.file_path)
        else:
            df = pd.read_excel(self.file_path)
            
        docs = []
        for index, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip() != "":
                    text_parts.append(f"{col} là {val}")
            
            if text_parts:
                page_content = "Bản ghi chi tiết: " + ", ".join(text_parts) + "."
                docs.append(Document(page_content=page_content, metadata={"source": self.file_path, "row": index}))
        return docs

# --- GIAI ĐOẠN 3: HÌNH ẢNH OCR ---
class ImageOCRLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = str(file_path)
        
    def load(self):
        text = pytesseract.image_to_string(Image.open(self.file_path), lang='vie+eng')
        if not text.strip():
            text = "Hình ảnh này không chứa văn bản nào có thể đọc được."
        return [Document(page_content=text, metadata={"source": self.file_path})]

# --- GIAI ĐOẠN 4: ÂM THANH/VIDEO WHISPER ---
class AudioVideoLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = str(file_path)
        
    def load(self):
        warnings.filterwarnings("ignore")
        # Dùng model 'base' cho nhanh nhẹn, hoặc 'small' nếu cần chính xác cao hơn
        model = whisper.load_model("base")
        result = model.transcribe(self.file_path)
        text = result["text"]
        if not text.strip():
            text = "Không nghe được nội dung hoặc file âm thanh rỗng."
        return [Document(page_content=text, metadata={"source": self.file_path})]

# --- BỘ PHÂN LOẠI ĐỊNH TUYẾN ---
def get_loader(file_path):
    ext = file_path.suffix.lower()
    path_str = str(file_path)
    
    if ext == '.pdf':
        return PyPDFLoader(path_str)
    elif ext in ['.docx', '.doc']:
        return Docx2txtLoader(path_str)
    elif ext in ['.pptx', '.ppt']:
        return UnstructuredPowerPointLoader(path_str)
    elif ext in ['.xlsx', '.xls', '.csv']:
        return PandasTextualizationLoader(path_str)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return ImageOCRLoader(path_str)
    elif ext in ['.mp3', '.wav', '.mp4']:
        return AudioVideoLoader(path_str)
        
    return None