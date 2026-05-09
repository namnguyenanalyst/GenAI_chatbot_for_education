from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader, 
    UnstructuredExcelLoader,
    CSVLoader
)

def get_loader(file_path):
    ext = file_path.suffix.lower()
    path_str = str(file_path) # Chuyển về string một lần duy nhất
    
    if ext == '.pdf':
        return PyPDFLoader(path_str)
    elif ext in ['.docx', '.doc']:
        return Docx2txtLoader(path_str)
    elif ext in ['.pptx', '.ppt']:
        return UnstructuredPowerPointLoader(path_str)
    elif ext in ['.xlsx', '.xls']:
        return UnstructuredExcelLoader(path_str, mode="elements")  
    elif ext == '.csv':
        # Thêm autodetect encoding nếu utf-8 lỗi
        return CSVLoader(
            file_path=path_str,
            encoding='utf-8', 
            csv_args={'delimiter': ','} 
        )
    return None