from src.modules.embedding.processors import PandasTextualizationLoader
loader = PandasTextualizationLoader("Data/Data_Files/thongtintruong.csv")
docs = loader.load()
for d in docs[:5]:
    print("---")
    print(d.page_content)
