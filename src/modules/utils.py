import os
import json
from datetime import datetime

CHAT_HISTORY_FILE = "config/chat_history.json"

def load_all_chats():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat(chat_id, messages):
    all_chats = load_all_chats()
    all_chats[chat_id] = {
        "messages": messages,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=4)

def get_folder_map():
    return {
        'pdf': 'PDF_Files',
        'docx': 'Word_Files',
        'xlsx': 'Data_Files',
        'csv': 'Data_Files',
        'pptx': 'PowerPoint_Files',
        'jpg': 'Image_Files',
        'png': 'Image_Files',
        'mp3': 'Audio_Files',
        'wav': 'Audio_Files',
        'mp4': 'Video_Files'
    }