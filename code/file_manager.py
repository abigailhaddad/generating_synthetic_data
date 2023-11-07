import json

class FileManager:
    @staticmethod
    def save_to_json(data, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    @staticmethod
    def load_from_json(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

