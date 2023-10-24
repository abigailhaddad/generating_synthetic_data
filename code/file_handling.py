import json
from task_generation import BaseTask, GeneratedText
from typing import List

def load_base_task_from_json(file_path: str) -> BaseTask:
    with open(file_path, 'r') as file:
        data = json.load(file)
        return BaseTask.from_dict(data)
    
def save_generated_texts_to_json(generated_texts: List[GeneratedText], file_name: str):
    data_list = [gt.to_dict() for gt in generated_texts]
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    
def save_to_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_generated_texts_from_json(file_path: str) -> List[GeneratedText]:
    with open(file_path, 'r') as file:
        data_list = json.load(file)
        return [GeneratedText.from_dict(data) for data in data_list]

