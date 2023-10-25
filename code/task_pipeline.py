import logging
from tqdm import tqdm
from openai_agent import OpenAIAgent
from file_manager import FileManager
from data_classes import BaseTask, GeneratedText
from typing import List
from marvin import settings
from prompts_for_model import list_of_prompts

# Set up logging
logging.basicConfig(level=logging.INFO)

class TaskGenerationPipeline:
    def __init__(self, base_task_str: str, prompts: List[str]):
        self.openai_agent = OpenAIAgent()
        self.base_task = BaseTask(base_task_str)
        self.prompts = prompts
        self.generated_texts = []
    
    def run(self, use_existing_files=False):
        if use_existing_files:
            self.load_data_from_files()
        else:
            self.generate_and_save_data()
    
    def generate_and_save_data(self):
        logging.info("Generating base task data...")
        base_task_data = self.base_task.to_dict()
        FileManager.save_to_json(base_task_data, '../results/base_task.json')
        logging.info("Generating texts for tasks...")
        self.generated_texts = self.generate_texts_for_task(self.base_task, self.prompts, base_task_runs=1, other_task_runs=1)
        all_generated_texts = [text.to_dict() for text in self.generated_texts]
        FileManager.save_to_json(all_generated_texts, '../results/generated_texts.json')

    
    def load_data_from_files(self):
        logging.info("Loading data from files...")
        self.base_task = BaseTask.from_dict(FileManager.load_from_json('../results/base_task.json'))
        loaded_texts = FileManager.load_from_json('../results/generated_texts.json')
        self.generated_texts = [GeneratedText.from_dict(data) for data in loaded_texts]

    def generate_texts_for_task(self, base_task: BaseTask, queries: List[str], model: str = None, base_task_runs: int = 1, other_task_runs: int = 1) -> List[GeneratedText]:
        if model is None:
            model = settings.llm_model

        # Select tasks with repetition
        all_tasks = [base_task.task] * base_task_runs + base_task.different_tasks * other_task_runs + base_task.similar_tasks * other_task_runs + base_task.others * other_task_runs

        # Generating responses for all combinations of queries and tasks, then storing them in GeneratedText instances
        generated_texts = []
        for query in queries[0:1]:
            for task in all_tasks:
                prompt = f"{query} {task}"
                response_text = self.openai_agent.call_openai(prompt, model) 
                generated_texts.append(GeneratedText(response_text, prompt, query))

        return generated_texts


if __name__ == "__main__":
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    pipeline.run(use_existing_files=False)
