import os
import openai
from marvin import ai_fn, settings
from typing import List


def configure_openai(llm_max_tokens=1500, llm_temperature=0.0, api_key=None, llm_model='openai/gpt-3.5-turbo'):
    import openai
    from marvin import settings  # Assuming this is the module where you set these settings
    settings.llm_max_tokens = llm_max_tokens
    # Use llm_max_context_tokens as required
    settings.llm_temperature = llm_temperature
    openai.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
    settings.llm_model = llm_model

@ai_fn
def generate_different_tasks(task: str) -> List[str]:
    """Given `task`, generate a list of ten totally different tasks that would be done in different spheres of life or by different people."""

@ai_fn
def generate_similar_tasks(task: str) -> List[str]:
    """Given `task`, generate a list of ten tasks which might involve similar objects, locations, or people."""

class BaseTask:
    def __init__(self, task: str):
        self.task = task
        self.different_tasks = generate_different_tasks(self.task)
        self.similar_tasks = generate_similar_tasks(self.task)
        self.others = ["a blue ballpoint pen", "the city bus stop", "the notion of luck", "Dayton, OH",
                       "getting rained on"]

    def to_dict(self):
        return {
            "task": self.task,
            "different_tasks": self.different_tasks,
            "similar_tasks": self.similar_tasks,
            "others": self.others
        }

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(task=data["task"])
        obj.different_tasks = data["different_tasks"]
        obj.similar_tasks = data["similar_tasks"]
        obj.others = data["others"]
        return obj
    
    def __repr__(self):
        return (f"BaseTask(\n"
                f"  Task: {self.task}\n"
                f"  Different Tasks: {self.different_tasks}\n"
                f"  Similiar Tasks: {self.similar_tasks}\n"
                f"  Other: {self.others}\n"
                f")")



class GeneratedText:
    def __init__(self, text: str, prompt: str, query: str):
        self.text = text
        self.prompt = prompt
        self.query = query

    def to_dict(self):
        return {
            "text": self.text,
            "prompt": self.prompt,
            "query": self.query
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data["text"],
            prompt=data["prompt"],
            query=data["query"]
        )

    def __repr__(self):
        return (f"GeneratedText(\n"
                f"  Text: {self.text}\n"
                f"  Prompt: {self.prompt}\n"
                f"  Query: {self.query}\n"
                f")")


def generate_texts_for_task(base_task: BaseTask, queries: List[str], model: str = None, base_task_runs: int = 1, other_task_runs: int = 1) -> List[GeneratedText]:
    if model is None:
        model = settings.llm_model

    # Select tasks with repetition
    all_tasks = [base_task.task] * base_task_runs + base_task.different_tasks * other_task_runs + base_task.similar_tasks * other_task_runs + base_task.others * other_task_runs

    # Generating responses for all combinations of queries and tasks, then storing them in GeneratedText instances
    generated_texts = []
    for query in queries[0:2]:
        for task in all_tasks[0:2]:
            prompt = f"{query} {task}"
            response_text = call_openai(prompt, model)
            generated_texts.append(GeneratedText(response_text, prompt, query))
            
    return generated_texts




def call_openai(prompt_text: str, model: str = settings.llm_model) -> str:
    """Calls OpenAI with the provided prompt and returns the generated response."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # bad, fix
            messages=messages,
            max_tokens=250,
            timeout = settings.llm_request_timeout_seconds,
            temperature=0.0  # You can adjust the temperature if needed
        )
        return response.choices[0]['message']['content'].strip()
    except Exception as e:
        # Handle other exceptions
        print(f"Unexpected error: {e}")
        return ""  # Return an empty string or whatever default you prefer
