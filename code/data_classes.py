from marvin import ai_fn
from typing import List

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
    def __init__(self, text: str, prompt: str, query: str, category: str = None, label: int = None):
        self.text = text
        self.prompt = prompt
        self.query = query
        self.category = category
        self.label = label

    def to_dict(self):
        return {
            "text": self.text,
            "prompt": self.prompt,
            "query": self.query,
            "category": self.category,
            "label": selcf.label
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data["text"],
            prompt=data["prompt"],
            query=data["query"],
            category=data["category"],
            label=data.get("label", None)
        )


