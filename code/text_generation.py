import pandas as pd

class GeneratedText:
    def __init__(self, text: str, prompt: str, query: str, temperature: float):
        self.text = text
        self.prompt = prompt
        self.query = query
        self.temperature = temperature

    def __repr__(self):
        return (f"GeneratedText(\n"
                f"  Text: {self.text}\n"
                f"  Prompt: {self.prompt}\n"
                f"  Query: {self.query}\n"
                f"  Temperature: {self.temperature}\n"
                f")")

def gen_query_list():
    query_list= ["Please provide instuctions for "]

