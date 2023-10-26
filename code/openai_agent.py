import openai
from marvin import settings
from typing import Optional
import os
import random

class OpenAIAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.configure_openai(api_key=api_key)
    
    def configure_openai(self, llm_max_tokens=1500, llm_temperature=0.0, api_key=None, llm_model='openai/gpt-3.5-turbo'):
        settings.llm_max_tokens = llm_max_tokens
        settings.llm_temperature = llm_temperature
        openai.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        settings.llm_model = llm_model
    
    def call_openai(self, prompt_text: str, model: str = settings.llm_model) -> str:
        """Calls OpenAI with the provided prompt and returns the generated response."""
        messages = [{"role": "user", "content": prompt_text}]
        temperature = random.uniform(0, 1.0)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=250,
                timeout = settings.llm_request_timeout_seconds,
                temperature = temperature
            )
            return response.choices[0]['message']['content'].strip()
        except Exception as e:
            # Handle other exceptions
            print(f"Unexpected error: {e}")
            return ""  # Return an empty string or whatever default you prefer