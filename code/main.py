from prompts_for_model import list_of_prompts
from task_pipeline import TaskGenerationPipeline

if __name__ == "__main__":
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    pipeline.run(use_existing_files=False)
