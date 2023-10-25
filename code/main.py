from prompts_for_model import list_of_prompts
from task_pipeline import TaskGenerationPipeline
from labeling import Labeler

if __name__ == "__main__":
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    generated_texts_result = pipeline.run(use_existing_files=True)
    
    labeler = Labeler(generated_texts_result, pipeline.base_task)
    labeler.run()
    
    for text in generated_texts_result:
        print(text.category)
        print(text.label)
    
