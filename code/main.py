import sys
from prompts_for_model import list_of_prompts
from task_pipeline import TaskGenerationPipeline
from labeling import Labeler

def generate_texts():
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    generated_texts_result = pipeline.run(use_existing_files=False)
    labeler = Labeler(generated_texts_result, pipeline.base_task)
    labeler.prepare_for_manual_labeling()

def process_labeled_texts():
    # We initialize a placeholder for generated_texts_result.
    generated_texts_result = []

    # Load directly from the CSV.
    labeler = Labeler(generated_texts_result, "How to cook pasta puttanesca")  # base task can be set directly here.
    labeler.import_from_csv('../results/texts_for_labeling.csv')

    for text in generated_texts_result:
        print(text.category)
        print(text.label)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a command: generate or process")
    elif sys.argv[1] == "generate":
        generate_texts()
    elif sys.argv[1] == "process":
        process_labeled_texts()
    else:
        print(f"Unknown command: {sys.argv[1]}")
