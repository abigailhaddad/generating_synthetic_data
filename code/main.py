import sys
from prompts_for_model import list_of_prompts
from task_pipeline import TaskGenerationPipeline
from labeling import Labeler

# This will be our global pipeline object.
pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)

def generate_texts():
    global pipeline
    generated_texts_result = pipeline.run(use_existing_files=False)
    labeler = Labeler(generated_texts_result, pipeline.base_task)
    labeler.prepare_for_manual_labeling()

def process_labeled_texts():
    # Load the pipeline with existing generated_texts data
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    pipeline.load_data_from_files()

    # We use the loaded generated_texts for the Labeler.
    generated_texts_result = pipeline.generated_texts

    # Load labels directly from the CSV.
    labeler = Labeler(generated_texts_result, "How to cook pasta puttanesca")
    labeler.import_from_csv('../results/texts_for_labeling.csv')

    # Save the labeled texts back to the file.
    pipeline.save_generated_texts()

    for text in generated_texts_result:
        print(text.category)
        print(text.label)

    return generated_texts_result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a command: generate or process")
    elif sys.argv[1] == "generate":
        generate_texts()
    elif sys.argv[1] == "process":
        process_labeled_texts()
    else:
        print(f"Unknown command: {sys.argv[1]}")
