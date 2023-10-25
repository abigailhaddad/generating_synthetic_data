import sys
from prompts_for_model import list_of_prompts
from task_pipeline import TaskGenerationPipeline
from labeling import Labeler

# This will be our global pipeline object.
pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)

def generate_texts(base_task_runs=1, other_task_runs=1):
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    generated_texts_result = pipeline.run(use_existing_files=False, base_task_runs=base_task_runs, other_task_runs=other_task_runs)
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
        base_task_runs = 1
        other_task_runs = 1

        # If more arguments are provided, use them to update the run counts
        if len(sys.argv) > 2:
            base_task_runs = int(sys.argv[2])
        if len(sys.argv) > 3:
            other_task_runs = int(sys.argv[3])

        generate_texts(base_task_runs=base_task_runs, other_task_runs=other_task_runs)

    elif sys.argv[1] == "process":
        process_labeled_texts()
    else:
        print(f"Unknown command: {sys.argv[1]}")

