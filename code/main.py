import os
from task_generation import BaseTask, configure_openai, generate_texts_for_task
from prompts_for_model import list_of_prompts
from file_handling import save_to_json, load_base_task_from_json, save_generated_texts_to_json, load_generated_texts_from_json

configure_openai()
base_prompt_obj = BaseTask("How to cook pasta puttanesca")
print(base_prompt_obj)


base_task_data = base_prompt_obj.to_dict()
save_to_json(base_task_data, '../results/base_task.json')
# Usage
base_task_object = load_base_task_from_json('../results/base_task.json')

# Get generated texts
generated_texts = generate_texts_for_task(base_prompt_obj, list_of_prompts, base_task_runs=1, other_task_runs=1)

# Now, `generated_texts` will be a list of GeneratedText objects.
for text in generated_texts:
    print(text)

save_generated_texts_to_json(generated_texts, '../results/generated_texts.json')

# Loading from JSON:
loaded_texts = load_generated_texts_from_json('../results/generated_texts.json')
print(loaded_texts)