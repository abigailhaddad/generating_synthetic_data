import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import datasets
import evaluate
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Load data from JSON file
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        return pd.DataFrame(data)

# Split data based on whether it's base task or not
def split_data(df, is_base_task, test_size=0.5, random_state=42):
    filtered_df = df[df['category'] == "base_task"] if is_base_task else df[df['category'] != "base_task"]
    unique_prompts = filtered_df['prompt'].unique()
    unique_queries = filtered_df['query'].unique()
    train_prompts, _ = train_test_split(unique_prompts, test_size=test_size, random_state=random_state)
    train_queries, _ = train_test_split(unique_queries, test_size=test_size, random_state=random_state)
    train_data = filtered_df[filtered_df['prompt'].isin(train_prompts) & filtered_df['query'].isin(train_queries)]
    test_data = filtered_df[~filtered_df['prompt'].isin(train_prompts) & ~filtered_df['query'].isin(train_queries)]
    return train_data, test_data

# Load data from CSV files
def load_split_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Create datasets for HuggingFace Trainer
def create_hf_datasets(train_df, test_df):
    d_train = datasets.Dataset.from_pandas(train_df)
    d_test = datasets.Dataset.from_pandas(test_df)
    d = datasets.DatasetDict({"train": d_train, "test": d_test})
    return d

# Tokenize the datasets
def tokenize_data(dataset, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer

# Setup training parameters
def training_setup(model_checkpoint, id2label, label2id):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    training_args = TrainingArguments(
        output_dir="../model/deberta-v3-base-injection",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    print(f"Resolved output directory: {os.path.abspath(training_args.output_dir)}")
    return model, training_args

# Calculate metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)

# Train the model
def train_model(tokenized_dataset, model, tokenizer, training_args):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

def predict_and_save_results(trainer, dataset, df, output_path, label_map):
    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=1)
    
    # Add predictions to the dataframe
    df['prediction'] = predictions
    df['prediction_label'] = df['prediction'].map(label_map)
    
    # Save the updated dataframe with predictions to a CSV file
    df.to_csv(output_path, index=False)

    return df

# A helper function to print the confusion matrix
def print_confusion_matrix(confusion_matrix, label_classes):
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("\nClassification Report:")
    print(classification_report(confusion_matrix, target_names=label_classes))

def main():
    # Load and split the data
    file_path = "../results/generated_texts.json"
    df = load_data(file_path)
    train_base, test_base = split_data(df, is_base_task=True)
    train_non_base, test_non_base = split_data(df, is_base_task=False)
    train_df = pd.concat([train_base, train_non_base])
    test_df = pd.concat([test_base, test_non_base])

    # Define the mappings for labels
    id2label = {0: "no_instructions", 1: "contains_instructions"}
    label2id = {"contains_instructions": 0, "no_instructions": 1}

    # Tokenize the data
    model_checkpoint = "microsoft/deberta-v3-base"
    dataset = create_hf_datasets(train_df, test_df)
    tokenized_dataset, tokenizer = tokenize_data(dataset, model_checkpoint)
    
    # Setup training parameters and train the model
    model, training_args = training_setup(model_checkpoint, id2label, label2id)
    trainer = train_model(tokenized_dataset, model, tokenizer, training_args)

    # Save the trained model
    model_save_path = "../model/trained_model"
    trainer.save_model(model_save_path)

    # Evaluate the model on the test dataset and save the results
    eval_results = trainer.evaluate(tokenized_dataset["test"])
    with open("../results/evaluation_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    print("Evaluation Results on Test Data:", eval_results)

    # Save predictions for training and test data
    train_predictions_output_path = "../results/train_data_with_predictions.csv"
    test_predictions_output_path = "../results/test_data_with_predictions.csv"
    predict_and_save_results(trainer, tokenized_dataset["train"], train_df, train_predictions_output_path, trainer.model.config.id2label)
    predict_and_save_results(trainer, tokenized_dataset["test"], test_df, test_predictions_output_path, trainer.model.config.id2label)
    
    return eval_results
    

if __name__ == "__main__":
    eval_results = main()

    

# Run the main function
if __name__ == "__main__":
    eval_results=main()

