import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def load_model_and_tokenizer(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def classify_text(model, tokenizer, text, device='cpu'):
    model.to(device)
    input_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

    with torch.no_grad():
        model.eval()
        logits = model(**input_tokens).logits
    probabilities = F.softmax(logits, dim=-1)
    return probabilities

def main():
    model_path = "../model/trained_model" # Update this to the path where your model is saved
    tokenizer_path = model_path # If your tokenizer is saved at the same place as your model

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    # Replace "cuda" with "cpu" if not using GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:  # Loop to allow continuous text inputs
        try:
            input_text = input("Enter the text you want to classify (or type 'exit' to quit): ")
            if input_text.lower() == 'exit':
                break

            probabilities = classify_text(model, tokenizer, input_text, device=device)
            class_0_prob, class_1_prob = probabilities.squeeze().tolist()

            print(f"The probabilities for each class are: class 'no_instructions': {class_0_prob:.3f}, class 'contains_instructions': {class_1_prob:.3f}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
