import pandas as pd
import json
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        return pd.DataFrame(data)

def split_data(df, is_base_task, test_size=0.5, random_state=42):
    filtered_df = df[df['category'] == "base_task"] if is_base_task else df[df['category'] != "base_task"]
    unique_prompts = filtered_df['prompt'].unique()
    unique_queries = filtered_df['query'].unique()
    train_prompts, _ = train_test_split(unique_prompts, test_size=test_size, random_state=random_state)
    train_queries, _ = train_test_split(unique_queries, test_size=test_size, random_state=random_state)
    train_data = filtered_df[filtered_df['prompt'].isin(train_prompts) & filtered_df['query'].isin(train_queries)]
    test_data = filtered_df[~filtered_df['prompt'].isin(train_prompts) & ~filtered_df['query'].isin(train_queries)]
    return train_data, test_data

def main():
    file_path = "../results/generated_texts.json"
    df = load_data(file_path)
    
    train_base, test_base = split_data(df, is_base_task=True)
    train_non_base, test_non_base = split_data(df, is_base_task=False)
    
    train_df = pd.concat([train_base, train_non_base])
    test_df = pd.concat([test_base, test_non_base])
    
    # You can now proceed with training or save the split data
    train_df.to_csv("../results/train_data.csv", index=False)
    test_df.to_csv("../results/test_data.csv", index=False)

if __name__ == "__main__":
    main()



