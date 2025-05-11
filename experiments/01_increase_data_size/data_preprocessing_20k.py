import os
import json
import numpy as np
import torch
from transformers import XLMRobertaTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = "experiments/01_increase_data_size/data_20k.json"
PROCESSED_DIR = "experiments/01_increase_data_size/processed/"
TOKENIZER_MODEL = "xlm-roberta-base"
MAX_LENGTH = 512

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(TOKENIZER_MODEL)

def preprocess_text(text, max_length=MAX_LENGTH):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding

def preprocess_data(data_path):
    abstracts = []
    subjects = []

    with open(data_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                abstract = " ".join(obj.get("abstract", []))
                subject_list = obj.get("subject", [])
                
                if abstract and subject_list:
                    abstracts.append(abstract)
                    subjects.append(subject_list)
            except json.JSONDecodeError:
                print(f"Error parsing line, skipping...")

    print(f"Total Samples Collected: {len(abstracts)}")

    # Encode labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(subjects)
    label_map = mlb.classes_

    # Save label map
    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump(label_map.tolist(), f)
    
    # Tokenize abstracts
    input_ids = []
    attention_masks = []

    for abstract in abstracts:
        encoding = preprocess_text(abstract)
        input_ids.append(encoding["input_ids"].squeeze().numpy())
        attention_masks.append(encoding["attention_mask"].squeeze().numpy())

    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    encoded_labels = torch.tensor(encoded_labels)

    # Save processed data
    torch.save(input_ids, os.path.join(PROCESSED_DIR, "input_ids.pt"))
    torch.save(attention_masks, os.path.join(PROCESSED_DIR, "attention_masks.pt"))
    torch.save(encoded_labels, os.path.join(PROCESSED_DIR, "labels.pt"))

    print(f"Data preprocessing complete. Data saved in '{PROCESSED_DIR}'")

if __name__ == "__main__":
    preprocess_data(DATA_PATH)
