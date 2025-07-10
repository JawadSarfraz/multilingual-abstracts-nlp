import os
import json
import pickle
import numpy as np
from sklearn.model_selection import KFold
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Paths
DATA_PATH = "data/processed/all_data.json"  # Placeholder, update as needed
ENCODER_PATH = "data/processed/label_encoder.pkl"
MODEL_BASE_PATH = "data/processed/cv_models/"

# Ensure output directory exists
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

# Load label encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

# Load all data (to be implemented: load all abstracts and labels)
def load_all_data():
    # Load and concatenate train, val, and test splits
    data_dir = "data/processed"
    splits = ["train.json", "val.json", "test.json"]
    all_texts = []
    all_labels = []
    for split in splits:
        with open(os.path.join(data_dir, split), "r") as f:
            data = json.load(f)
            all_texts.extend(data["X"])
            all_labels.extend(data["y"])
    return all_texts, all_labels

class SubjectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

if __name__ == "__main__":
    # Load all data
    texts, labels = load_all_data()
    texts = np.array(texts)
    labels = np.array(labels)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        X_train, X_val = texts[train_idx], texts[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Initialize tokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        # Create datasets
        train_dataset = SubjectDataset(X_train, y_train, tokenizer)
        val_dataset = SubjectDataset(X_val, y_val, tokenizer)

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        # Training logic will be added next 