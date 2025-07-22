import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import RobertaTokenizer
import pickle
import torch
from collections import Counter
import os

import os

DATA_DIR = "../data/filtered"

RAW_DATA_FILE = os.path.join(DATA_DIR, "english_stw_filtered.json")
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "roberta_trained_model")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_filtered.pkl")
WEIGHTS_PATH = os.path.join(DATA_DIR, "pos_weights.pt")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
VAL_FILE = os.path.join(DATA_DIR, "val.json")

# DATA_FILE = "data.json"
# ENCODER_PATH = "label_encoder.pkl"
# TRAIN_PATH = "train.json"
# VAL_PATH = "val.json"
# WEIGHTS_PATH = "pos_weights.pt"

TOKENIZER_NAME = "roberta-base"
MAX_LENGTH = 256  # based on average abstract length ~158 words
TOP_N_LABELS = 50  # Adjust as needed based on label distribution

def load_and_prepare_data(data_file, top_n_labels=TOP_N_LABELS):
    # Load data
    with open(data_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Get all labels
    all_labels = [label for labels in df.subject for label in labels]
    label_counts = Counter(all_labels)

    # Keep only top_n_labels
    top_labels = set(label for label, _ in label_counts.most_common(top_n_labels))

    # Filter data
    df["filtered_labels"] = df["subject"].apply(lambda labels: [l for l in labels if l in top_labels])
    df = df[df["filtered_labels"].map(len) > 0]

    return df

def create_train_val_split(df, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, val_df

def encode_labels(train_labels, val_labels, encoder_path=ENCODER_PATH):
    mlb = MultiLabelBinarizer()
    train_encoded = mlb.fit_transform(train_labels)
    val_encoded = mlb.transform(val_labels)

    with open(encoder_path, "wb") as f:
        pickle.dump(mlb, f)

    return train_encoded, val_encoded, mlb

def compute_pos_weights(train_encoded, weights_path=WEIGHTS_PATH):
    label_freq = train_encoded.sum(axis=0)
    neg_freq = train_encoded.shape[0] - label_freq
    pos_weight = neg_freq / (label_freq + 1e-5)  # Avoid division by zero
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)
    torch.save(pos_weight_tensor, weights_path)

def tokenize_data(df, tokenizer_name=TOKENIZER_NAME, max_length=MAX_LENGTH):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(df["abstract"].tolist(), truncation=True, padding="max_length", max_length=max_length)
    return encodings

def save_json(df, labels_encoded, path):
    records = []
    for i in range(len(df)):
        records.append({
            "abstract": df["abstract"].iloc[i],
            "labels_encoded": labels_encoded[i].tolist(),
            "subject": df["filtered_labels"].iloc[i]
        })
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

def main():
    os.makedirs("data/processed", exist_ok=True)

    # Load and filter data
    df = load_and_prepare_data(RAW_DATA_FILE)

    # Split into train/validation sets
    train_df, val_df = create_train_val_split(df)

    # Encode labels
    train_encoded, val_encoded, mlb = encode_labels(train_df["filtered_labels"], val_df["filtered_labels"])

    # Compute class imbalance weights
    compute_pos_weights(train_encoded)

    # Tokenize data
    train_encodings = tokenize_data(train_df)
    val_encodings = tokenize_data(val_df)

    # Save processed data for training
    save_json(train_df, train_encoded, TRAIN_PATH)
    save_json(val_df, val_encoded, VAL_PATH)

    print("Data preprocessing and tokenization completed successfully!")

if __name__ == "__main__":
    main()
