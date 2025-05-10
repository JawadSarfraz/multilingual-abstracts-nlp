import json
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/sample_data.json"
OUTPUT_DIR = "data/processed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.json")
ENCODER_FILE = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Basic text cleaning."""
    return text.lower().strip()

def preprocess_data(file_path):
    """Extract relevant fields and clean text."""
    abstracts = []
    subjects = []

    with open(file_path, "r") as f:
        data = json.load(f)
        for item in data:
            abstract = item.get("abstract", [""])[0]
            subject = item.get("subject", [])
            
            if abstract and subject:
                abstracts.append(clean_text(abstract))
                subjects.append(subject)

    return abstracts, subjects

def encode_labels(subjects):
    """Encode subjects using MultiLabelBinarizer."""
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(subjects)
    return labels, mlb

def split_and_save_data(abstracts, labels):
    """Split data and save to files."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        abstracts, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Save the datasets
    datasets = {
        "train": {"X": X_train, "y": y_train.tolist()},
        "val": {"X": X_val, "y": y_val.tolist()},
        "test": {"X": X_test, "y": y_test.tolist()},
    }

    for split, data in datasets.items():
        output_path = os.path.join(OUTPUT_DIR, f"{split}.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

def main():
    abstracts, subjects = preprocess_data(DATA_PATH)
    labels, mlb = encode_labels(subjects)

    # Save label encoder
    with open(ENCODER_FILE, "wb") as f:
        import pickle
        pickle.dump(mlb, f)

    split_and_save_data(abstracts, labels)
    print("Data preprocessing complete. Data saved in 'data/processed/'")

if __name__ == "__main__":
    main()
