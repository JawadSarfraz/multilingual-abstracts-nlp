import json
import os
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle

INPUT_PATH = "data/filtered/english_stw_filtered.json"
OUTPUT_DIR = "data/filtered"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.json")
ENCODER_FILE = os.path.join(OUTPUT_DIR, "label_encoder_filtered.pkl")

SEED = 42
random.seed(SEED)

# Load data
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    try:
        print(f"Saving to {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving {path}: {e}")

def main():
    print(f"Loading data from {INPUT_PATH}")
    data = load_data(INPUT_PATH)
    print(f"Loaded {len(data)} records.")
    abstracts = [item["abstract"] for item in data]
    subjects = [item["subject"] for item in data]

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        abstracts, subjects, test_size=0.2, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Fit label encoder
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)
    try:
        print(f"Saving label encoder to {ENCODER_FILE}")
        with open(ENCODER_FILE, "wb") as f:
            pickle.dump(mlb, f)
    except Exception as e:
        print(f"Error saving label encoder: {e}")

    # Save splits
    save_json({"X": X_train, "y": mlb.transform(y_train).tolist()}, TRAIN_FILE)
    save_json({"X": X_val, "y": mlb.transform(y_val).tolist()}, VAL_FILE)
    save_json({"X": X_test, "y": mlb.transform(y_test).tolist()}, TEST_FILE)
    print(f"Saved train/val/test splits to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 