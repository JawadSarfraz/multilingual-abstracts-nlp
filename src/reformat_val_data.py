import os
import json
import pickle
import numpy as np

# Paths
VAL_PATH = "data/processed/val.json"
VAL_FIXED_PATH = "data/processed/val_fixed.json"
ENCODER_PATH = "data/processed/label_encoder.pkl"

def reformat_val_data():
    # Load the label encoder
    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)

    # Read the input JSON
    with open(VAL_PATH, "r") as f:
        data = json.load(f)

    # Ensure correct format
    if not ("X" in data and "y" in data):
        print("Error: Expected 'X' and 'y' keys in the input JSON")
        return

    abstracts = data["X"]
    labels = data["y"]

    if len(abstracts) != len(labels):
        print("Error: Length of 'X' and 'y' must be the same")
        return

    # Open the output file for writing
    with open(VAL_FIXED_PATH, "w") as f_out:
        for abstract, label_vector in zip(abstracts, labels):
            # Convert label vector to list of subject strings
            subject_indices = [idx for idx, val in enumerate(label_vector) if val == 1]
            subjects = list(mlb.inverse_transform(np.array([label_vector]))[0])


            # Construct the JSON object
            data_point = {
                "abstract": abstract,
                "subject": subjects
            }

            # Write to file as JSONL
            f_out.write(json.dumps(data_point) + "\n")

    print(f"Data successfully reformatted and saved to {VAL_FIXED_PATH}")

if __name__ == "__main__":
    reformat_val_data()
