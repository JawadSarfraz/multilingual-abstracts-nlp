import pickle
import os

# Path to the label encoder
ENCODER_PATH = "data/processed/label_encoder.pkl"

def display_subjects():
    # Load the label encoder
    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)

    # Display the subjects list
    print("Subjects List:")
    for index, subject in enumerate(mlb.classes_):
        print(f"{index}: {subject}")

if __name__ == "__main__":
    display_subjects()
