import json
import os

DATA_PATH = "data/raw/data.json"

def analyze_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    num_objects = len(data)
    all_subjects = [subject for item in data for subject in item.get("subject", [])]
    unique_subjects = set(all_subjects)
    
    print(f"Total Objects (Abstracts): {num_objects}")
    print(f"Total Unique Subjects: {len(unique_subjects)}")
    print(f"Sample Subjects: {list(unique_subjects)[:10]}")
    print(f"Data Sample: {data[0]}")
    
    return num_objects, unique_subjects

if __name__ == "__main__":
    analyze_data(DATA_PATH)