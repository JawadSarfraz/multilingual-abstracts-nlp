import json
import os

DATA_PATH = "data/raw/data.json"

def analyze_data(file_path):
    num_objects = 0
    all_subjects = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                num_objects += 1
                subjects = obj.get("subject", [])
                all_subjects.extend(subjects)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {num_objects + 1}: {e}")
                continue

    unique_subjects = set(all_subjects)
    
    print(f"Total Objects (Abstracts): {num_objects}")
    print(f"Total Unique Subjects: {len(unique_subjects)}")
    print(f"Sample Subjects: {list(unique_subjects)[:10]}")

if __name__ == "__main__":
    analyze_data(DATA_PATH)
