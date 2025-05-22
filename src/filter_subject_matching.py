import json
import os

# Paths
DATA_PATH = 'data/raw/sample_data.json'
WORDS_PATH = 'stw-en-cleaned.txt'
OUTPUT_PATH = 'data/filtered/matched_data.json'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set([line.strip() for line in file.readlines()])

def check_subjects_in_words(subjects, words_set):
    return any(subject in words_set for subject in subjects)

def main():
    # Load data and words
    data = load_json(DATA_PATH)
    words_set = load_words(WORDS_PATH)

    # Filter records where at least one subject matches
    filtered_data = []
    for record in data:
        subjects = record.get('subject', [])
        if check_subjects_in_words(subjects, words_set):
            filtered_data.append(record)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save filtered data
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Filtered {len(filtered_data)} matching records saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
