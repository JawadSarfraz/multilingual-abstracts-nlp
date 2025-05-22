import json
import os

DATA_PATH = 'data/raw/sample_data.json'
WORDS_PATH = 'stw-en-cleaned.txt'

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

    # Iterate over each record
    for record in data:
        subjects = record.get('subject', [])
        matched = check_subjects_in_words(subjects, words_set)
        print(f"Record ID: {record.get('econbiz_id', 'N/A')}, Match Found: {matched}")

if __name__ == "__main__":
    main()
