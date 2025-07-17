import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, 'data/raw/data.json')
WORDS_PATH = os.path.join(ROOT_DIR, 'stw-en-cleaned.txt')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data/filtered/english_stw_filtered.json')

ENG_CODE = 'eng'  # ISO 639-2 for English

def load_ndjson(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set([line.strip() for line in file.readlines()])

def check_subjects_in_words(subjects, words_set):
    return any(subject in words_set for subject in subjects)

def is_english(language_list):
    return ENG_CODE in language_list if isinstance(language_list, list) else False

def main():
    print(f"Loading data from {DATA_PATH}")
    words_set = load_words(WORDS_PATH)
    print(f"Loaded {len(words_set)} STW subject terms.")

    filtered_data = []
    count = 0
    for record in load_ndjson(DATA_PATH):
        count += 1
        # Language filter
        language = record.get('language', [])
        if not is_english(language):
            continue
        # Subject filter
        subjects = record.get('subject', [])
        if not check_subjects_in_words(subjects, words_set):
            continue
        # Abstract filter and formatting
        abstract_list = record.get('abstract', [])
        if not abstract_list or not isinstance(abstract_list, list):
            continue
        abstract = abstract_list[0].strip()
        if not abstract:
            continue
        # Only keep abstract and subject
        filtered_data.append({
            'abstract': abstract,
            'subject': subjects
        })
        if count % 100000 == 0:
            print(f"Processed {count} records...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"Filtered {len(filtered_data)} records saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 