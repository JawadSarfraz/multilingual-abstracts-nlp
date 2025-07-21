import json
import re
from collections import Counter

INPUT_PATH = "data/filtered/english_stw_filtered.json"
OUTPUT_PATH = "data/filtered/english_stw_filtered_cleaned.json"
MIN_SAMPLES_PER_LABEL = 50

# Heuristic: English labels are those with only ASCII letters, numbers, spaces, and common punctuation
ENGLISH_LABEL_PATTERN = re.compile(r'^[A-Za-z0-9 ,\-\'\"\(\)\[\]&/.:]+$')

def is_english_label(label):
    return bool(ENGLISH_LABEL_PATTERN.match(label))

def main():
    print(f"Loading data from {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f]

    # First pass: collect all English labels
    all_english_labels = []
    for obj in data:
        all_english_labels.extend([label for label in obj["subject"] if is_english_label(label)])
    label_counts = Counter(all_english_labels)
    
    # Keep labels that appear at least MIN_SAMPLES_PER_LABEL times
    frequent_labels = {label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_LABEL}
    print(f"Original unique English labels: {len(label_counts)}")
    print(f"Labels kept (freq >= {MIN_SAMPLES_PER_LABEL}): {len(frequent_labels)}")
    print(f"Labels removed as rare: {len(label_counts) - len(frequent_labels)}")

    cleaned_data = []
    original_abstract_count = len(data)

    for obj in data:
        english_labels = [label for label in obj["subject"] if is_english_label(label)]
        filtered_labels = [label for label in english_labels if label in frequent_labels]
        
        # Only keep abstracts that still have at least one label
        if filtered_labels:
            obj["subject"] = filtered_labels
            cleaned_data.append(obj)

    print(f"Original abstracts: {original_abstract_count}")
    print(f"Abstracts kept (with >=1 frequent label): {len(cleaned_data)}")
    print(f"Abstracts removed (all labels were rare): {original_abstract_count - len(cleaned_data)}")

    print(f"Saving cleaned data to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 