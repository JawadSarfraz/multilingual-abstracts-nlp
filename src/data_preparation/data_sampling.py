import json
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATA_PATH = os.path.join(BASE_DIR, 'data/raw/data.json')
SAMPLE_SIZE = 10000  # Change this number anytime (e.g., 20000 or 50000)

OUTPUT_FILE_NAME = f"sample_data_{SAMPLE_SIZE}.json"
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", OUTPUT_FILE_NAME)

# DATA_PATH = "data/raw/data.json"

# Dynamically name output file based on sample size
# OUTPUT_PATH = os.path.join("data", "raw", OUTPUT_FILE_NAME)

def sample_data(input_path, output_path, sample_size):
    sampled_data = []

    with open(input_path, "r") as f:
        lines = f.readlines()
        sampled_lines = random.sample(lines, sample_size)

    for line in sampled_lines:
        try:
            obj = json.loads(line.strip())
            sampled_data.append(obj)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing line: {e}")

    with open(output_path, "w") as out_file:
        json.dump(sampled_data, out_file, indent=2)

    print(f"✅ Sampled {sample_size} objects and saved to {output_path}")

if __name__ == "__main__":
    sample_data(DATA_PATH, OUTPUT_PATH, SAMPLE_SIZE)
