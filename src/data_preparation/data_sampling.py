import json
import random

DATA_PATH = "data/raw/data.json"
OUTPUT_PATH = "data/raw/sample_data.json"
SAMPLE_SIZE = 5000

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
            print(f"Error parsing line: {e}")

    with open(output_path, "w") as out_file:
        json.dump(sampled_data, out_file, indent=2)

    print(f"Sampled {sample_size} objects and saved to {output_path}")

if __name__ == "__main__":
    sample_data(DATA_PATH, OUTPUT_PATH, SAMPLE_SIZE)
