import json
import random
import os

DATA_PATH = "data/raw/data.json"
OUTPUT_PATH = "experiments/01_increase_data_size/data_20k.json"
SAMPLE_SIZE = 20000

def sample_data(file_path, output_path, sample_size):
    """Extracts a sample of data and saves it to a specified output path."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

    # Ensure sample size does not exceed data length
    sample_size = min(sample_size, len(data))
    sampled_data = random.sample(data, sample_size)

    # Write sampled data to output file
    with open(output_path, "w") as f:
        for obj in sampled_data:
            json.dump(obj, f)
            f.write("\n")

    print(f"Sampled {sample_size} objects and saved to {output_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sample_data(DATA_PATH, OUTPUT_PATH, SAMPLE_SIZE)
