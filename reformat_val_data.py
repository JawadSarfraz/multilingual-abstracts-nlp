import json

INPUT_PATH = "data/processed/val.json"
OUTPUT_PATH = "data/processed/val_fixed.json"

def reformat_data(input_path, output_path):
    """ Reformat data to NDJSON format. """
    with open(input_path, "r") as f:
        data = f.read()

    try:
        # Attempt to parse the entire file as a single JSON object
        json_data = json.loads(data)

        # Open the output file
        with open(output_path, "w") as f_out:
            for obj in json_data:
                json.dump(obj, f_out)
                f_out.write("\n")

        print(f"Data successfully reformatted and saved to {output_path}")

    except json.JSONDecodeError as e:
        print(f"Error processing the data: {e}")

if __name__ == "__main__":
    reformat_data(INPUT_PATH, OUTPUT_PATH)
