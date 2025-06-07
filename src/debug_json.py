import json

file_path = "data/raw/data.json"

with open(file_path, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        print("✅ JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print(f"Line {e.lineno}, Column {e.colno}, Char {e.pos}")
