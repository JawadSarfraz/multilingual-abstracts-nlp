import json
import os

# Automatically resolve path to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute path
file_path = os.path.join(BASE_DIR, "data", "raw", "data.json")

with open(file_path, "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        print("✅ JSON is valid!")
    except json.JSONDecodeError as e:
        print("❌ JSON Decode Error:")
        print(f"  → Message: {e.msg}")
        print(f"  → Line: {e.lineno}, Column: {e.colno}, Char: {e.pos}")
