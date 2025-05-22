import openai
import json
import os

# Load API key
openai.api_key = os.getenv("OPENAI_API_KEY") 

DATA_PATH = 'data/raw/data.json'
OUTPUT_PATH = 'data/prompt_responses/gpt_subject_predictions.json'

PROMPT_TEMPLATE = """
Given the following abstract, guess relevant economic subject labels:

Abstract:
{abstract}
Subjects:
"""

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def query_gpt(abstract):
    prompt = PROMPT_TEMPLATE.format(abstract=abstract)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in classifying academic abstracts into economic subjects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    data = load_json(DATA_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []
    for record in data[:20]:  # Only run on first 20 to avoid quota drain
        abstract_text = " ".join(record['abstract']) if isinstance(record['abstract'], list) else record.get('abstract', '')
        prediction = query_gpt(abstract_text)
        results.append({
            "econbiz_id": record.get("econbiz_id"),
            "abstract": abstract_text,
            "gpt_subjects": prediction
        })

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved GPT predictions for {len(results)} abstracts to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
