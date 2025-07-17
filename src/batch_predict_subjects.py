import os
import json
import torch
import pickle
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
import numpy as np

MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"
INPUT_PATH = "data/filtered/english_stw_filtered.json"
OUTPUT_PATH = "data/filtered/english_stw_predicted.json"

# Load model, tokenizer, and label encoder
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)
model.eval()

def predict_subjects(abstract, threshold=0.5):
    encoding = tokenizer(
        abstract,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = sigmoid(logits).squeeze()
    predictions = (probs >= threshold).int().tolist()
    predictions = np.array([predictions])
    predicted_labels = mlb.inverse_transform(predictions)[0]
    return list(predicted_labels)

def main():
    print(f"Loading input from {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")
    results = []
    for i, record in enumerate(data):
        abstract = record["abstract"]
        true_subject = record["subject"]
        predicted_subject = predict_subjects(abstract)
        results.append({
            "abstract": abstract,
            "true_subject": true_subject,
            "predicted_subject": predicted_subject
        })
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} records...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 