import os
import json
import torch
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import sigmoid
import numpy as np

MODEL_PATH = "data/filtered/roberta_trained_model"
ENCODER_PATH = "data/filtered/label_encoder_filtered.pkl"
TEST_PATH = "data/filtered/test.json"
OUTPUT_PATH = "data/filtered/test_predicted.json"

# Load model, tokenizer, and label encoder
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)
model.eval()

def predict_subjects(abstract, threshold=0.2, debug=False):
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
        probs = sigmoid(logits).squeeze().cpu().numpy()
    predictions = (probs >= threshold).astype(int)
    predictions = np.array([predictions])
    predicted_labels = mlb.inverse_transform(predictions)[0]
    if debug:
        print("Raw probabilities:", probs)
        print("Predicted labels:", predicted_labels)
    return list(predicted_labels)

def main():
    print(f"Loading test set from {TEST_PATH}")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    X = data["X"]
    y = data["y"]
    results = []
    for i, (abstract, label_vec) in enumerate(zip(X, y)):
        true_subject = list(mlb.inverse_transform(np.array([label_vec]))[0])
        debug = i < 15  # Print debug info for first 5 examples
        predicted_subject = predict_subjects(abstract, threshold=0.01, debug=debug)
        results.append({
            "abstract": abstract,
            "true_subject": true_subject,
            "predicted_subject": predicted_subject
        })
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} records...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 