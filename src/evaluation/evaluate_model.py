import os
import json
import torch
import pickle
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np

# Paths
MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"
VAL_PATH = "data/processed/val.json"
RESULTS_PATH = "data/processed/evaluation_results.json"

# Load model and tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Load label encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

# Set model to evaluation mode
model.eval()

def preprocess_text(texts, max_length=512):
    """Tokenize and preprocess the input texts."""
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding

def evaluate_model():
    # Load validation data
    with open(VAL_PATH, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    texts = [item["abstract"] for item in data]
    true_labels = [item["subject"] for item in data]

    # Encode labels using MultiLabelBinarizer
    true_labels = mlb.transform(true_labels)

    # Preprocess text data
    encoding = preprocess_text(texts)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = sigmoid(logits).cpu().numpy()

    # Apply threshold
    threshold = 0.5
    predictions = (probs >= threshold).astype(int)

    # Calculate metrics
    f1 = f1_score(true_labels, predictions, average="micro")
    precision = precision_score(true_labels, predictions, average="micro")
    recall = recall_score(true_labels, predictions, average="micro")
    hamming = hamming_loss(true_labels, predictions)

    # Print results
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Hamming Loss: {hamming}")

    # Save results
    results = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hamming,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_model()
