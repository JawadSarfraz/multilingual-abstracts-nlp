import os
import json
import pickle
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np

MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"
VAL_PATH = "data/processed/val_fixed.json"
RESULTS_PATH = "data/processed/evaluation_results.json"

# Load tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Load the label encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

# Set model to evaluation mode
model.eval()

def preprocess_text(text, max_length=512):
    """Tokenize and preprocess the input text."""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding

def evaluate_model():
    # Load the validation data
    texts = []
    true_labels = []
    
    with open(VAL_PATH, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                texts.append(item["abstract"])
                true_labels.append(item["subject"])
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
    
    # Preprocess texts and obtain predictions
    predictions = []
    
    for text in texts:
        encoding = preprocess_text(text)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = sigmoid(logits).squeeze()
            preds = (probs >= 0.5).int().tolist()
            predictions.append(preds)

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Compute metrics
    f1 = f1_score(true_labels, predictions, average="micro")
    precision = precision_score(true_labels, predictions, average="micro")
    recall = recall_score(true_labels, predictions, average="micro")
    hamming = hamming_loss(true_labels, predictions)

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
