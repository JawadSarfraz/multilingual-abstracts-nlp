import os
import json
import pickle
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"
VAL_PATH = "data/processed/val_fixed.json"
STW_PATH = "stw-test-data.txt"
RESULTS_PATH = "data/processed/evaluation_results.json"

# Load tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Load the label encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

# Load STW subject list
with open(STW_PATH, "r") as f:
    stw_subjects = set(line.strip() for line in f if line.strip())

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
    texts = []
    true_labels = []
    exact_match_count = 0
    relaxed_match_count = 0
    stw_match_count = 0
    
    with open(VAL_PATH, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if isinstance(item, dict):
                    texts.append(item["abstract"])
                    true_labels.append(item["subject"])
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

    predictions = []
    actuals = []

    for text, subjects in zip(texts, true_labels):
        encoding = preprocess_text(text)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = sigmoid(logits).squeeze()
            preds = (probs >= 0.5).int().tolist()

        pred_subjects = mlb.inverse_transform(np.array([preds]))[0]
        actual_subjects = subjects

        # Binary vectors for sklearn metrics
        predictions.append(preds)
        actuals.append(mlb.transform([actual_subjects])[0].tolist())

        # Strict match
        if set(pred_subjects) == set(actual_subjects):
            exact_match_count += 1
        
        # Relaxed match (at least one overlap)
        if any(p in actual_subjects for p in pred_subjects):
            relaxed_match_count += 1

        # STW-based match (at least one predicted subject in STW)
        if any(p in stw_subjects for p in pred_subjects):
            stw_match_count += 1

    # Convert to arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Compute metrics
    f1 = f1_score(actuals, predictions, average="micro")
    precision = precision_score(actuals, predictions, average="micro")
    recall = recall_score(actuals, predictions, average="micro")
    hamming = hamming_loss(actuals, predictions)
    strict_accuracy = exact_match_count / len(texts)
    relaxed_accuracy = relaxed_match_count / len(texts)
    stw_accuracy = stw_match_count / len(texts)

    print("\n🔍 Evaluation Metrics:")
    print(f"F1 Score (micro): {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Strict Accuracy (Exact Match): {strict_accuracy:.4f}")
    print(f"Relaxed Accuracy (At least one match): {relaxed_accuracy:.4f}")
    print(f"STW Accuracy (Predicted ∩ STW ≠ ∅): {stw_accuracy:.4f}")

    results = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hamming,
        "strict_accuracy": strict_accuracy,
        "relaxed_accuracy": relaxed_accuracy,
        "stw_accuracy": stw_accuracy
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_model()
