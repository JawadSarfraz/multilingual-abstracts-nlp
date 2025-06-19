import os
import json
import pickle
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np

# === Paths ===
MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"
VAL_PATH = "data/processed/val_fixed.json"
RESULTS_PATH = "data/processed/evaluation_results.json"

# === Load tokenizer and model ===
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# === Load the label encoder ===
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

model.eval()  # set model to inference mode

def preprocess_text(text, max_length=512):
    """Tokenize and preprocess the input text."""
    return tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

def relaxed_accuracy(predicted_labels, true_labels):
    """Returns 1 if any predicted subject matches ground truth."""
    return int(bool(set(predicted_labels) & set(true_labels)))

def evaluate_model():
    texts, true_labels = [], []

    # === Load validation data ===
    with open(VAL_PATH, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                texts.append(item["abstract"])
                true_labels.append(item["subject"])
            except Exception as e:
                print(f"Error parsing line: {e}")

    print(f"Total samples: {len(texts)}")

    predictions = []
    actuals = []
    exact_match_count = 0
    relaxed_match_count = 0

    for text, true_subjects in zip(texts, true_labels):
        # Tokenize input
        encoding = preprocess_text(text)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = sigmoid(outputs.logits).squeeze()
            pred_vector = (probs >= 0.5).int().tolist()

        # Encode ground truth
        true_vector = mlb.transform([true_subjects])[0].tolist()

        # Save predictions
        predictions.append(pred_vector)
        actuals.append(true_vector)

        # Exact match
        if pred_vector == true_vector:
            exact_match_count += 1

        # Relaxed match
        pred_labels = set(mlb.inverse_transform(np.array([pred_vector]))[0])
        true_labels_set = set(true_subjects)
        if pred_labels & true_labels_set:
            relaxed_match_count += 1

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # === Compute metrics ===
    f1 = f1_score(actuals, predictions, average="micro")
    precision = precision_score(actuals, predictions, average="micro")
    recall = recall_score(actuals, predictions, average="micro")
    hamming = hamming_loss(actuals, predictions)
    strict_accuracy = exact_match_count / len(texts)
    relaxed_acc = relaxed_match_count / len(texts)

    print("\n📊 Evaluation Metrics:")
    print(f"F1 Score (micro): {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Accuracy (Exact Match): {strict_accuracy:.4f}")
    print(f"✅ Relaxed Accuracy (at least one match): {relaxed_acc:.4f}")

    # === Save to JSON ===
    results = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hamming,
        "strict_accuracy": strict_accuracy,
        "relaxed_accuracy": relaxed_acc
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_model()
