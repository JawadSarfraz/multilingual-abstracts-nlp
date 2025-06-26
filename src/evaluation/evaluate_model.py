import os
import json
import pickle
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

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

model.eval()

def preprocess_text(text, max_length=512):
    return tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

def evaluate_model():
    texts, true_labels = [], []

    # === Load data ===
    with open(VAL_PATH, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                texts.append(item["abstract"])
                true_labels.append(item["subject"])
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    print(f"📄 Total samples: {len(texts)}")

    predictions = []
    actuals = []
    exact_match_count = 0
    relaxed_match_count = 0
    relaxed_f1_scores = []

    for text, true_subjects in zip(texts, true_labels):
        # === Inference ===
        encoding = preprocess_text(text)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = sigmoid(logits).squeeze()
            pred_vector = (probs >= 0.5).int().tolist()

        # === Ground truth encoding ===
        true_vector = mlb.transform([true_subjects])[0].tolist()
        predictions.append(pred_vector)
        actuals.append(true_vector)

        # === Exact match
        if pred_vector == true_vector:
            exact_match_count += 1

        # === Relaxed match
        pred_labels = set(mlb.inverse_transform([pred_vector])[0])
        true_labels_set = set(true_subjects)

        if pred_labels & true_labels_set:
            relaxed_match_count += 1

        # === Relaxed F1
        if not pred_labels or not true_labels_set:
            relaxed_f1_scores.append(0.0)
        else:
            intersection = pred_labels & true_labels_set
            if not intersection:
                relaxed_f1_scores.append(0.0)
            else:
                p_i = len(intersection) / len(pred_labels)
                r_i = len(intersection) / len(true_labels_set)
                f1_i = 2 * p_i * r_i / (p_i + r_i)
                relaxed_f1_scores.append(f1_i)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # === Metrics ===
    micro_f1 = f1_score(actuals, predictions, average="micro")
    precision = precision_score(actuals, predictions, average="micro")
    recall = recall_score(actuals, predictions, average="micro")
    hamming = hamming_loss(actuals, predictions)
    strict_acc = exact_match_count / len(texts)
    relaxed_acc = relaxed_match_count / len(texts)
    relaxed_f1 = sum(relaxed_f1_scores) / len(relaxed_f1_scores)

    # === Output ===
    print("\n📊 Evaluation Metrics:")
    print(f"F1 Score (micro): {micro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Strict Accuracy (Exact Match): {strict_acc:.4f}")
    print(f"Relaxed Accuracy (at least one match): {relaxed_acc:.4f}")
    print(f"🧠 Relaxed F1-score (overlap-aware): {relaxed_f1:.4f}")

    # === Save JSON ===
    results = {
        "f1_score_micro": micro_f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hamming,
        "strict_accuracy": strict_acc,
        "relaxed_accuracy": relaxed_acc,
        "relaxed_f1": relaxed_f1
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_model()
