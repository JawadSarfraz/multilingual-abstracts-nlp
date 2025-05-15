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
                if isinstance(item, dict):
                    texts.append(item.get("abstract", ""))
                    true_labels.append(item.get("subject", []))
                else:
                    print(f"Skipping line as it is not a dictionary: {line}")
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

    print(f"Total Texts: {len(texts)}")
    print(f"Total Labels: {len(true_labels)}")

    predictions = []
    actual_labels = []

    for i, text in enumerate(texts):
        encoding = preprocess_text(text)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = sigmoid(logits).squeeze()
            preds = (probs >= 0.5).int().tolist()

        # Get the relevant subjects (filter)
        relevant_subjects = true_labels[i]
        relevant_indices = mlb.transform([relevant_subjects])[0]

        # Apply mask to predictions and ground truth
        relevant_preds = [preds[idx] for idx in range(len(preds)) if relevant_indices[idx] == 1]
        relevant_truth = [1 if idx in relevant_indices.nonzero()[0] else 0 for idx in range(len(preds))]

        predictions.append(relevant_preds)
        actual_labels.append(relevant_truth)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)

    # Compute metrics
    f1 = f1_score(actual_labels, predictions, average="micro")
    precision = precision_score(actual_labels, predictions, average="micro")
    recall = recall_score(actual_labels, predictions, average="micro")
    hamming = hamming_loss(actual_labels, predictions)

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
