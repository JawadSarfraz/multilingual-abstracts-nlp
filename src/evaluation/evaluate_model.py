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
    texts = []
    true_labels = []
    # Store match count for accuracy
    exact_match_count = 0
    
    with open(VAL_PATH, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if isinstance(item, dict):
                    texts.append(item["abstract"])
                    true_labels.append(item["subject"])
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
    
    print(f"Total Texts: {len(texts)}")
    print(f"Total Labels: {len(true_labels)}")

    predictions = []
    actual_labels = []

    for text, subjects in zip(texts, true_labels):
        # Preprocess the input text
        encoding = preprocess_text(text)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = sigmoid(logits).squeeze()
            preds = (probs >= 0.5).int().tolist()

        # Convert true labels to binary vector
        true_label_vector = mlb.transform([subjects])[0].tolist()

        predictions.append(preds)
        actual_labels.append(true_label_vector)
        if preds == true_label_vector:
            exact_match_count += 1

    # Convert to numpy arrays and ensure consistent shapes
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    

    # Compute metrics
    f1 = f1_score(actual_labels, predictions, average="micro")
    precision = precision_score(actual_labels, predictions, average="micro")
    recall = recall_score(actual_labels, predictions, average="micro")
    hamming = hamming_loss(actual_labels, predictions)
    accuracy = exact_match_count / len(predictions)  

    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy (Exact Match): {accuracy}")

    # print(f"Hamming Loss: {hamming}")

    # Save results
    results = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        #"hamming_loss": hamming,
        "accuracy": accuracy
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_model()

