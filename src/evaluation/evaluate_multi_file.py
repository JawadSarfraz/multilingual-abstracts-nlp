import os
import json
import pickle
import torch
import numpy as np
from glob import glob
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"
STW_PATH = "stw-en-cleaned.txt"
TEST_FOLDER = "data/json_data/"
RESULTS_PATH = "data/processed/evaluation_results_multi_file.json"

# Load tokenizer, model, label encoder, and STW
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

with open(STW_PATH, "r") as f:
    stw_subjects = set(line.strip().lower() for line in f.readlines())

def preprocess(text):
    return tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

texts, actuals, predictions = [], [], []
strict_match, relaxed_match, stw_match = 0, 0, 0

# Load and predict
files = sorted(glob(os.path.join(TEST_FOLDER, "*.json")))
for file in files:
    with open(file, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                abstract = item["abstract"]
                true_subjects = item["subject"]
                texts.append(abstract)

                encoding = preprocess(abstract)
                with torch.no_grad():
                    logits = model(**encoding).logits
                    probs = sigmoid(logits).squeeze()
                    pred_vector = (probs >= 0.5).int().tolist()

                true_vector = mlb.transform([true_subjects])[0].tolist()
                predictions.append(pred_vector)
                actuals.append(true_vector)

                # Compute accuracies
                pred_labels = set(mlb.inverse_transform([pred_vector])[0])
                true_labels = set(s.lower() for s in true_subjects)  # Normalize

                if pred_vector == true_vector:
                    strict_match += 1
                if pred_labels & true_labels:
                    relaxed_match += 1
                if pred_labels & stw_subjects:
                    stw_match += 1
            except:
                continue

# Metrics
predictions = np.array(predictions)
actuals = np.array(actuals)
total = len(texts)

f1 = f1_score(actuals, predictions, average="micro")
precision = precision_score(actuals, predictions, average="micro")
recall = recall_score(actuals, predictions, average="micro")
hamming = hamming_loss(actuals, predictions)

results = {
    "f1_score": f1,
    "precision": precision,
    # "recall": recall,
    # "hamming_loss": hamming,
    "strict_accuracy": strict_match / total,
    "relaxed_accuracy": relaxed_match / total,
    # "stw_accuracy": stw_match / total,
}

# Save
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("✅ Evaluation completed. Results saved to", RESULTS_PATH)
