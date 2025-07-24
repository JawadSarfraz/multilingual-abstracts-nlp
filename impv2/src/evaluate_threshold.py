import torch
import json
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Corrected paths:

DATA_DIR = "../data/filtered"

TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
VAL_FILE = os.path.join(DATA_DIR, "val.json")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_filtered.pkl")
WEIGHTS_PATH = os.path.join(DATA_DIR, "pos_weights.pt")


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "filtered")

VAL_FILE = os.path.join(DATA_DIR, "val.json")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_filtered.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "roberta_trained_model")
MAX_LENGTH = 256
BATCH_SIZE = 16

class EvalDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.texts = [item["abstract"] for item in data]
        self.labels = [item["labels_encoded"] for item in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def evaluate_threshold(model, loader, thresholds=np.arange(0.2, 0.8, 0.05)):
    all_labels, all_logits = [], []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            logits = model(**{k: v for k, v in batch.items()}).logits
            all_labels.append(labels)
            all_logits.append(logits.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = torch.sigmoid(torch.tensor(np.vstack(all_logits))).numpy()

    print("\nEvaluating Thresholds for Optimal F1 Score:\n")
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, average='micro')
        precision = precision_score(all_labels, preds, average='micro')
        recall = recall_score(all_labels, preds, average='micro')
        print(f"Threshold: {thresh:.2f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

def main():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    dataset = EvalDataset(VAL_FILE, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    evaluate_threshold(model, loader)

if __name__ == "__main__":
    main()
