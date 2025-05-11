import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "data/processed/trained_model"
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# Ensure output directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load Label Encoder
with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

# Load Data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["X"], data["y"]

class SubjectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(pred):
    """Custom metrics calculation for multi-label classification."""
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    preds = (preds >= 0.5).astype(int)

    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    f1 = f1_score(labels, preds, average="micro")

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def train_model():
    # Load data
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, "train.json"))
    val_texts, val_labels = load_data(os.path.join(DATA_DIR, "val.json"))

    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # Create datasets
    train_dataset = SubjectDataset(train_texts, train_labels, tokenizer)
    val_dataset = SubjectDataset(val_texts, val_labels, tokenizer)

    # Initialize model
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(mlb.classes_)
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=500,
        logging_dir=f"{MODEL_SAVE_PATH}/logs",
        load_best_model_at_end=True,
        evaluation_strategy="epoch",  # Align both strategies
        save_strategy="epoch",         # Align both strategies
        metric_for_best_model="f1_score",
    )



    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
