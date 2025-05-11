import os
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np
import json

# Paths
DATA_DIR = "experiments/01_increase_data_size/processed/"
MODEL_SAVE_PATH = "experiments/01_increase_data_size/trained_model/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load data
input_ids = torch.load(os.path.join(DATA_DIR, "input_ids.pt"))
attention_masks = torch.load(os.path.join(DATA_DIR, "attention_masks.pt"))
labels = torch.load(os.path.join(DATA_DIR, "labels.pt"))

# Split data
dataset_size = len(input_ids)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = torch.utils.data.TensorDataset(input_ids[:train_size], attention_masks[:train_size], labels[:train_size])
val_dataset = torch.utils.data.TensorDataset(input_ids[train_size:], attention_masks[train_size:], labels[train_size:])

# Model
MODEL_NAME = "xlm-roberta-base"
num_labels = labels.shape[1]

model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits >= 0.5).astype(int)

    f1 = f1_score(labels, predictions, average="micro")
    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    hamming = hamming_loss(labels, predictions)

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "hamming_loss": hamming,
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{MODEL_SAVE_PATH}/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_score",
)

# Trainer
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
print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")
