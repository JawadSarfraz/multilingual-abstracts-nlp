import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import random
from torch.nn import BCEWithLogitsLoss

DATA_DIR = "data/filtered"
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "roberta_trained_model")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_filtered.pkl")
WEIGHTS_PATH = os.path.join(DATA_DIR, "pos_weights.pt")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
VAL_FILE = os.path.join(DATA_DIR, "val.json")
BATCH_SIZE = 4
EPOCHS = 5  # Increased epochs
TRAIN_SAMPLE_SIZE = 10000  # Increase sample size for better learning
LEARNING_RATE = 2e-5

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def load_data_list(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    texts = [item["abstract"] for item in data]
    labels = [item["subject"] for item in data]
    return texts, labels

class SubjectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, mlb, max_length=512):
        self.texts = texts
        self.labels = mlb.transform(labels)
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
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    preds = (preds >= 0.1).astype(int)  # Lowered threshold to 0.1
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)
    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

# Custom Trainer for Weighted Loss
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.load(WEIGHTS_PATH).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

def train_model():
    train_texts, train_labels = load_data_list(TRAIN_FILE)
    val_texts, val_labels = load_data_list(VAL_FILE)
    # Optionally subsample training data for quick experiments
    if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(train_texts):
        indices = random.sample(range(len(train_texts)), TRAIN_SAMPLE_SIZE)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"Using a random subset of {TRAIN_SAMPLE_SIZE} training samples.")
    # We load the encoder that was saved by the weight calculation script
    with open(ENCODER_PATH, "rb") as f:
        mlb = pickle.load(f)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = SubjectDataset(train_texts, train_labels, tokenizer, mlb)
    val_dataset = SubjectDataset(val_texts, val_labels, tokenizer, mlb)

    # Debug: print a batch
    loader = DataLoader(train_dataset, batch_size=4)
    batch = next(iter(loader))
    print('Batch input_ids shape:', batch['input_ids'].shape)
    print('Batch labels shape:', batch['labels'].shape)
    print('Batch labels:', batch['labels'])
    print('Batch labels sum:', batch['labels'].sum(dim=1))

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(mlb.classes_)
    )
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=500,
        logging_dir=f"{MODEL_SAVE_PATH}/logs",
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1_score",
    )
    
    # Use the CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}")

    # Print example predictions on validation set
    print("\nExample predictions on validation set:")
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    n_print = 10
    printed = 0
    device = next(model.parameters()).device
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        pred_indices = (probs >= 0.1).nonzero()[0]
        pred_labels = [mlb.classes_[i] for i in pred_indices]
        top5_indices = probs.argsort()[-5:][::-1]
        top5_labels = [(mlb.classes_[i], float(probs[i])) for i in top5_indices]
        idx = printed
        print(f"\nSample {idx+1}:")
        print("Abstract:", val_texts[idx][:300], "...")
        print("True labels:", val_labels[idx])
        print("Predicted labels (threshold 0.1):", pred_labels)
        print("Top 5 predicted (label, prob):", top5_labels)
        printed += 1
        if printed >= n_print:
            break

if __name__ == "__main__":
    train_model() 