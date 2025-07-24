import json
import torch
import pickle
import random
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = "../data/filtered"
MODEL_PATH = f"{DATA_DIR}/roberta_trained_model"
TRAIN_FILE = f"{DATA_DIR}/train.json"
ENCODER_PATH = f"{DATA_DIR}/label_encoder_filtered.pkl"
THRESHOLD = 0.2  # Use best threshold from evaluation

# Load tokenizer, model, label encoder
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

with open(ENCODER_PATH, "rb") as f:
    mlb = pickle.load(f)

# Load 20 random abstracts from train set
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)
samples = random.sample(data, 20)

for i, sample in enumerate(samples, 1):
    abstract = sample["abstract"]
    inputs = tokenizer(abstract, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()

    predicted_indices = np.where(probs >= THRESHOLD)[0]
    predicted_subjects = mlb.classes_[predicted_indices]

    print(f"\n--- Sample {i} ---")
    print(f"Abstract:\n{abstract[:300]}...")  # Show first 300 chars
    print(f"Predicted Subjects: {list(predicted_subjects)}")
