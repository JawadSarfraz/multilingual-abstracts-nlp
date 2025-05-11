import os
import torch
import pickle
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.nn.functional import sigmoid

MODEL_PATH = "data/processed/trained_model"
ENCODER_PATH = "data/processed/label_encoder.pkl"

# Load tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)

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

def predict_subjects(abstract, threshold=0.5):
    """Predict subjects for the given abstract."""
    # Preprocess the input
    encoding = preprocess_text(abstract)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = sigmoid(logits).squeeze()

    # Apply threshold to determine active labels
    predictions = (probs >= threshold).int().tolist()

    # Check structure of predictions
    print(f"Predictions: {predictions}")
    print(f"Type of predictions: {type(predictions)}")

    # Decode labels - Ensure predictions is a 2D array
    predicted_labels = mlb.inverse_transform([predictions])[0]

    return predicted_labels

def main():
    # Example abstract
    abstract = """
    This study explores the impact of climate change on urban migration patterns in Europe, focusing on socioeconomic factors and policy implications.
    """

    # Predict subjects
    predicted_subjects = predict_subjects(abstract)
    print("\nPredicted Subjects:", predicted_subjects)

if __name__ == "__main__":
    main()
