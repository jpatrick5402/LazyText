from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
from typing import List

# --- Global Variables and Model Loading ---
MODEL_PATH = "./nlp_classifier_model.pth"
TOKENIZER_DIR = "./tokenizer_cache/"
MODEL_NAME = "bert-base-uncased"  # Use the model name that matches your training

# Initialize FastAPI app
app = FastAPI(
    title="NLP Classification API",
    description="Predicts the intent/category of a given text string.",
    version="1.0.0"
)

# --- Pydantic Schema for Input Validation ---
class TextPayload(BaseModel):
    """Schema for the request\ body."""
    texts: List[str]

# --- Model Loading Function ---
def load_model():
    """Loads the tokenizer and modeli weights into memory."""
    try:
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        
        # Load Model structure (You might need to adjust this depending on 
        # how you saved the final classifier layer)
        # For simplicity, we re-initialize the structure and load weights
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Set model to evaluation mode
        
        print("✅ Model and Tokenizer loaded successfully into memory.")
        return tokenizer, model
    except FileNotFoundError as e:
        print(f"🚨 CRITICAL ERROR: Could not find model files. Did you run train_and_save.py? Error: {e}")
        raise RuntimeError("Model assets not found. Please train the model first.")

# Load models immediately when the server starts
try:
    tokenizer, model = load_model()
except RuntimeError as e:
    # If loading fails, the server should not start cleanly
    print(f"Server failed to start due to model error: {e}")
    exit(1)


# --- API Endpoints ---

@app.get("/")
def root():
    """Root endpoint check."""
    return {"status": "Model Service Operational", "version": "1.0.0"}

@app.post("/predict")
async def predict_text(payload: TextPayload):
    """
    Takes a list of texts and predicts their classification label.
    """
    texts = payload.texts
    if not texts:
        raise HTTPException(status_code=400, detail="The 'texts' field cannot be empty.")

    print(f"Received {len(texts)} texts for prediction.")

    # 1. Tokenize the inputs
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 2. Inference
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Assuming the last dimension of the output logits corresponds to the number of classes
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get the predicted class index and corresponding probability
    predicted_indices = torch.argmax(probabilities, dim=-1).cpu().numpy()
    max_probabilities = torch.max(probabilities, dim=-1)[0].cpu().numpy()
    
    # Map indices back to human-readable labels (You must define this mapping)
    # !!! IMPORTANT: Replace this with the actual mapping from your training script !!!
    class_labels = ["Intent_Booking", "Intent_Support", "Intent_Inquiry"] 
    
    predictions = []
    for i in range(len(texts)):
        label = class_labels[predicted_indices[i]]
        confidence = float(max_probabilities[i])
        predictions.append({
            "text": texts[i],
            "predicted_label": label,
            "confidence": f"{confidence:.4f}"
        })

    return {"predictions": predictions}

# --- How to Run ---
# uvicorn api_server:app --reload
