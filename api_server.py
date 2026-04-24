from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib

app = FastAPI()
svm_model = joblib.load("message_classifier.pkl")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # load once

class Input(BaseModel):
    text: list[str]  # expect list of texts

def predict_texts(texts: list[str]) -> dict:
    # same preprocessing as training
    vectors = embedder.encode(texts, show_progress_bar=False)
    preds = svm_model.predict(vectors)
    probs = svm_model.predict_proba(vectors)
    return {
        "predictions": preds.tolist(),
        "probabilities": probs.tolist()
    }

@app.post("/predict")
async def predict_endpoint(payload: Input):
    try:
        return {"ok": True, "result": predict_texts(payload.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
