from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import ollama

app = FastAPI()
svm_model = joblib.load("message_classifier.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def predict_text(text: str) -> dict:
    vector = embedder.encode([text], show_progress_bar=False)

    pred = svm_model.predict(vector)[0]
    prob = svm_model.predict_proba(vector)[0]

    if (pred == 1):
        response = ollama.generate(model='gemma4', prompt=f"I want you to respond to \
            the following message as if you were a human. Do not ask them any questions\
            and don't tell them you're an AI. Respond in one sentence or one word preferably\
            : {text}")["response"]

    else:
        response = ""

    return {
        "prediction": int(pred) if hasattr(pred, "item") else pred,
        "probabilities": prob.tolist(),
        "response": response
    }

class Input(BaseModel):
    text: str

@app.post("/predict")
async def predict_endpoint(payload: Input):
    try:
        return {"ok": True, "result": predict_text(payload.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
