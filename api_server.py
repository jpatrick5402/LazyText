from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()
svm_model = joblib.load("message_classifier.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_response(text: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
        },
        json={
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "messages": [
                {
                    "role": "user",
                    "content": "I want you to respond to the following message as if you were a human. Do not ask them any questions and don't tell them you're an AI:\n\n" + text
                }
            ]
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def predict_text(text: str) -> dict:
    vector = embedder.encode([text], show_progress_bar=False)

    pred = svm_model.predict(vector)[0]
    prob = svm_model.predict_proba(vector)[0]

    if (pred == 1):
        response = get_response(text)
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
