from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import os
from google import genai

app = FastAPI()
svm_model = joblib.load("message_classifier.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def predict_text(text: str) -> dict:
    vector = embedder.encode([text], show_progress_bar=False)
    prediction = svm_model.predict(vector)[0]
    probability = svm_model.predict_proba(vector)[0]

    try:
        if (prediction == 1):
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            gemini_response = client.models.generate_content(
                model = "gemini-3-flash-preview",
                contents=f"I want you to respond to \
                    the following message as if you were a human. Do not ask them any questions\
                    and don't tell them you're an AI. Respond in one sentence or one word preferably\
                    : {text}",
                config = genai.types.GenerateContentConfig(
                    http_options = genai.types.HttpOptions(
                        retry_options = genai.types.HttpRetryOptions(
                            attempts=5,
                            initial_delay=1.0,
                            http_status_codes=[408, 429, 500, 502, 503, 504],
                        )
                    )
                ),
            )
            client.close()
            response = gemini_response.text
            candidate = gemini_response.candidates[0]
        else:
            response = ""
            candidate = None
    except:
        response = ""
        candidate = None

    finish_reason = "NOT NEEDED"
    if candidate and candidate.finish_reason:
        finish_reason = candidate.finish_reason.name

    return {
        "description": "prediction: 0 = No response needed, 1 = AI response needed, 2 = Human response needed",
        "prediction": int(prediction) if hasattr(prediction, "item") else prediction,
        "probabilities": probability.tolist(),
        "response": response,
        # "STOP" is a success for the gemini_response below
        "geneartion_success?": finish_reason
    }

class Input(BaseModel):
    text: str

@app.post("/predict")
async def predict_endpoint(payload: Input):
    try:
        return {"ok": True, "result": predict_text(payload.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
