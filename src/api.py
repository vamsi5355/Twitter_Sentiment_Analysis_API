import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertForSequenceClassification


MODEL_PATH = os.getenv("MODEL_PATH", "model_output")

app = FastAPI()

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


model.eval()

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(
        req.text, return_tensors="pt", truncation=True, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        confidence, label = torch.max(probs, dim=0)

    sentiment = "positive" if label.item() == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": float(confidence)
    }
