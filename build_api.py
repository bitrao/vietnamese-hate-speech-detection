
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.transformer_classifier import HateSpeechDetector
import asyncio

class HODRequest(BaseModel):
    texts: str | list[str]

    
app = FastAPI(
    title = "VN Hate Speech Detector API",
    description="API for VN Hate Speech using Fast API",
    version="1.0"
)

model_path ='models/phobert-binary'
tokenizer = 'vinai/phobert-base'
model = HateSpeechDetector.load_model(model_dir=model_path, tokenizer=tokenizer)


@app.get("/") 
async def root():
    return {"message": "Welcome to the VN Hate Speech Detector Model API"}


@app.post("/predict", response_model= int | list[int])
async def predict(request: HODRequest):
    try:
        prediction = await asyncio.get_event_loop().run_in_executor(
            None, model.predict, request.texts
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}