"""FastAPI service for the Brain Tumor MRI classifier (educational use only)."""
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import predictor  # loads the model once at startup

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description=(
        "Fine-tuned ViT (google/vit-base-patch16-224) that classifies brain MRI "
        "scans into glioma, meningioma, notumor, or pituitary. "
        "Educational demo only - NOT for medical diagnosis."
    ),
    version="1.0.0",
)


class PredictionOutput(BaseModel):
    label: str = Field(..., examples=["glioma"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9821])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "ViT MRI classifier (educational)",
        "base_checkpoint": predictor.CHECKPOINT,
        "classes": predictor.CLASS_NAMES,
        "device": "cpu",
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = predictor.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionOutput(**result)
