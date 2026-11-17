"""Loads the fine-tuned ViT artifacts once and exposes predict(image_bytes)."""
import io
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification, ViTImageProcessor

CHECKPOINT = "google/vit-base-patch16-224"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "vit_brain_tumor.pt"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
PROCESSOR_DIR = ARTIFACTS_DIR / "image_processor"

for required in (WEIGHTS_PATH, CLASS_NAMES_PATH, PROCESSOR_DIR):
    if not required.exists():
        raise FileNotFoundError(
            f"Missing artifact: {required}. "
            "Download the artifacts from Kaggle and place them in artifacts/ (see README)."
        )

# Class labels in the exact index order used during training
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

# Image processor saved during training (resize 224x224 + normalization)
image_processor = ViTImageProcessor.from_pretrained(str(PROCESSOR_DIR))

# Rebuild the architecture with a 4-class head, then load the fine-tuned weights on CPU
model = ViTForImageClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True,
    id2label={i: name for i, name in enumerate(CLASS_NAMES)},
    label2id={name: i for i, name in enumerate(CLASS_NAMES)},
)
model.load_state_dict(
    torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
)
model.eval()


def predict(image_bytes: bytes) -> dict:
    """Return {'label': str, 'confidence': float in [0, 1]} for one image."""
    if not image_bytes:
        raise ValueError("Empty file upload")

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded:
            uploaded.load()                 # force full decode -> catches truncated files
            image = uploaded.convert("RGB") # MRI scans are often grayscale
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid image upload") from exc

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
    }
