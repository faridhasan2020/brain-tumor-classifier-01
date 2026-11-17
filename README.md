# Brain Tumor MRI Classifier

## Overview
This project fine-tunes a Vision Transformer (ViT) to classify brain MRI scans into four categories (glioma, meningioma, pituitary tumor, and no tumor) and serves predictions through a FastAPI backend that runs locally on CPU. Training was done on Kaggle with a T4 GPU. The trained weights, class labels, and image processor are loaded once at server startup.

> ⚠️ **Educational project only. This model must not be used for medical diagnosis.**

## Model
- **Architecture:** `google/vit-base-patch16-224` (fine-tuned, 4-class head)
- **Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** glioma, meningioma, notumor, pituitary
- **Training setup:** 6 epochs, AdamW (lr = 2e-5), linear warmup (10% of steps), batch size 32
- **Data split:** stratified 80/20 split of `Training/` → 4,480 train / 1,120 validation; `Testing/` (1,600 images) used **once** for final evaluation
- **Selected checkpoint:** best validation accuracy, epoch **[X]** (val acc **[0.XXXX]**)
- **Test Accuracy:** **[0.XXXX]** (held-out `Testing/` set, evaluated after model selection)

### Classes
| Class | Description |
|-------|-------------|
| glioma | Image labeled as glioma in the dataset |
| meningioma | Tumor arising from the meninges, usually benign |
| notumor | Image labeled as containing no tumor |
| pituitary | Tumor of the pituitary gland |

### Limitations
- The dataset's `Training/` and `Testing/` folders contain some identical images and possibly other closely related scans. The reported test accuracy may therefore **overestimate** performance on genuinely unseen patients.
- The data is not split by patient, and it comes from a single public compilation, so the model has not been validated on other scanners, hospitals, or populations.
- The model always outputs one of the four classes, even for images that are not brain MRIs.
- **This is not a medical device and must not be used for diagnosis.**

## Project Structure
```
brain-tumor-classifier/
├── artifacts/
│   ├── vit_brain_tumor.pt        # fine-tuned weights (Git LFS)
│   ├── class_names.json
│   └── image_processor/
├── training/
│   └── vit_finetuning.ipynb
├── screenshots/
├── main.py
├── predictor.py
├── requirements.txt
└── README.md
```

## API Endpoints
| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| GET | `/health` | — | Server status and model info |
| POST | `/predict` | `multipart/form-data` with field `file` (an MRI image: JPG/PNG) | Predicted class and confidence |

### `GET /health` response
```json
{
  "status": "ok",
  "model": "ViT MRI classifier (educational)",
  "base_checkpoint": "google/vit-base-patch16-224",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"],
  "device": "cpu"
}
```

### `POST /predict` response
```json
{
  "label": "glioma",
  "confidence": 0.9821
}
```
`confidence` is the softmax probability of the predicted class (between 0 and 1).
An invalid or corrupt upload returns **400** with `{"detail": "Invalid image upload"}`.

### Example with curl
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@sample_mri.jpg"
```

## Screenshots
### Swagger UI
![Swagger](screenshots/swagger.png)

### Successful prediction
![Prediction](screenshots/prediction.png)

## Installation
```bash
git lfs install
git clone [your-repo-url]
cd brain-tumor-classifier
pip install -r requirements.txt
```

The model weights (`artifacts/vit_brain_tumor.pt`, about 340 MB) are stored with **Git LFS**. If they did not download, run `git lfs pull`.
<!-- If you host the weights elsewhere instead, replace with: Download vit_brain_tumor.pt from [LINK] and place it in artifacts/ -->

> The first run needs internet access once: `predictor.py` fetches the base ViT architecture from Hugging Face, and then loads the fine-tuned weights on top.

## Run
```bash
fastapi dev main.py
# Open http://localhost:8000/docs
```

## Reproduce Training
1. Open `training/vit_finetuning.ipynb` on Kaggle.
2. Enable **Internet** and a **T4 GPU** in the notebook settings.
3. Add the dataset `masoudnickparvar/brain-tumor-mri-dataset`.
4. Run all cells (about 40+ minutes), then download `artifacts.zip` from the Output panel and unzip it into the repo root.

## Technologies Used
- Python
- PyTorch, Hugging Face Transformers
- scikit-learn, Matplotlib (training and evaluation)
- FastAPI
- Pillow
