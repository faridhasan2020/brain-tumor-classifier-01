# Final Project 02 – Brain Tumor MRI Classification

**Student:** [Md. Farid Hasan]

## GitHub Repository
[https://github.com/faridhasan007/brain-tumor-classifier]

## Model Used
Vision Transformer `google/vit-base-patch16-224`, fine-tuned on the Brain Tumor MRI Dataset (Kaggle) for 4 classes: glioma, meningioma, notumor, pituitary.
6 epochs, AdamW lr=2e-5, linear warmup, batch size 32, trained on a Kaggle T4 GPU.

## Results
- Best validation accuracy: **[0.9830]** at epoch **[6]** (stratified 80/20 split of Training/, 1,120 validation images)
- **Final held-out test accuracy: [0.9387]** on Testing/ (1,600 images), evaluated once after selecting the best validation checkpoint

Note: Training/ and Testing/ contain some identical images, so the test accuracy may overestimate real-world performance. The model is educational only and not for medical diagnosis.

## API Endpoints Implemented
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Returns server status and model info |
| POST | /predict | Accepts an MRI image (multipart field `file`), returns `{"label", "confidence"}` |

## Screenshot of a Successful Prediction (Swagger UI)
![Prediction](screenshots/prediction.png)
