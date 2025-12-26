from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import io

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Barnacle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load ONNX models
# ----------------------------
unet_sess = ort.InferenceSession(
    "model/unet.onnx",
    providers=["CPUExecutionProvider"]
)

deeplab_sess = ort.InferenceSession(
    "model/deeplab.onnx",
    providers=["CPUExecutionProvider"]
)

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = 512

# ----------------------------
# Helper functions
# ----------------------------
def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return img[np.newaxis, :]           # 1x3x512x512

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def severity_from_coverage(cov):
    if cov <= 20:
        return "LOW"
    elif cov <= 50:
        return "MEDIUM"
    else:
        return "HIGH"

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(image)

    # Preprocess
    inp = preprocess(img)

    # Inference
    u_logits = unet_sess.run(None, {"input": inp})[0]
    d_logits = deeplab_sess.run(None, {"input": inp})[0]

    # Convert logits → probabilities
    u_prob = sigmoid(u_logits)
    d_prob = sigmoid(d_logits)

    # Ensemble
    prob = 0.75 * u_prob + 0.25 * d_prob

    # 🔍 Debug (keep while testing)
    print("Ensemble prob min/max:", prob.min(), prob.max())

    # ✅ Adaptive threshold (IMPORTANT)
    thr = prob.mean() + 0.5 * prob.std()
    mask = (prob > thr).astype(np.uint8)[0, 0]

    # Coverage calculation
    coverage = float(np.count_nonzero(mask) / mask.size * 100)

    # Confidence (strongest prediction)
    confidence = float(prob.max() * 100)

    severity = severity_from_coverage(coverage)

    return {
        "coverage": round(coverage, 2),
        "severity": severity,
        "confidence": round(confidence, 2),
        "organisms": ["Barnacles"]
    }

# ----------------------------
# Root check
# ----------------------------
@app.get("/")
def root():
    return {"status": "API running"}
