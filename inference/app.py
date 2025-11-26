import os
import json
import io   # 👈 IMPORTANT: needed for BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI(title="Animals Classification API")

MODEL_PATH = "model/model.keras"
CLASS_INDICES_PATH = "model/class_indices.json"
IMAGE_SIZE = (64, 64)


# -----------------------------
# Load class indices
# -----------------------------
def load_class_indices(path=CLASS_INDICES_PATH):
    """Loads class indices in whichever format training produced."""
    if not os.path.exists(path):
        raise RuntimeError(f"class_indices.json not found at {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Case 1 — standard format: {"cat": 0, "dog": 1, ...}
    if isinstance(data, dict) and all(isinstance(v, int) for v in data.values()):
        return data

    # Case 2 — your current format: {"class_names": ["class_0", "class_1", "class_2"]}
    if "class_names" in data and isinstance(data["class_names"], list):
        class_names = data["class_names"]
        return {name: idx for idx, name in enumerate(class_names)}

    raise RuntimeError(f"Unsupported class_indices.json format: {data}")


# -----------------------------
# Load Model
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading class indices...")
    class_indices = load_class_indices()
    index_to_class = {v: k for k, v in class_indices.items()}

    print("Loaded classes:", index_to_class)
    return model, index_to_class


model, index_to_class = load_model()


# -----------------------------
# Helper: preprocess image
# -----------------------------
def preprocess_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# API Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Animals Classification API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) Read & open image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        print("Error reading image:", e)
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    # 2) Preprocess
    img_array = preprocess_image(image)

    # 3) Predict
    try:
        predictions = model.predict(img_array)
    except Exception as e:
        print("Error during prediction:", e)
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    pred_index = int(np.argmax(predictions))
    pred_class = index_to_class[pred_index]
    confidence = float(np.max(predictions))

    return JSONResponse(
        content={
            "predicted_class": pred_class,
            "confidence": confidence,
        }
    )
