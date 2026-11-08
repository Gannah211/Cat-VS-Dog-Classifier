import io, os, time
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (160, 160)

app = FastAPI(title="Cat vs Dog")
MODEL_PATH = os.path.join(HERE, "cat_dog.keras")
model = tf.keras.models.load_model(MODEL_PATH)                      # loaded once at startup
print("Loaded model:", MODEL_PATH, "| last saved:", time.ctime(os.path.getmtime(MODEL_PATH)))


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="I couldn't read that image.")

    batch = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)   # (1, 160, 160, 3)
    p_dog = float(model.predict(batch, verbose=0)[0][0])
    label = "dog" if p_dog >= 0.5 else "cat"

    return {
        "prediction": label,
        "confidence": round(p_dog if label == "dog" else 1 - p_dog, 3),
        "probabilities": {"cat": round(1 - p_dog, 3), "dog": round(p_dog, 3)},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)