import os, sys, glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import utils

model = tf.keras.models.load_model("cat_dog.keras")
val_dir = os.path.join("cats_and_dogs_filtered", "validation")


# Three ways of turning an image file into the (160,160,3) array the model expects
def way_load_img(p):        # what predict_image.py does (Keras default resizing)
    return utils.img_to_array(utils.load_img(p, target_size=(160, 160)))

def way_pil(p):             # what app.py does
    img = Image.open(p).convert("RGB").resize((160, 160), Image.BILINEAR)
    return np.asarray(img, dtype="float32")

def way_tf(p):              # what the training loader does
    img = np.asarray(Image.open(p).convert("RGB"), dtype="float32")
    return tf.image.resize(img, (160, 160), method="bilinear").numpy()

WAYS = {"load_img (predict_image.py)": way_load_img,
        "PIL bilinear (app.py)": way_pil,
        "tf bilinear (training)": way_tf}

# ---- Check 1: accuracy measured the same way as during training
val_ds = utils.image_dataset_from_directory(val_dir, image_size=(160, 160), batch_size=32)
print(f"\nCheck 1 - Keras loader accuracy: {model.evaluate(val_ds, verbose=0)[1]:.3f}\n")

# ---- Check 2: accuracy for each preparation method (up to 100 validation images)
paths, labels = [], []
for label, cls in enumerate(["cats", "dogs"]):          # cats = 0, dogs = 1
    files = sorted(glob.glob(os.path.join(val_dir, cls, "*.jpg")))[:50]
    paths += files
    labels += [label] * len(files)
labels = np.array(labels)

print("Check 2 - accuracy per preparation method:")
for name, fn in WAYS.items():
    X = np.stack([fn(p) for p in paths])
    pred = (model.predict(X, verbose=0)[:, 0] >= 0.5).astype(int)
    print(f"  {name:30s} {(pred == labels).mean():.3f}")

# ---- Check 3 (optional): one image, all methods.  Usage: python diagnose.py my_dog.jpg
if len(sys.argv) > 1:
    print(f"\nCheck 3 - P(dog) for {sys.argv[1]}  (above 0.5 = dog):")
    for name, fn in WAYS.items():
        p = float(model.predict(fn(sys.argv[1])[None], verbose=0)[0][0])
        print(f"  {name:30s} {p:.3f}")
