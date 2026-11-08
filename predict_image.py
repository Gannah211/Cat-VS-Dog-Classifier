import sys, json
import numpy as np
import  tensorflow as tf
from tensorflow.keras import utils

model = tf.keras.models.load_model("cat_dog.keras")

def predict_image(path:str) -> str:
    img = utils.load_img(path, target_size=(160, 160))
    arr =np.expand_dims(utils.img_to_array(img), axis=0)
    p_dog = float(model.predict(arr, verbose=0)[0][0])
    label = "dog" if p_dog > 0.5 else "cat"
    confidence = p_dog if label == "dog" else 1 - p_dog
    return json.dumps({"prediction": label, "confidence": round(confidence, 3)}, indent= 2)

if __name__ == "__main__":
    print(predict_image(sys.argv[1]))