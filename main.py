import os
import tensorflow as tf
from tensorflow.keras import layers, utils

BASE = "cats_and_dogs_filtered"
train_dir = os.path.join(BASE, "train")
validation_dir = os.path.join(BASE, "validation")

IMG_SIZE = (160, 160)
BATCH = 32
train_ds =utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH, shuffle=True)
validation_ds =utils.image_dataset_from_directory(validation_dir,image_size=IMG_SIZE, batch_size=BATCH)
class_names = train_ds.class_names
print(class_names)

# build a model using Transfer learning
augment =tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

base_model =tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs =tf.keras.Input(shape=IMG_SIZE + (3,))
x =augment(inputs)
x =layers.Rescaling(1/ 127.5, offset = -1)(x)
x =base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer= tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy",metrics=["accuracy"])

#train model
model.fit(train_ds, validation_data=validation_ds, epochs=5)

# Evaluate and save ----------
loss, acc = model.evaluate(validation_ds)
print(f"Validation accuracy: {acc:.3f}")
model.save("cat_dog.keras")
print("Saved cat_dog.keras")