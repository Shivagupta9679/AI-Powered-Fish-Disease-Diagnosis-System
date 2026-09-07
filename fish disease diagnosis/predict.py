import numpy as np
import tensorflow as tf

MODEL_PATH = "VGG16_fish_model_fixed.h5"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("MODEL LOADED SUCCESSFULLY!")
def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        predicted_class = "Infected"
        confidence = probability
    else:
        predicted_class = "Healthy"
        confidence = 1 - probability

    return predicted_class, confidence