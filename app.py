from flask import Flask, request, render_template, flash
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # нужно для flash-сообщений

model = tf.keras.models.load_model("my_model.keras")
IMG_HEIGHT, IMG_WIDTH = 28, 28

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
    return image_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Файл не был загружен.")
            return render_template("index.html", prediction=None)

        try:
            image = Image.open(file.stream)
            processed = preprocess_image(image)
            pred = model.predict(processed)
            prediction = np.argmax(pred)
        except UnidentifiedImageError:
            flash("Загруженный файл не является изображением.")
        except Exception as e:
            flash(f"Произошла ошибка: {str(e)}")

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
