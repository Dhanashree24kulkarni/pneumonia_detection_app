from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your model
model = load_model("pneumonia_model.h5")

@app.route("/")
def home():
    return render_template("index.html")   # 👈 serves frontend

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
