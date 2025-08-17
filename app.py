from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/dog_cat_model.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # change size to your CNN input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_image = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            uploaded_image = filepath

            img_array = preprocess_image(filepath)
            result = model.predict(img_array)

            result = model.predict(img_array)[0][0]  # probability between 0 and 1
            if result > 0.5:
                prediction = "Dog"
            else:
                prediction = "Cat"



    return render_template("index.html", prediction=prediction, uploaded_image=uploaded_image)

if __name__ == "__main__":
    app.run(debug=True)
