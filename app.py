import os
from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef
from unet import *

app = Flask(__name__)

# Load the model
model_path = os.path.join("files", "model.h5")

# Define image and result paths
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(model_path)

# Ensure 'uploads' and 'results' directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Define the path to the results directory
results_dir = RESULTS_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"
    
    image = request.files['image']
    if image.filename == '':
        return "No selected file"
    
    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)
    
    # Perform segmentation
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256)) / 255.0
    result = model.predict(np.expand_dims(img, axis=0))[0]
    
    # Save the segmentation result
    result_filename = f"result_{image.filename}"
    result_path = os.path.join(results_dir, result_filename)
    cv2.imwrite(result_path, (result * 255).astype(np.uint8))
    
    # Define the URL for the result image
    result_url = f'/static/results/{result_filename}'
    
    return render_template('index.html', result=result_url)

if __name__ == '__main__':
    app.run(debug=True)
