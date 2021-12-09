from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
model = load_model('model3.h5')

def preprocess(img,input_size):
    nimg = cv2.resize(img, (224, 224))
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

def predict_label(img_path):
    input_size = (224,224)
    channel = (3,)
    input_shape = input_size + channel
    labels = ['Covid19', 'Normal', 'Pneumonia']
    im = cv2.imread(img_path)
    x = preprocess(im,input_size)
    x = reshape([x])
    y = model.predict(x)
    return labels[np.argmax(y)], np.max(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)