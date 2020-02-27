import os
import re

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageFile
import base64
from io import BytesIO

from flask import Flask, request, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.python.framework import ops

from recognizer import Recognizer

UPLOAD_FOLDER = 'uploads'
ImageFile.LOAD_TRUNCATED_IMAGES = True

app: Flask = Flask('Recognizer')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
reco = Recognizer('hw.h5')


@app.route('/api/recognize', methods=['POST'])
def get_image():
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imgData'])

    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.load()
    background = Image.new("RGB", im.size, (255, 255, 255))
    background.paste(im, mask=im.split()[3])

    reco.load_image(background)
    last_digit = reco.predict()

    return last_digit


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)