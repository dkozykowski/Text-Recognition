import json

import numpy
from tensorflow.keras.models import load_model
from PIL import Image

from preprocessing import Preprocessing


class Recognizer:

    def __init__(self, model_path: str, *, img_size=(32, 32), encoding_path='encoding.json'):
        self.__picture_size = img_size
        self.__model_path = model_path
        self.__data = None
        self.__model = load_model(model_path)
        self.__load_encoding(encoding_path)

    def __load_encoding(self, path):
        with open(path, 'r') as f:
            self.__encoding_dict = json.load(f)

    def load_image_path(self, path: str):
        tmp = Image.open(path)
        pro = Preprocessing.single()
        self.__data = pro.prepare_single(tmp)

    def load_image(self, img: Image):
        pro = Preprocessing.single()
        self.__data = pro.prepare_single(img)

    def predict(self):
        prediction = self.__model.predict(numpy.array([[self.__data]]).reshape((1, 32, 32, 1)))[0]

        index = numpy.argmax(prediction)

        return self.__encoding_dict[str(index)]
