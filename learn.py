from __future__ import print_function

import os
import random
from os import listdir
from os.path import isfile, join

import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from PIL import Image

from preprocessing import Preprocessing

batch_size = 8
num_classes = 62
epochs = 8

DEST_DIR = 'After'

img_rows, img_cols = 40, 40


def shuffle_dataset(data: list, labels: list) -> list:
    tmp_list = []

    print(len(data))

    for i, row in enumerate(data):
        for col in row:
            tmp_list.append({'img': col, 'label': labels[i]})

    random.shuffle(tmp_list)

    return tmp_list


def split_dataset(data):
    imgs = numpy.array([x['img'] for x in data])
    labels = numpy.array([x['label'] for x in data])
    split_begin = int(len(data) / 10)

    return (imgs[0:split_begin], labels[0:split_begin]), (imgs[split_begin:], labels[split_begin:])


def learn():
    labels = list(range(num_classes))

    characters_list = []

    data = []

    for dir in [x[0] for x in os.walk(DEST_DIR)]:
        if dir != DEST_DIR:
            number = int(dir.split(os.path.sep)[1][-3:])

            if number > num_classes - 1:
                continue
            characters_list.append(dir)

    for path in characters_list:

        tmp = []

        for t in [f for f in listdir(path) if isfile(join(path, f))]:
            img_array = numpy.array(Image.open(join(path, t)))

            result_array = numpy.zeros((img_rows, img_cols))

            for i, row in enumerate(img_array):
                for j, pixel in enumerate(row):
                    result_array[i][j] = numpy.mean(pixel)

            tmp.append(result_array)

        data.append(tmp)

    randomized = shuffle_dataset(data, labels)

    (x_test, y_test), ( x_train, y_train) = split_dataset(randomized)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(40, 40, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('sigmoid'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('hw.h5')


def prepare():
    SEARCH_FOLDER = 'Img'

    characters_list = []

    for dir in [x[0] for x in os.walk(SEARCH_FOLDER)]:
        if dir != SEARCH_FOLDER:
            number = int(dir.split(os.path.sep)[1][-3:])

            if number > num_classes:
                continue
            characters_list.append(dir)

    for j, path in enumerate(characters_list):
        print(path)
        os.mkdir(join(DEST_DIR, str(j)))
        pro = Preprocessing([Image.open(join(path, x)) for x in [f for f in listdir(path) if isfile(join(path, f))]])

        tmp = pro.prepare_batch()

        for i, t in enumerate(tmp):
            img = Image.fromarray(numpy.uint8(t), 'L')
            img.save(join(DEST_DIR, str(j), f'{i}.jpg'))

    print('Completed!')


def test():
    characters_list = []

    data = []

    for dir in [x[0] for x in os.walk(DEST_DIR)]:
        if dir != DEST_DIR:
            number = int(dir.split(os.path.sep)[1][-3:])

            if number > num_classes - 1:
                continue
            characters_list.append(dir)

    helper = {ind: x.split('/')[1] for ind, x in enumerate(characters_list)}

    for path in characters_list:
        for t in [f for f in listdir(path) if isfile(join(path, f))]:
            img_array = numpy.array(Image.open(join(path, t)))

            result_array = numpy.zeros((img_rows, img_cols))

            for i, row in enumerate(img_array):
                for j, pixel in enumerate(row):
                    result_array[i][j] = numpy.mean(pixel)

            data.append(result_array)
            break

    model = keras.models.load_model('hw.h5')

    result = dict()

    for i, d in enumerate(data):
        prediction = model.predict(numpy.array([[d]]).reshape((1, 40, 40, 1)))
        result[helper[i]] = numpy.argmax(prediction)

    print(result)


if __name__ == '__main__':
    test()
