from __future__ import print_function

import os
import random

import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image

from preprocessing import Preprocessing

batch_size = 8
num_classes = 35
epochs = 10

img_rows, img_cols = 32, 32


def shuffle_dataset(data: list, labels: list) -> list:
    tmp_list = []

    for i, row in enumerate(data):
        for col in row:
            tmp_list.append({'img': col, 'label': labels[i]})

    random.shuffle(tmp_list)

    return tmp_list


def split_dataset(data):
    imgs = numpy.array([x['img'] for x in data])
    labels = numpy.array([x['label'] for x in data])
    split_begin = int(len(data) / 4)

    return (imgs[0:split_begin], labels[0:split_begin]), (imgs[split_begin:], labels[split_begin:])


def learn(data, labels):
    randomized = shuffle_dataset(data, labels)

    (x_test, y_test), (x_train, y_train) = split_dataset(randomized)

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
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
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


def main():
    SEARCH_FOLDER = 'Img'

    characters_list = []

    for dir in [x[0] for x in os.walk(SEARCH_FOLDER)]:
        if dir != SEARCH_FOLDER:
            number = int(dir.split(os.path.sep)[1][-3:])

            if number > num_classes:
                continue
            characters_list.append(dir)

    labels = list(range(num_classes))

    pro = Preprocessing([numpy.array(Image.open(x)) for x in characters_list])

    data = pro.prepare_batch()

    print('Completed!')

    learn(data, labels)


def test():
    image = Image.open('Img/Sample002/img002-002.png')

    pro = Preprocessing.single()
    result = pro.prepare_single(numpy.array(image))

    model = keras.models.load_model('hw.h5')
    prediction = model.predict(numpy.array([[result]]).reshape((1, 32, 32, 1)))
    print(prediction)


if __name__ == '__main__':
    main()
