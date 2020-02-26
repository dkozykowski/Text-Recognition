from __future__ import print_function

import os
import random
import threading
from math import floor

import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image


tmp_global = []
batch_size = 8
num_classes = 35
epochs = 10

img_rows, img_cols = 32, 32

characters_dict = {
    1: '0',
    2: '1',
    3: '2',
    4: '3',
    5: '4',
    6: '5',
    7: '6',
    8: '7',
    9: '8',
    10: '9',
    11: 'a',
    12: 'b',
    13: 'c',
    14: 'd',
    15: 'e',
    16: 'f',
    17: 'g',
    18: 'h',
    19: 'i',
    20: 'j',
    21: 'k',
    22: 'l',
    23: 'm',
    24: 'n',
    25: 'o',
    26: 'p',
    27: 'r',
    28: 's',
    29: 't',
    30: 'u',
    31: 'w',
    32: 'v',
    33: 'x',
    34: 'y',
    35: 'z'
}


def convert_and_crop_grayscale(image: Image) -> numpy.array:
    tmp = numpy.array(image)
    width, height = image.size

    # find "bounding box"
    TRESHHOLD = 100

    top_limit = 0
    bottom_limit = height
    left_limit = 0
    right_limit = width

    grayscale_array = numpy.array([[pixel[0] / 3 + pixel[1] / 3 + pixel[2] / 3 for pixel in row] for row in tmp])

    for ind, row in enumerate(grayscale_array):
        if top_limit == 0:
            if numpy.any(row < TRESHHOLD):
                top_limit = ind
        else:
            if numpy.all(row > TRESHHOLD):
                bottom_limit = ind
                break

    for ind, row in enumerate(numpy.transpose(grayscale_array)):
        if left_limit == 0:
            if numpy.any(row < TRESHHOLD):
                left_limit = ind
        else:
            if numpy.all(row > TRESHHOLD):
                right_limit = ind
                break

    fixed_width = right_limit - left_limit
    fixed_height = bottom_limit - top_limit

    if fixed_height > fixed_width:
        diff = fixed_height - fixed_width
        to_be_added_right = int(floor(diff / 2))
        to_be_added_left = int(floor((diff + 1) / 2))

        to_few_pixels_left = min(0, left_limit - to_be_added_left)
        to_few_pixels_right = min(0, width - right_limit - to_be_added_right)

        to_be_added_left += to_few_pixels_left
        to_be_added_right += to_few_pixels_right

        left_limit -= to_be_added_left
        right_limit += to_be_added_right
    elif fixed_height < fixed_width:
        diff = fixed_width - fixed_height
        to_be_added_top = int(floor(diff / 2))
        to_be_added_bottom = int(floor((diff + 1) / 2))

        to_few_pixels_top = min(0, top_limit - to_be_added_top)
        to_few_pixels_bottom = min(0, height - bottom_limit - to_be_added_bottom)

        to_be_added_top += to_few_pixels_top
        to_be_added_bottom += to_few_pixels_bottom

        top_limit -= to_be_added_top
        bottom_limit += to_be_added_bottom

    return grayscale_array[top_limit:bottom_limit, left_limit:right_limit]


def scale_down_matrix(raw: numpy.array, width:int, height: int) -> numpy.array:
    per_width = raw.shape[1] / width
    per_height = raw.shape[0] / height

    result = numpy.zeros((height, width))

    for i in range(height):
        for j in range(width):
            start_x = int(per_width * j)
            end_x = int(per_width * (j + 1))
            start_y = int(per_height * i)
            end_y = int(per_height * (i + 1))

            result[i][j] = numpy.array([numpy.mean(raw[start_y:end_y, start_x:end_x])])

    return result


def transform_images(*paths):
    global tmp_global
    character_data = []
    for path in paths:
        print(path)
        image = Image.open(path)
        baseheight = 200
        hpercent = (baseheight / float(image.size[1]))
        wsize = int((float(image.size[0]) * float(hpercent)))
        image = image.resize((wsize, baseheight), Image.ANTIALIAS)
        result = convert_and_crop_grayscale(image)
        result = scale_down_matrix(result, img_cols, img_rows)
        character_data.append(result)

    tmp_global += character_data


def get_character_data(path):

    que = []
    global tmp_global
    tmp_global = []

    for file in os.listdir(path):
        if file != path:
            que.append(os.path.join(path, file))

    divider = int(len(que) / 8)

    th_1 = threading.Thread(target=transform_images, args=(que[0:divider]))
    th_2 = threading.Thread(target=transform_images, args=(que[divider:divider*2]))
    th_3 = threading.Thread(target=transform_images, args=(que[divider*2:divider*3]))
    th_4 = threading.Thread(target=transform_images, args=(que[divider*3:divider*4]))
    th_5 = threading.Thread(target=transform_images, args=(que[divider*4:divider*5]))
    th_6 = threading.Thread(target=transform_images, args=(que[divider*5:divider * 6]))
    th_7 = threading.Thread(target=transform_images, args=(que[divider * 6:divider * 7]))
    th_8 = threading.Thread(target=transform_images, args=(que[divider * 7:]))

    th_1.start()
    th_2.start()
    th_3.start()
    th_4.start()
    th_5.start()
    th_6.start()
    th_7.start()
    th_8.start()

    th_1.join()
    th_4.join()
    th_2.join()
    th_3.join()
    th_8.join()
    th_5.join()
    th_6.join()
    th_7.join()

    return tmp_global


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
            characters_list.append({
                'character': characters_dict[number],
                'path': dir
            })

    data = []
    labels = [x - 1 for x in characters_dict.keys()]

    for character in characters_list:
        data.append(get_character_data(character['path']))

    print('Completed!')

    learn(data, labels)


def test():
    image = Image.open('Img/Sample002/img002-001.png')
    baseheight = 200
    hpercent = (baseheight / float(image.size[1]))
    wsize = int((float(image.size[0]) * float(hpercent)))
    image = image.resize((wsize, baseheight), Image.ANTIALIAS)
    result = convert_and_crop_grayscale(image)
    result = scale_down_matrix(result, img_cols, img_rows)

    model = keras.models.load_model('hw.h5')
    prediction = model.predict(numpy.array([[result]]).reshape((1, 32, 32, 1)))
    print(prediction)


if __name__ == '__main__':
    main()
