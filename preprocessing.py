from math import floor

import numpy
from PIL import Image


def _convert_and_crop_grayscale(image: Image) -> numpy.array:
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
                break

    for ind in range(grayscale_array.shape[0] - 1, 0, -1):
        if numpy.any(grayscale_array[ind] < TRESHHOLD):
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


def _scale_down_matrix(raw: numpy.array, width: int, height: int) -> numpy.array:
    per_width = raw.shape[1] / width
    per_height = raw.shape[0] / height

    result = numpy.zeros((height, width))

    for i in range(height):
        for j in range(width):
            start_x = int(per_width * j)
            end_x = int(per_width * (j + 1))
            start_y = int(per_height * i)
            end_y = int(per_height * (i + 1))

            result[i][j] = numpy.mean(raw[start_y:end_y, start_x:end_x])

    return result


class Preprocessing:

    def __init__(self, batch: list, size=(40, 40), height=200):
        self.__data = batch
        self.__img_size = size
        self.__image_pre_height = height

    @classmethod
    def single(cls):
        return cls([])

    def __prepare_image(self, image) -> numpy.array:
        baseheight = self.__image_pre_height
        img_rows, img_cols = self.__img_size
        hpercent = (baseheight / float(image.size[1]))
        wsize = int((float(image.size[0]) * float(hpercent)))
        image = image.resize((wsize, baseheight), Image.ANTIALIAS)
        result = _convert_and_crop_grayscale(image)
        result = _scale_down_matrix(result, img_cols, img_rows)

        return result

    def prepare_batch(self) -> numpy.array:
        result = []

        for i, image in enumerate(self.__data):
            print(i)
            result.append(self.__prepare_image(image))

        return numpy.array(result)

    def prepare_single(self, image):
        return self.__prepare_image(image)


