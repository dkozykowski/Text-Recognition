from PIL import Image
import numpy as np


class ImagePreprocessing:
    def __init__(self, img_original):
        self.__img_original = img_original
        self.__image_matrix = [[self.img_original.getpixel((y, x)) for x in range(29)] for y in range(29)]

    def __check(self, i, o):
        """ check the color of the given pixel """
        if i not in range(0, 28) or o not in range(0, 28):
            return 0
        if self.img_original.getpixel((i, o)) == 0:
            return 1
        else:
            return 2

    def thin(self, i, o):
        """ create a thinner copy of a given character """
        new_image = Image.new('P', (28, 28))
        for i in range(28):
            for o in range(28):
                if self.matrix[i][o] != 0 and (
                        self.__check(i - 1, o) == 1 or self.__check(i - 1, o - 1) == 1 or self.__check(i, o) == 1 or
                        self.__check(i, o - 1)) == 1:
                    new_image.putpixel((i, o), 0)
        return Image.fromarray(np.array(new_image, dtype=np.uint8))

    # create a bold copy of a given character
    def bold(self, i, o):
        new_image = Image.new('P', (28, 28))
        for i in range(28):
            for o in range(28):
                if self.matrix[i][o] == 0 and (
                        self.__check(i - 1, o) == 2 or self.__check(i - 1, o - 1) == 2 or self.__check(i, o) == 2 or
                        self.__check(i, o - 1)) == 2:
                    new_image.putpixel((i, o), 255)
        return Image.fromarray(np.array(new_image, dtype=np.uint8))

