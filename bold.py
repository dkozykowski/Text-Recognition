from PIL import Image
import numpy as np


def __check(i, o):
    if i not in range(0, 28) or o not in range(0, 28):
        return False
    return imgOriginal.getpixel((i, o)) != 0


def bold(path_input, path_output):
    img_original = Image.open(path_input)

    img_new = Image.new('P', (28, 28))
    matrix = [[0 for x in range(29)] for y in range(29)]

    for i in range(28):
        for o in range(28):
            matrix[i][o] = img_original.getpixel((i, o))

    for i in range(28):
        for o in range(28):
            if matrix[i][o] == 0 and (__check(i - 1, o) or __check(i - 1, o - 1) or __check(i, o) or __check(i, o - 1)):
                matrix[i][o] = 255
                img_new.putpixel((i, o), matrix[i][o])

    picture = np.array(img_new, dtype=np.uint8)
    img = Image.fromarray(picture)
    img.save(path_output)
