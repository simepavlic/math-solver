import cv2
import keras
import os
import math_parser
import numpy as np
from symbol_locator import write_contour_images
from math_solver import compute
import re
import sys
import shutil


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def solve_photo(image):
    tmp_dir = 'tmp_images'
    os.mkdir(tmp_dir)
    write_contour_images(image, tmp_dir)

    model = keras.models.load_model("math_model")
    label_array = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']
    expression = ''
    for _, _, files in os.walk(tmp_dir):
        files = natural_sort(files)
        for file in files:
            image = cv2.imread("%s/%s" % (tmp_dir, file))
            image = image.astype('float32')
            image /= 255
            x = np.expand_dims(image, axis=0)
            labels = model.predict(x)
            label = np.argmax(labels)
            char = label_array[label]
            expression += char

    shutil.rmtree(tmp_dir)
    try:
        p = math_parser.parse(expression)
    except:
        return "error parsing photo content"
    return compute(p)


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    print(solve_photo(image))
