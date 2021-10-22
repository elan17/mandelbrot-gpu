from backends.cuda_back import get_render
import numpy as np
import cv2


def initialize_array(left_right: complex, zoom: float, shape: tuple, dtype=np.complex256):
    array = np.empty(shape, dtype=dtype)
    for y, row in enumerate(array):
        for x, column in enumerate(row):
            array[y][x] = left_right + complex(x, y)*zoom
    return array



if __name__ == "__main__":
    array = initialize_array(complex(-2, -2), np.array(0.004), (1000, 1000))
    array = get_render(array, 255)
    cv2.imshow("", array)
    cv2.waitKey()
    