import numpy as np
import time

import tqdm
import cv2

from backends.backend import Backend

class CPUBackend(Backend):

    def __init__(self, initial_array):
        self.array = initial_array

    def update(self):
        cv2.imshow("", get_render(self.array, 100))
        cv2.waitKey(1)
    
    def move(self, d):
        self.array += d
    
    def zoom(self, ratio):
        # TODO
        pass

def get_render(array: np.ndarray, iterations: int):
    dtype = array.dtype
    z = np.array(complex(0, 0))
    iterations_arr = np.zeros_like(array, dtype=np.uint8)
    new_array = array
    for x in tqdm.tqdm(range(iterations)):
        new_array = new_array ** 2 + array
        bounded = ((new_array.imag**2 + new_array.real**2) < 2.0)
        iterations_arr += bounded
    return iterations_arr
