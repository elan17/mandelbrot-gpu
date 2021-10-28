import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from backends.backend import Backend

from pycuda import gpuarray
from pycuda.compiler import SourceModule

import cv2

kernel = ""

with open("kernels/kernel.cu", "r") as f:
    kernel = f.read()

kernel = SourceModule(kernel, no_extern_c=True)


class CudaBackend(Backend):

    def __init__(self, pos, zoom, shape, iterations: int):
        self.pos = pos
        self.zoom_attr = zoom
        self.shape = shape
        self.iterations = iterations

    def update(self):
        cv2.imshow("", get_render(self.pos, self.zoom_attr, self.shape, self.iterations))
        return cv2.waitKey(1)
    
    def move(self, d):
        self.pos += d * self.zoom_attr
    
    def zoom(self, ratio):
        self.zoom_attr *= ratio


def get_render(pos, zoom, shape, iterations: int, block=(32, 32, 1)):
    f = kernel.get_function("compute")
    iterations_arr = np.empty(shape, dtype=np.uint8)
    f(cuda.InOut(iterations_arr), np.array(pos), zoom,
      np.array(iterations, dtype="uint8"), 
      np.array(shape[1], dtype="uint8"),
      block=block, grid=(shape[1]//block[0], shape[0]//block[1])
    )
    return iterations_arr
