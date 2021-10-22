import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from pycuda import gpuarray
from pycuda.compiler import SourceModule

kernel = ""

with open("kernels/kernel.cu", "r") as f:
    kernel = f.read()

kernel = SourceModule(kernel, no_extern_c=True)

def get_render(array: np.ndarray, iterations: int):
    f = kernel.get_function("compute")
    array = array.astype(np.complex128)
    iterations_arr = np.zeros_like(array, dtype=np.uint8)
    f(cuda.In(np.zeros_like(array)), cuda.In(array),
      cuda.InOut(iterations_arr),
      np.array(array.shape[1], dtype=np.uint16),
      block=(10, 10, 1), grid=(100, 100))
    return iterations_arr