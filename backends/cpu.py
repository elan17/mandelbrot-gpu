import numpy as np
import tqdm
import time

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
