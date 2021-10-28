import time

from backends.cuda_back import get_render
from backends.cpu import CPUBackend
from backends.cuda_back import CudaBackend
import numpy as np
import cv2


if __name__ == "__main__":
    
    backend = CudaBackend(complex(-2, -2), np.array(0.008), (224, 224), 100)

    while True:
        t = time.time()
        k = backend.update()
        print(1/(time.time() - t))
        if k == ord("q"):
            break
        if k == ord("w"):
            backend.move(complex(0.0, -10.0))
        if k == ord("s"):
            backend.move(complex(0.0, 10.0))
        if k == ord("a"):
            backend.move(complex(-10.0, 0.0))
        if k == ord("d"):
            backend.move(complex(10.0, 0.0))
        if k == ord("+"):
            backend.zoom(0.9)
        if k == ord("-"):
            backend.zoom(1.1)
    