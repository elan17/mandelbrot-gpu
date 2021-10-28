from abc import ABC, abstractmethod

import numpy as np

class Backend:

    @abstractmethod
    def __init__(self, pos: tuple, zoom: float, shape: tuple):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def move(self, d: complex):
        pass

    @abstractmethod
    def zoom(self, ratio: float):
        pass