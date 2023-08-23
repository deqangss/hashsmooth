from abc import ABC, abstractmethod
import numpy as np
import warnings


class BasicClassifier(ABC):
    """
    classification model template
    """

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def certify(self, x: np.ndarray, label: (int, np.ndarray), ):
        raise NotImplementedError
