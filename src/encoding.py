import numpy as np
import pandas as pd

class TargetOHE:
    """
    One Hot Encoding per la variabile target (y)
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise Exception("Encoder not fitted")

        return self._one_hot(y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def _one_hot(self, y):
        one_hot = np.zeros((len(y), len(self.classes_)))

        for i, val in enumerate(y):
            idx = np.where(self.classes_ == val)[0][0]
            one_hot[i, idx] = 1

        return one_hot

    def inverse_transform(self, y_encoded):
        return self.classes_[np.argmax(y_encoded, axis=1)]