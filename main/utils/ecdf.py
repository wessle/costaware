import numpy as np
from typing import Callable

class ECDFEstimator:

    def __init__(self):
        self.__lower  = None
        self.__upper  = None
        self.__k      = 0
        self.__values = []

    def __len__(self):
        return self.__k

    @property
    def lower(self):
        return self.__lower

    @property
    def upper(self):
        return self.__upper

    @property
    def values(self):
        return np.array(self.__values)

    def __call__(self, next_value):
        self.__k += 1
        self.__values.append(next_value)

        vectorized_values = np.array(self.__values)
        self.__lower = np.min(vectorized_values)
        self.__upper = np.max(vectorized_values)

        @np.vectorize
        def ecdf(x: float) -> float:
            """
            This is the actual empirical cdf. Given an input, it computes the
            fraction of total observations which are no greater than the input.
            """
            nonlocal vectorized_values

            return np.sum(vectorized_values <= x) / self.__k

        return ecdf

