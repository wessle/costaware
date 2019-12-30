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
        return self.__values

    def __call__(self, next_value):
        self.__k += 1
        self.__values.append(next_value)

        vectorized_values = np.array(values)
        self.__lower = np.min(vectorized_values)
        self.__upper = np.min(vectorized_values)

        @np.vectorize
        def ecdf(x: float) -> float:
            """
            This is the actual empirical cdf. Given an input, it computes the
            fraction of total observations which are no greater than the input.
            """
            nonlocal vectorized_values

            return np.sum(vectorized_values <= x) / k

        return ecdf



def ecdf_estimator() -> Callable[[float], Callable[[float], float]]:
    """
    This function creates an online empirical cumulative distribution function
    which updates as new data is recorded.

    Note: This is a very high-level function. It returns a function which takes
    as input a float and returns a function which itself takes as input a float
    and returns a float. In other words,

    ecdf_estimator : () -> (float -> (float -> float))


    """
    k = 0
    values = []

    def estimator(next_value: float) -> Callable[[float], float]:
        """
        Each input refines the structure of the empirical cdf, which requires a
        new Python function. Therefore, this estimator returns a new function
        for each subsequent input.
        """
        nonlocal values, k

        k += 1
        values.append(next_value)
                return ecdf 

    return estimator



