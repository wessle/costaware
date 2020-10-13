import numpy as np


def welford_estimator():
    """
    For a sequence of identically distributed random variates

    X1, X2, X3, X4, ...

    this algorithm computes unbiased estimates of the mean and variance of the
    underlying random variable X which has the same distribution as X1, X2, ...

    This implementation has two important features:
    
    1. It is online. The estimates are updated as each new data point becomes
       available.
    2. It is numerically stable. See
       https://www.johndcook.com/blog/standard_deviation/

    This function returns a closure which can be used to re-estimate the value
    of the mean and variance for each new piece of input.
    """
    k = 0
    mean = 0.
    unnorm_var = 0.

    def estimate(next_input):
        """
        Closure which re-estimates the value of the mean and variance of an
        underlying distribution by sampling an additional point

        Parameters
        ----------
        next_input: float
            the next element of the sequence of identically distributed random
            variables

        Returns
        -------
        mean, var: (float, float)
            new estimates of the mean and variance of the underlying
            distribution
        """
        nonlocal k, mean, unnorm_var
        k += 1
        old_mean = mean
        mean += next_input / k - old_mean / k
        unnorm_var += (next_input - old_mean) * (next_input - mean)
        var = 0. if k == 1 else unnorm_var / (k - 1)

        return mean, var

    return estimate


class WelfordEstimator:
    """Object-oriented version of the above function."""

    def __init__(self):
        self.k = 0
        self.mean = 0.0
        self.unnorm_var = 0.0

    def __call__(self, next_input):
        self.k += 1
        old_mean = self.mean
        self.mean += (next_input - old_mean) / self.k
        self.unnorm_var += \
            (next_input - old_mean) * (next_input - self.mean)
        var = 0 if self.k == 1 else self.unnorm_var / (self.k - 1)

        return self.mean, var


