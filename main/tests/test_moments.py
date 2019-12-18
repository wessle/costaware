import main.utils.moments as moments
import numpy as np

from math import isclose
from scipy.special import erf, erfinv
from scipy.stats import chi2



class TestWelfordAlgorithm: 

    def test_constant_sequence(self):
        estimator = moments.welford_estimator()
        sequence = lambda: 1.
        for _ in range(100):
            value = sequence()
            mean, var = estimator(value)
            assert mean == value
            assert var == 0.

    def __test_distribution_sequence(self, sequence, mean, var, n=10000):
        """
        Generic testing method. Given a sequence generator with a theoretical
        mean and variance, and given a fixed number of iterations, estimate the
        sample mean and variance using the welford estimator. Then, it asserts
        that the estimated mean and variance are within a >99.99% confidence
        interval of the theoretical quantities.

        In other words, a correctly-implemented Welford estimator will fail this
        test <0.01% of the time. But given the constraints on this project, I
        think it's acceptable to let this fail on occasion if the tradeoff is
        that we can actually write unit tests.

        For the mean, the relevant confidence interval is 

            theoretical mean +/- 4 * sqrt(theoretical variance / n)

        For the variance, the relevant confidence interval is



        """
        alpha = 1-1e-2
        chi_low, chi_hi = chi2.cdf(n, alpha/2), chi2.cdf(n, 1 - alpha/2)
        estimator = moments.welford_estimator()
        for _ in range(n-1):
            _, _ = estimator(sequence())
        est_mean, est_var = estimator(sequence())

        assert np.abs(est_mean - mean) <= 3 * np.sqrt(var / n)
        assert (n-1) * est_var / chi_low <= var \
            and var <= (n-1) * est_var / chi_hi

    def test_constant_sequence1(self):
        self.__test_distribution_sequence(
            lambda: 1.,
            1.,
            0
        )
        
    def test_uniform_sequence(self):
        self.__test_distribution_sequence(
            np.random.rand,
            0.5,
            1/12
        )

    def test_normal_sequence(self):
        self.__test_distribution_sequence(
            np.random.randn,
            0.,
            1.
        )

    def test_exponential_sequence(self):
        self.__test_distribution_sequence(
            np.random.exponential,
            1.,
            1.
        )
        

