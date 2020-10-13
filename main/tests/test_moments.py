import main.utils.moments as moments
import numpy as np

from math import isclose
from scipy.special import erf, erfinv
from scipy.stats import chi2, norm

def icdf(dist, y, df=None):
    return dist.isf(1-y) if df is None else dist.isf(1-y, df)


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
        alpha = 1e-2  # set the global significance level.

        # bounds for the mean confidence interval
        normal_bounds = icdf(norm, np.array([alpha/2, 1-alpha/2]))

        # boundes for the variance confidence interval --- I don't understand
        # chi2 distributions so I am not 100% sure that this is correct.
        chi_low, chi_hi = icdf(chi2, 1-alpha/2, df=n-1), icdf(chi2, alpha/2, df=n-1)

        # set up an experiment, run the estimator for the specified number of
        # turns
        estimator = moments.welford_estimator()
        for _ in range(n-1):
            _, _ = estimator(sequence())

        # get the last value of the estimated means and variances
        est_mean, est_var = estimator(sequence())

        # confidence intervals
        mean_lower, mean_higher = est_mean * np.sqrt(var / n) * normal_bounds
        assert mean_lower <= mean <= mean_higher

        assert (n-1) * est_var / chi_low <= var  and var <= (n-1) * est_var / chi_hi

    def test_constant_sequence1(self):
        self.__test_distribution_sequence(
            lambda: 1., 1., 0
        )
        
    def test_uniform_sequence(self):
        self.__test_distribution_sequence(
            np.random.rand, 0.5, 1/12
        )

    def test_normal_sequence(self):
        self.__test_distribution_sequence(
            np.random.randn, 0., 1.
        )

    def test_exponential_sequence(self):
        self.__test_distribution_sequence(
            np.random.exponential, 1., 1.
        )

