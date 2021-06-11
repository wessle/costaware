import numpy as np
from typing import Generator, Tuple
import main.utils.moments as moments


# Core generator methods  


def wiener_process(increment_t: float) -> Generator[float, None, None]:
    """Wiener process generator

    This produces a generator of the form W(0), W(dt), W(2dt), ..., where dt is
    the amount of time by which the process increments.

    The Wiener process W(t) is determined by the following properties:
    
    1. W(0) = 0
    2. W(t) has independent increments.
    3. W(t+u) - W(t) is normally distributed with mean 0 and variance u
    4. W has continuous paths.

    Parameters
    ----------
    increment_t: float
        the amount of time by which the process increments

    """
    normal_generator = np.random.normal
    wiener = 0.
    while True:
        yield wiener
        wiener += normal_generator(scale=np.sqrt(increment_t))


def brownian_motion(increment_t: float, 
                    origin_x: float, 
                    drift: float, 
                    scale: float) -> Generator[float, None, None]:
    """
    Brownian motion generator

    A Brownian motion satisfies the stochastic differential equation

        dB = mu dt + sigma dW

    where dW is the standard Wiener process, mu is a drift parameter, and sigma
    is a scale parameter.

    Parameters
    ----------
    increment_t: float
        the amount of time by which the process increments
    origin_x: float
        the initial value of the process
    drift: float
        the drift parameter of the process (mu)
    scale: float
        the scale parameter of the process (sigma)
    """
    wiener = wiener_process(increment_t)
    brownian = origin_x + next(wiener)
    while True:
        yield brownian
        brownian += drift * increment_t + scale * next(wiener)


def geometric_brownian_motion(increment_t: float, 
                              origin_x: float, 
                              drift: float, 
                              scale: float) -> Generator[float, None, None]:
    """
    Geometric Brownian motion generator

    A Geometric Brownian motioon satisfies the stochastic differential equation

        dG = mu G dt + sigma G dW

    where dW is the standard Wiener process, mu is a drift parameter, and sigma
    is a scale parameter.

    Parameters
    ----------
    increment_t: float
        the amount of time by which the process increments
    origin_x: float
        the initial value of the process
    drift: float
        the drift parameter of the process (mu)
    scale: float
        the scale parameter of the process (sigma)
    """
    brownian = brownian_motion(increment_t, 0., drift, scale)
    geometric = origin_x * np.exp(next(brownian))
    while True:
        yield geometric
        geometric = origin_x * np.exp(next(brownian))


# Asset class


class Asset:
    """
    """

    def __init__(self, initial_price: float, mean_return: float, stdev_return:
                 float, increment_t: float = 1., ema_decay: float = 0.9) -> None:
        """
        """
        self.initial_price   = initial_price
        self.mean_return     = mean_return
        self.stdev_return    = stdev_return
        self.increment_t     = increment_t
        self.ema_decay       = ema_decay  # ema == Exponential Moving Average
        self.__process       = None  # possibly want to not initialize until self.reset() 
                                   # is called explicitly by the user
        self.__avg_estimator = None
        self.__vol_estimator = None
        self.__price         = None
        self.__avg_price     = None
        self.__sqr_vol       = None
        self.__momentum      = None
        self.__ema_var       = None
        self.__bollinger     = (None, None)

        # lower bounds on asset statistics, to be used in environments
        self.lower_bounds = [0, -np.inf, -np.inf, -np.inf, 0, 0]

    @property
    def process(self) -> float:
        """
        Easy access to RNG, equipped with geometric brownian motion
        """
        return self.__process
        
    @property
    def price(self) -> float:
        """
        Most recent asset price
        """
        return self.__price

    @property
    def average_price(self) -> float:
        """
        Average price of the asset
        """
        return self.__avg_price

    @property
    def momentum(self) -> float:
        """]
        Exponential moving average
        """
        return self.__momentum

    @property
    def bollinger(self) -> Tuple[float, float]:
        """
        Exponential Bollinger bands
        """
        return self.__bollinger

    @property
    def volatility(self) -> float:
        """
        Volatility of the asset price
        """
        return np.sqrt(self.__sqr_vol)

    def reset(self) -> None:
        """
        Resets the asset price, momentum, and volatility measures
        """
        self.__process   = geometric_brownian_motion(self.increment_t,
                                                     self.initial_price,
                                                     self.mean_return,
                                                     self.stdev_return)
        self.__price     = next(self.__process)
        self.__momentum  = self.__price
        self.__ema_var   = 0.
        self.__bollinger = tuple(np.array([self.__momentum, self.__momentum]))

        # Average return and volatility processes
        self.__avg_estimator = moments.welford_estimator()
        self.__vol_estimator = moments.welford_estimator()

        self.__avg_price, _ = self.__avg_estimator(self.__price)
        self.__sqr_vol      = 0.

    def step(self) -> Tuple[float, float, Tuple[float, float]]:
        """
        Computes one step of the asset price change.

        Returns the updated price, momentum, and Bollinger bands
        """
        old_price        = self.__price
        old_momentum     = self.__momentum
        self.__price     = next(self.__process)
        self.__momentum  = self.ema_decay * self.__momentum + \
            (1. - self.ema_decay) * self.__price
        self.__ema_var  += (self.__price - old_momentum) * \
            (self.__price - self.__momentum)
        self.__bollinger = tuple(self.__momentum + \
                                 2. * np.sqrt(self.__ema_var) * np.array([-1., 1.]))

        self.__avg_price , _ = self.__avg_estimator(self.__price) 
        _, self.__sqr_vol    = self.__vol_estimator(np.log(self.__price / old_price))

        return {
            "price":     self.price,
            "momentum":  self.momentum,
            "bollinger": self.bollinger,
            "average price": self.average_price,
            "volatility": self.volatility
        }

    def summary(self):
        """
        Get a snapshot of the current state without updating the state at all.
        """

        return (self.price, self.momentum, *self.bollinger, self.average_price, self.volatility)

    def __len__(self):

        return len(self.lower_bounds)

    def __str__(self):
        """
        String representation
        """
        return f"Asset(price=${0 if self.price is None else self.price:0.2f})"

    def __repr__(self):
        """
        String representation, technically more detailed but at the moment the
        same as __str__
        """
        return self.__str__()



# Simplified Asset class

class SimpleAsset:
    """
    Simpler asset class tracking the price, average return, and average
    standard deviation of the return.

    NOTE: unlike in Asset above, volatility is defined to be the square
    root of the variance of the return.
    """

    def __init__(self, initial_price, mean_return, stdev_return,
                 increment_t=1):
        """
        """
        self.initial_price = initial_price
        self.mean_return = mean_return
        self.stdev_return = stdev_return
        self.increment_t = increment_t
        self.__process = None  # do not initialize until self.reset() called
        self.__return_estimator = None
        self.__price = None
        self.__avg_return = None
        self.__avg_variance = None

        self.__lower_bounds = [0, -np.inf, 0]

    @property
    def process(self):

        return self.__process
        
    @property
    def price(self):

        return self.__price

    @property
    def avg_return(self):

        return self.__avg_return

    @property
    def avg_variance(self):

        return self.__avg_variance

    @property
    def avg_volatility(self):

        return np.sqrt(self.__avg_variance)

    def _update_averages(self, return_over_previous=1):
        """Update the average return and variance."""

        self.__avg_return, self.__avg_variance = self.__return_estimator(
            return_over_previous)

    @property
    def lower_bounds(self):
        
        return self.__lower_bounds

    def reset(self):
        """
        Resets the asset price, return, and volatility measures

        self.reset() must be called to initialize the asset.
        """
        self.__process = geometric_brownian_motion(self.increment_t,
                                                     self.initial_price,
                                                     self.mean_return,
                                                     self.stdev_return)
        self.__price = next(self.__process)

        # Average return process
        self.__return_estimator = moments.WelfordEstimator()
        self._update_averages()

    def step(self):
        """
        Computes one step of the asset price change.

        Returns the updated price, average return, and average volatility.
        """
        old_price = self.__price
        self.__price = next(self.__process)
        self._update_averages((self.__price - old_price) / old_price)

        return {
            "price": self.price,
            "return": self.avg_return,
            "volatility": self.avg_volatility
        }

    def summary(self):
        """
        Get a snapshot of the current state without updating the state at all.
        """

        return (self.price, self.avg_return, self.avg_volatility)

    def __len__(self):
        """
        Return the number of asset statistics that are being kept track of.

        This will always be the same as the number of lower bounds.
        """
        
        return len(self.lower_bounds)

    def __str__(self):

        return f"Asset(price=${0 if self.price is None else self.price:0.2f})"

    def __repr__(self):
        """Same as __str__ at the moment."""
        
        return self.__str__()