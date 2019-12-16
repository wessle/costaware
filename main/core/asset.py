import numpy as np
from typing import Generator, Tuple


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
        self.initial_price = initial_price
        self.mean_return = mean_return
        self.stdev_return = stdev_return
        self.increment_t = increment_t
        self.ema_decay = ema_decay
        self.__process   = None  # possibly want to not initialize until self.reset() 
                                 # is called explicitly by the user
        self.__price     = None
        self.__momentum  = None
        self.__ema_var   = None
        self.__bollinger = (None, None)

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
    def momentum(self) -> float:
        """
        Exponential moving average
        """
        return self.__momentum

    @property
    def bollinger(self) -> Tuple[float, float]:
        """
        Exponential Bollinger bands
        """
        return self.__bollinger

    def reset(self) -> None:
        """
        Resets the asset price, momentum, and volatility measures
        """
        self.__process   = geometric_brownian_motion(0., self.increment_t,
                                                     self.initial_price,
                                                     self.mean_return,
                                                     self.stdev_return)
        self.__price     = next(self.__process)
        self.__momentum  = self.__price
        self.__ema_var   = 0.
        self.__bollinger = tuple(np.array([self.__momentum, self.__momentum]))

    def step(self) -> Tuple[float, float, Tuple[float, float]]:
        """
        Computes one step of the asset price change.

        Returns the updated price, momentum, and Bollinger bands
        """
        old_momentum     = self.__momentum
        self.__price     = next(self.__process)
        self.__momentum  = self.ema_decay * self.__momentum + (1. - self.ema_decay) * self.__price
        self.__ema_var  += (self.__price - old_momentum) * (self.__price - self.__momentum)
        self.__bollinger = tuple(self.__momentum + 2. * np.sqrt(self.__ema_var) * np.array([-1., 1.]))

        return self.price, self.momentum, self.bollinger

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

