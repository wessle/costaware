import numpy as np
from typing import Generator


# Core generator methods  
# 
# These are the underlying models needed for asset price simulations. 


def wiener_process(start_t: float, 
                   increment_t: float) -> Generator[float, None, None]:
    """
    Wiener process generator
    """
    wiener = 0.
    normal_generator = np.random.normal
    while True:
        yield wiener
        wiener += normal_generator(scale=np.sqrt(increment_t))


def brownian_motion(start_t: float, 
                    increment_t: float, 
                    origin_x: float, 
                    drift: float, 
                    scale: float) -> Generator[float, None, None]:
    """
    Brownian motion generator
    """
    wiener = wiener_process(start_t, increment_t)
    brownian = origin_x + next(wiener)
    while True:
        yield brownian
        brownian += drift * increment_t + scale * next(wiener)


def geometric_brownian_motion(start_t: float, 
                              increment_t: float, 
                              origin_x: float, 
                              drift: float, 
                              scale: float) -> Generator[float, None, None]:
    """
    Geometric Brownian motion generator
    """
    brownian = brownian_motion(start_t, increment_t, 0., drift, scale)
    geometric = origin_x * np.exp(next(brownian))
    while True:
        yield geometric
        geometric = np.exp(next(brownian))


class Asset:
    """
    """

    def __init__(self, initial_price, mean_return, stdev_return) -> None:
        """
        """
        self.initial_price = initial_price
        self.mean_return = mean_return
        self.stdev_return = stdev_return

    def reset(self) -> None:
        """
        """
        pass

    def one_step_return(self):
        """
        """
        pass


class Portfolio(Asset):
    """
    """

    def __init__(self):
        """
        """
        pass

    def reset(self):
        """
        """
        pass

    def one_step_return(self):
        """
        """
        pass
