import numpy as np
from typing import Generator, Tuple


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
        self.__bollinger = None

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



class Portfolio:
    """
    """

    def __init__(self, assets, weights, principal):
        """
        """
        self.__assets    = assets
        self.__weights   = weights
        self.__principal = principal

    @property
    def assets(self):
        return self.__assets

    @property 
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, new_weights):
        self.__weights = new_weights

    @property
    def principal(self):
        return self.__principal

    @principal.setter
    def principal(self, new_principal):
        self.__principal = new_principal

    @property
    def value(self):
        return self.principal * sum(asset.price * weight for asset, weight in zip(self.assets, self.weights))



    def reset(self):
        """
        Resets each asset in the portfolio
        """
        for asset in self.assets:
            asset.reset()

    def one_step_return(self):
        """
        """
        pass
